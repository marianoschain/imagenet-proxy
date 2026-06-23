"""Patch-based model for ImageNet-proxy pipeline.

Processes images as independent patches through residual blocks with
frozen random convolutions and trainable convolutions, then aggregates
and classifies.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchProcessingBlock(nn.Module):
    """Block that processes patches:
    - Random (frozen) 2D conv
    - Nonlinear activation
    - Trainable 2D conv (zero-initialized)
    """

    def __init__(self, features, kernel_size=3, activation=None,
                 train_random_conv=False):
        super().__init__()
        self.activation = activation or nn.ReLU()

        # Random conv: frozen by default, or trainable if train_random_conv=True.
        self.random_conv = nn.Conv2d(
            features, features, kernel_size,
            padding=kernel_size // 2, bias=True
        )
        if not train_random_conv:
            for param in self.random_conv.parameters():
                param.requires_grad = False

        # Trainable conv (zero-initialized for near-identity at start)
        self.trainable_conv = nn.Conv2d(
            features, features, kernel_size,
            padding=kernel_size // 2, bias=True
        )
        nn.init.zeros_(self.trainable_conv.weight)
        nn.init.zeros_(self.trainable_conv.bias)

    def forward(self, x):
        """
        Input: (batch*num_patches, features, H, W)
        Output: (batch*num_patches, features, H, W)
        """
        x = self.random_conv(x)
        x = self.activation(x)
        x = self.trainable_conv(x)
        return x


class PatchBasedModel(nn.Module):
    """Patch-based model for image classification.

    Processes images as independent patches through residual blocks,
    aggregates, and classifies.
    """

    def __init__(self, num_classes=10, num_blocks=4, hidden_features=128, patch_size=8,
                 train_random_conv=False, attn_pool=False, downsample=1, aux_heads=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.hidden_features = hidden_features
        self.patch_size = patch_size
        self.train_random_conv = train_random_conv
        self.attn_pool = attn_pool
        self.aux_heads = aux_heads

        # Initial projection per patch: (3, H, W) -> (hidden_features, H, W)
        self.initial_proj = nn.Conv2d(3, hidden_features, kernel_size=1, padding=0)

        # Optional spatial downsampling right after the projection, before the
        # (expensive) conv blocks. Since patches are mean-pooled over space at the
        # end anyway, running the blocks at lower resolution cuts FLOPs ~quadratically
        # with little effect on the pooled representation. ceil_mode keeps it valid
        # for patch sizes not divisible by the factor.
        self.downsample = (nn.AvgPool2d(downsample, ceil_mode=True)
                           if downsample > 1 else None)

        # Processing blocks with batch norm
        self.blocks = nn.ModuleList([
            PatchProcessingBlock(hidden_features, kernel_size=3,
                                 train_random_conv=train_random_conv)
            for _ in range(num_blocks)
        ])

        self.batch_norms = nn.ModuleList([
            nn.BatchNorm2d(hidden_features)
            for _ in range(num_blocks)
        ])

        # Final projection per patch
        self.final_proj = nn.Conv2d(hidden_features, hidden_features, kernel_size=1, padding=0)

        # Optional attention pooling: score each patch, softmax over patches, and
        # take a weighted sum instead of a plain mean. Lets the model focus on the
        # patches that contain the object and down-weight background — making it
        # robust to object scale/position rather than diluting signal across patches.
        if attn_pool:
            self.attn_score = nn.Linear(hidden_features, 1)

        # Classification head
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_features, num_classes)

        # Per-layer auxiliary heads for greedy layer-wise training: one classifier
        # (and, if attn pooling, one patch-scorer) attached to each block's output.
        if aux_heads:
            self.aux_classifiers = nn.ModuleList(
                [nn.Linear(hidden_features, num_classes) for _ in range(num_blocks)])
            self.aux_attn = (nn.ModuleList(
                [nn.Linear(hidden_features, 1) for _ in range(num_blocks)])
                if attn_pool else None)
        else:
            self.aux_classifiers = None
            self.aux_attn = None

    def _pool_layer(self, x, batch_size, num_patches, idx):
        """Pool a block's output (B*num_patches, F, ph, pw) to (B, F): mean over
        the patch's spatial dims, then mean/attention over patches."""
        _, C, ph, pw = x.shape
        x = x.reshape(batch_size, num_patches, C, ph, pw)
        x = x.mean(dim=(3, 4))  # (B, num_patches, F)
        if self.attn_pool:
            weights = torch.softmax(self.aux_attn[idx](x), dim=1)
            x = torch.sum(x * weights, dim=1)
        else:
            x = x.mean(dim=1)
        return x

    def layer_logits(self, x, up_to=None):
        """Run the stem + first `up_to` blocks and return a list of per-block
        logits from the auxiliary heads (used by greedy layer-wise training).
        With up_to=None, runs all blocks. The last entry is the deepest head."""
        assert self.aux_classifiers is not None, "build with aux_heads=True"
        n = self.num_blocks if up_to is None else up_to

        batch_size, _, H, W = x.shape
        patches = self._extract_patches(x, self.patch_size)
        batch_size, num_patches, _, patch_h, patch_w = patches.shape
        h = self.initial_proj(patches.reshape(batch_size * num_patches, 3, patch_h, patch_w))
        if self.downsample is not None:
            h = self.downsample(h)

        logits = []
        for i in range(n):
            residual = h
            h = self.blocks[i](h)
            h = self.batch_norms[i](h)
            h = F.relu(h + residual)
            pooled = self._pool_layer(h, batch_size, num_patches, i)
            logits.append(self.aux_classifiers[i](self.dropout(pooled)))
        return logits

    def forward(self, x):
        """
        Input: (batch_size, 3, 160, 160)
        Output: (batch_size, num_classes)
        """
        batch_size, _, H, W = x.shape

        # Extract patches: (B, 3, 160, 160) -> (B, num_patches, 3, patch_h, patch_w)
        patches = self._extract_patches(x, self.patch_size)
        batch_size, num_patches, _, patch_h, patch_w = patches.shape

        # Flatten patches for processing: (B*num_patches, 3, patch_h, patch_w)
        patches_flat = patches.reshape(batch_size * num_patches, 3, patch_h, patch_w)

        # Initial projection: (B*num_patches, 3, ph, pw) -> (B*num_patches, F, ph, pw)
        x = self.initial_proj(patches_flat)

        # Optional downsample before the conv blocks to save compute.
        if self.downsample is not None:
            x = self.downsample(x)

        # Process through blocks with residual connections
        for block, bn in zip(self.blocks, self.batch_norms):
            residual = x

            # Apply block
            x = block(x)

            # Batch norm
            x = bn(x)

            # Residual add + activation
            x = x + residual
            x = F.relu(x)

        # Final projection
        x = self.final_proj(x)

        # Reshape back to patch grid: (B*num_patches, C, ph, pw) -> (B, num_patches, C, ph, pw)
        _, C, ph, pw = x.shape
        x = x.reshape(batch_size, num_patches, C, ph, pw)

        # Aggregate patches by averaging spatial dimensions. Mean (not sum) keeps
        # feature magnitude independent of patch area, so logit scale — and thus
        # the right LR — stays comparable across patch sizes (8/16/32).
        # (B, num_patches, F, ph, pw) -> (B, num_patches, F)
        x = torch.mean(x, dim=(3, 4))  # Mean over patch spatial dims

        # Pool over patches: (B, num_patches, F) -> (B, F)
        if self.attn_pool:
            # Weight patches by a learned, softmaxed score, then sum.
            weights = torch.softmax(self.attn_score(x), dim=1)  # (B, num_patches, 1)
            x = torch.sum(x * weights, dim=1)
        else:
            x = torch.mean(x, dim=1)  # Plain average over patches

        # Dropout and classification
        x = self.dropout(x)
        x = self.classifier(x)

        return x

    def _extract_patches(self, x, patch_size):
        """
        Extract non-overlapping patches from image.

        Input: (B, C, H, W)
        Output: (B, num_patches, C, patch_h, patch_w)
        """
        B, C, H, W = x.shape
        assert H % patch_size == 0 and W % patch_size == 0, \
            f"Image dimensions {H}x{W} must be divisible by patch_size {patch_size}"

        num_patches_h = H // patch_size
        num_patches_w = W // patch_size

        # Reshape to create patches
        # (B, C, H, W) -> (B, C, num_patches_h, patch_size, num_patches_w, patch_size)
        x = x.reshape(B, C, num_patches_h, patch_size, num_patches_w, patch_size)

        # Permute to (B, num_patches_h, num_patches_w, patch_size, patch_size, C)
        x = x.permute(0, 2, 4, 3, 5, 1)

        # Flatten patch grid: (B, num_patches_h*num_patches_w, patch_size, patch_size, C)
        num_patches = num_patches_h * num_patches_w
        x = x.reshape(B, num_patches, patch_size, patch_size, C)

        # Rearrange to (B, num_patches, C, patch_h, patch_w) for PyTorch Conv2d
        x = x.permute(0, 1, 4, 2, 3)

        return x


def build_model(num_classes=10, patch_size=8, train_random_conv=False, num_blocks=4,
                attn_pool=False, hidden_features=128, downsample=1, aux_heads=False):
    """Single entry point used by train.py.

    Args:
        num_classes: Number of output classes (default: 10 for Imagenette)
        patch_size: Size of square patches (8, 16, or 32; default: 8)
        train_random_conv: If True, the per-block "random" convs are trainable
            instead of frozen (default: False).
        num_blocks: Number of residual processing blocks; trainable params scale
            ~linearly with this (default: 4).
        attn_pool: If True, pool patches with learned attention weights instead of
            a plain mean, letting the model focus on object-bearing patches
            (default: False).
        hidden_features: Channel width of the conv body. Compute scales ~with the
            square of this, so halving it (128 -> 64) cuts FLOPs ~4x (default: 128).
        downsample: Spatial downsampling factor applied before the conv blocks;
            >1 cuts per-block FLOPs ~quadratically (default: 1, i.e. none).
        aux_heads: If True, attach a classifier head to every block's output and
            expose layer_logits(); required for greedy layer-wise training
            (default: False).
    """
    return PatchBasedModel(num_classes=num_classes, patch_size=patch_size,
                           train_random_conv=train_random_conv, num_blocks=num_blocks,
                           attn_pool=attn_pool, hidden_features=hidden_features,
                           downsample=downsample, aux_heads=aux_heads)
