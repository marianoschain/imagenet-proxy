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
                 train_random_conv=False):
        super().__init__()
        self.num_classes = num_classes
        self.num_blocks = num_blocks
        self.hidden_features = hidden_features
        self.patch_size = patch_size
        self.train_random_conv = train_random_conv

        # Initial projection per patch: (3, H, W) -> (hidden_features, H, W)
        self.initial_proj = nn.Conv2d(3, hidden_features, kernel_size=1, padding=0)

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

        # Classification head
        self.dropout = nn.Dropout(0.5)
        self.classifier = nn.Linear(hidden_features, num_classes)

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

        # Aggregate patches by summing spatial dimensions
        # (B, num_patches, F, ph, pw) -> (B, num_patches, F)
        x = torch.sum(x, dim=(3, 4))  # Sum over patch spatial dims

        # Average over patches
        # (B, num_patches, F) -> (B, F)
        x = torch.mean(x, dim=1)

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


def build_model(num_classes=10, patch_size=8, train_random_conv=False):
    """Single entry point used by train.py.

    Args:
        num_classes: Number of output classes (default: 10 for Imagenette)
        patch_size: Size of square patches (8, 16, or 32; default: 8)
        train_random_conv: If True, the per-block "random" convs are trainable
            instead of frozen (default: False).
    """
    return PatchBasedModel(num_classes=num_classes, patch_size=patch_size,
                           train_random_conv=train_random_conv)
