import argparse

import torch
from model import build_model

parser = argparse.ArgumentParser()
parser.add_argument("--num-classes", type=int, default=10,
                    help="Output classes (imagenette=10, imagenet-100=100, "
                         "tiny-imagenet=200).")
parser.add_argument("--img-size", type=int, default=160,
                    help="Square input size (imagenette/imagenet-100=160, "
                         "tiny-imagenet=64).")
parser.add_argument("--patch-size", type=int, default=8,
                    help="Square patch size; must divide --img-size.")
parser.add_argument("--num-blocks", type=int, default=4,
                    help="Number of residual processing blocks.")
parser.add_argument("--train-random-conv", action="store_true",
                    help="Train the per-block 'random' convs instead of freezing.")
parser.add_argument("--attn-pool", action="store_true",
                    help="Pool patches with learned attention weights.")
args = parser.parse_args()

# Create model
model = build_model(num_classes=args.num_classes, num_blocks=args.num_blocks,
                    patch_size=args.patch_size,
                    train_random_conv=args.train_random_conv,
                    attn_pool=args.attn_pool)
model.train()

# Create random batch and target
x = torch.randn(2, 3, args.img_size, args.img_size)
target = torch.randint(0, args.num_classes, (2,))

# Forward pass
output = model(x)
expected = (2, args.num_classes)
assert output.shape == expected, f"Expected output shape {expected}, got {output.shape}"

# Backward pass
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(output, target)
loss.backward()

# Print parameter counts
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
fixed_params = total_params - trainable_params
print(f"classes={args.num_classes} img={args.img_size} patch={args.patch_size} "
      f"blocks={args.num_blocks} | output {tuple(output.shape)}")
print(f"Parameter count: {total_params:,} "
      f"(trainable: {trainable_params:,} | fixed: {fixed_params:,})")
