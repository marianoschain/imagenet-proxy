import argparse

import torch
from model import build_model

parser = argparse.ArgumentParser()
parser.add_argument("--num-blocks", type=int, default=4,
                    help="Number of residual processing blocks.")
args = parser.parse_args()

# Create model
model = build_model(num_classes=10, num_blocks=args.num_blocks)
model.train()

# Create random batch and target
x = torch.randn(2, 3, 160, 160)
target = torch.randint(0, 10, (2,))

# Forward pass
output = model(x)
assert output.shape == (2, 10), f"Expected output shape (2, 10), got {output.shape}"

# Backward pass
loss_fn = torch.nn.CrossEntropyLoss()
loss = loss_fn(output, target)
loss.backward()

# Print parameter counts
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
fixed_params = total_params - trainable_params
print(f"Parameter count: {total_params:,} "
      f"(trainable: {trainable_params:,} | fixed: {fixed_params:,})")
