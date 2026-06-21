import torch
from model import build_model

# Create model
model = build_model(num_classes=10)
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

# Print parameter count
total_params = sum(p.numel() for p in model.parameters())
print(f"Parameter count: {total_params:,}")
