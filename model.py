"""Custom model architecture for the ImageNet-proxy pipeline.

This is a from-scratch small residual network — a working starting point that you
replace with your own architecture. Keep the `build_model(num_classes)` entry point
so train.py needs no changes when you swap the internals.
"""
import torch.nn as nn


class BasicBlock(nn.Module):
    """A standard two-conv residual block with an optional projection shortcut."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class SmallResNet(nn.Module):
    """A compact ResNet. Edit the stem, widths, or block counts — this is *your*
    architecture, written explicitly rather than imported, so you can change it freely.
    """

    def __init__(self, num_classes=10, widths=(64, 128, 256), blocks=(2, 2, 2)):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
        )
        layers = []
        in_ch = widths[0]
        for w, n in zip(widths, blocks):
            for i in range(n):
                # Downsample at the first block of every stage except the first.
                stride = 2 if (i == 0 and w != widths[0]) else 1
                layers.append(BasicBlock(in_ch, w, stride))
                in_ch = w
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, num_classes),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        return self.head(x)


def build_model(num_classes=10):
    """Single entry point used by train.py. Swap the returned module for your own."""
    return SmallResNet(num_classes=num_classes)
