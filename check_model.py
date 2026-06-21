import argparse

import torch
from model import build_model

# Per-dataset shape configs: (num_classes, img_size, patch_size).
DATASET_CONFIGS = {
    "imagenette":    {"num_classes": 10,  "img_size": 160, "patch_size": 16},
    "imagenet-100":  {"num_classes": 100, "img_size": 160, "patch_size": 16},
    "tiny-imagenet": {"num_classes": 200, "img_size": 64,  "patch_size": 8},
}

parser = argparse.ArgumentParser()
parser.add_argument("--num-blocks", type=int, default=4,
                    help="Number of residual processing blocks.")
parser.add_argument("--hidden-features", type=int, default=128,
                    help="Conv channel width (compute scales ~with its square).")
parser.add_argument("--downsample", type=int, default=1,
                    help="Spatial downsample factor before the conv blocks.")
parser.add_argument("--train-random-conv", action="store_true",
                    help="Train the per-block 'random' convs instead of freezing.")
parser.add_argument("--attn-pool", action="store_true",
                    help="Pool patches with learned attention weights.")
args = parser.parse_args()


def check(name, cfg):
    model = build_model(num_classes=cfg["num_classes"], num_blocks=args.num_blocks,
                        patch_size=cfg["patch_size"],
                        train_random_conv=args.train_random_conv,
                        attn_pool=args.attn_pool,
                        hidden_features=args.hidden_features,
                        downsample=args.downsample)
    model.train()

    x = torch.randn(2, 3, cfg["img_size"], cfg["img_size"])
    target = torch.randint(0, cfg["num_classes"], (2,))

    output = model(x)
    expected = (2, cfg["num_classes"])
    assert output.shape == expected, \
        f"[{name}] expected output shape {expected}, got {output.shape}"

    loss = torch.nn.CrossEntropyLoss()(output, target)
    loss.backward()

    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"{name:14s} classes={cfg['num_classes']:3d} img={cfg['img_size']:3d} "
          f"patch={cfg['patch_size']:2d} | output {tuple(output.shape)} | "
          f"params {total:,} (trainable {trainable:,} | fixed {total - trainable:,})")


print(f"blocks={args.num_blocks} train_random_conv={args.train_random_conv} "
      f"attn_pool={args.attn_pool}")
for name, cfg in DATASET_CONFIGS.items():
    check(name, cfg)
print("All datasets OK.")
