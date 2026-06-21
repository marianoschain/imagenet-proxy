"""Training entry point for the ImageNet-proxy pipeline.

Run in Colab (after putting your HF token in the environment so the train.py
subprocess can read it):

    import os
    from google.colab import userdata
    os.environ["HF_TOKEN"] = userdata.get("HF_TOKEN")
    !python train.py --epochs 50 --out-repo YOUR_USERNAME/imagenet-proxy-ckpts
"""
import argparse
import os

import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler

from data import build_loaders
from model import build_model
from hf_checkpoint import HFCheckpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=160)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--out-repo", type=str, required=True,
                   help="HF repo id for checkpoints, e.g. you/imagenet-proxy-ckpts")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore any existing checkpoint and start fresh.")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp):
    model.train()
    running, seen = 0.0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=use_amp):
            logits = model(images)
            loss = nn.functional.cross_entropy(logits, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running += loss.item() * images.size(0)
        seen += images.size(0)
    return running / max(seen, 1)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        preds = model(images).argmax(1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return correct / max(total, 1)


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = device == "cuda"
    print(f"Device: {device} | AMP: {use_amp}")

    token = os.environ.get("HF_TOKEN")
    if not token:
        raise SystemExit(
            "HF_TOKEN not set. In the Colab cell before running this script:\n"
            "    os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')"
        )
    ckpt = HFCheckpoint(repo_id=args.out_repo, token=token)

    train_loader, val_loader, num_classes = build_loaders(
        root=args.data_root, img_size=args.img_size,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    model = build_model(num_classes=num_classes).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler("cuda", enabled=use_amp)

    start_epoch, best_acc = 0, 0.0
    if not args.no_resume:
        state = ckpt.load("last.pt", map_location=device)
        if state is not None:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            start_epoch = state["epoch"] + 1
            best_acc = state.get("best_acc", 0.0)
            print(f"Resumed at epoch {start_epoch} | best_acc={best_acc:.4f}")
        else:
            print("No checkpoint found — starting fresh.")

    for epoch in range(start_epoch, args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, scaler, device, use_amp)
        acc = validate(model, val_loader, device)
        scheduler.step()
        print(f"epoch {epoch:3d} | loss {loss:.4f} | val_acc {acc:.4f} | "
              f"lr {scheduler.get_last_lr()[0]:.2e}")

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_acc": max(best_acc, acc),
        }
        ckpt.save(state, "last.pt", commit_message=f"epoch {epoch} acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            ckpt.save(state, "best.pt", commit_message=f"best epoch {epoch} acc={acc:.4f}")

    print(f"Done. Best val_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
