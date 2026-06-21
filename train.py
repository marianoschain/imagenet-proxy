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
from tqdm.auto import tqdm
from huggingface_hub.utils import disable_progress_bars

from data import build_loaders
from model import build_model
from hf_checkpoint import HFCheckpoint

# Stop huggingface_hub from printing an upload progress bar on every checkpoint save.
disable_progress_bars()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=160)
    p.add_argument("--patch-size", type=int, default=8,
                   help="Square patch size; must divide --img-size (e.g. 8, 16, 32).")
    p.add_argument("--train-random-conv", action="store_true",
                   help="Train the per-block 'random' convs instead of freezing them.")
    p.add_argument("--num-blocks", type=int, default=4,
                   help="Number of residual processing blocks; trainable params "
                        "scale ~linearly with this.")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--out-repo", type=str, required=True,
                   help="HF repo id for checkpoints, e.g. you/imagenet-proxy-ckpts")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore any existing checkpoint and start fresh.")
    p.add_argument("--tensorboard", action="store_true",
                   help="Log loss/acc/lr curves to --logdir for TensorBoard.")
    p.add_argument("--logdir", type=str, default="runs",
                   help="Directory for TensorBoard event files.")
    p.add_argument("--run-name", type=str, default="run",
                   help="Names this experiment's checkpoints, e.g. resnet_a -> "
                        "resnet_a_last.pt / resnet_a_best.pt. Use a fresh name for a "
                        "new architecture so runs don't overwrite each other.")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp, epoch):
    model.train()
    running, seen = 0.0, 0
    # leave=False makes this bar erase itself when the epoch ends, so only the
    # one-line summary printed in main() persists — no scroll buildup.
    pbar = tqdm(loader, desc=f"epoch {epoch}", leave=False, dynamic_ncols=True)
    for images, targets in pbar:
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
        pbar.set_postfix(loss=f"{running / seen:.3f}")
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
    run = args.run_name.strip().replace(" ", "_")
    last_ckpt, best_ckpt = f"{run}_last.pt", f"{run}_best.pt"
    print(f"Run: {run} | Device: {device} | AMP: {use_amp}")

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
    model = build_model(num_classes=num_classes, patch_size=args.patch_size,
                        train_random_conv=args.train_random_conv,
                        num_blocks=args.num_blocks).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler("cuda", enabled=use_amp)

    writer = None
    if args.tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(os.path.join(args.logdir, run))
            print(f"TensorBoard logging to ./{args.logdir}/{run}")
        except Exception as e:  # never let logging break training
            print(f"TensorBoard logging disabled ({e})")

    start_epoch, best_acc = 0, 0.0
    if not args.no_resume:
        state = ckpt.load(last_ckpt, map_location=device)
        if state is not None:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            scaler.load_state_dict(state["scaler"])
            start_epoch = state["epoch"] + 1
            best_acc = state.get("best_acc", 0.0)
            print(f"Resumed '{run}' at epoch {start_epoch} | best_acc={best_acc:.4f}")
        else:
            print(f"No checkpoint for '{run}' — starting fresh.")

    for epoch in range(start_epoch, args.epochs):
        loss = train_one_epoch(model, train_loader, optimizer, scaler,
                               device, use_amp, epoch)
        acc = validate(model, val_loader, device)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print(f"epoch {epoch:3d} | loss {loss:.4f} | val_acc {acc:.4f} | lr {lr:.2e}")
        if writer:
            writer.add_scalar("train/loss", loss, epoch)
            writer.add_scalar("val/acc", acc, epoch)
            writer.add_scalar("lr", lr, epoch)

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_acc": max(best_acc, acc),
        }
        ckpt.save(state, last_ckpt, commit_message=f"{run} epoch {epoch} acc={acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            ckpt.save(state, best_ckpt,
                      commit_message=f"{run} best epoch {epoch} acc={acc:.4f}")

    if writer:
        writer.close()
    print(f"Done. Best val_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
