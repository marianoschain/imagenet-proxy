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

from data import DATASETS, build_loaders
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
    p.add_argument("--dataset", type=str, default="imagenette", choices=DATASETS,
                   help="Which ImageNet proxy to train on. tiny-imagenet/imagenet-100 "
                        "load from the HF Hub (need the 'datasets' library). "
                        "tiny-imagenet is 64px — use --img-size 64 for it.")
    p.add_argument("--img-size", type=int, default=160)
    p.add_argument("--patch-size", type=int, default=8,
                   help="Square patch size; must divide --img-size (e.g. 8, 16, 32).")
    p.add_argument("--train-random-conv", action="store_true",
                   help="Train the per-block 'random' convs instead of freezing them.")
    p.add_argument("--num-blocks", type=int, default=4,
                   help="Number of residual processing blocks; trainable params "
                        "scale ~linearly with this.")
    p.add_argument("--hidden-features", type=int, default=128,
                   help="Conv channel width. Compute scales ~with its square, so "
                        "64 cuts FLOPs ~4x vs the 128 default.")
    p.add_argument("--downsample", type=int, default=1,
                   help="Spatial downsample factor before the conv blocks; >1 cuts "
                        "per-block FLOPs ~quadratically (e.g. 2 -> ~4x less).")
    p.add_argument("--attn-pool", action="store_true",
                   help="Pool patches with learned attention weights instead of a "
                        "plain mean (focuses on object-bearing patches).")
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader workers. Raise (e.g. 8) on a fast GPU so data "
                        "loading doesn't starve it.")
    p.add_argument("--compile", action="store_true",
                   help="Wrap the model in torch.compile for a faster compute path "
                        "(first step is slow while it compiles).")
    p.add_argument("--channels-last", action="store_true",
                   help="Use channels_last memory format (better tensor-core use).")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--out-repo", type=str, required=True,
                   help="HF repo id for checkpoints, e.g. you/imagenet-proxy-ckpts")
    p.add_argument("--no-resume", action="store_true",
                   help="Ignore any existing checkpoint and start fresh.")
    p.add_argument("--save-every", type=int, default=5,
                   help="Upload checkpoints every N epochs (plus the final epoch). "
                        "Each upload is one HF commit and HF caps commits at "
                        "128/hour, so keep this > 1 for fast epochs.")
    p.add_argument("--tensorboard", action="store_true",
                   help="Log loss/acc/lr curves to --logdir for TensorBoard.")
    p.add_argument("--logdir", type=str, default="runs",
                   help="Directory for TensorBoard event files.")
    p.add_argument("--run-name", type=str, default="run",
                   help="Names this experiment's checkpoints, e.g. resnet_a -> "
                        "resnet_a_last.pt / resnet_a_best.pt. Use a fresh name for a "
                        "new architecture so runs don't overwrite each other.")
    return p.parse_args()


def train_one_epoch(model, loader, optimizer, scaler, device, use_amp, epoch,
                    channels_last=False):
    model.train()
    running, seen = 0.0, 0
    mem_fmt = torch.channels_last if channels_last else torch.contiguous_format
    # leave=False makes this bar erase itself when the epoch ends, so only the
    # one-line summary printed in main() persists — no scroll buildup.
    pbar = tqdm(loader, desc=f"epoch {epoch}", leave=False, dynamic_ncols=True)
    for images, targets in pbar:
        images = images.to(device, non_blocking=True, memory_format=mem_fmt)
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
def validate(model, loader, device, channels_last=False):
    model.eval()
    mem_fmt = torch.channels_last if channels_last else torch.contiguous_format
    correct, total = 0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True, memory_format=mem_fmt)
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
        root=args.data_root, dataset=args.dataset, img_size=args.img_size,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    print(f"Dataset: {args.dataset} | classes: {num_classes} | img_size: {args.img_size}")
    model = build_model(num_classes=num_classes, patch_size=args.patch_size,
                        train_random_conv=args.train_random_conv,
                        num_blocks=args.num_blocks, attn_pool=args.attn_pool,
                        hidden_features=args.hidden_features,
                        downsample=args.downsample).to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
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

    # Compile after loading so checkpoints stay keyed to the original module
    # (torch.compile shares the same parameters, so `model` remains the source of
    # truth for state_dict save/load and the optimizer).
    fwd_model = torch.compile(model) if args.compile else model

    best_state = None
    best_dirty = False
    for epoch in range(start_epoch, args.epochs):
        loss = train_one_epoch(fwd_model, train_loader, optimizer, scaler,
                               device, use_amp, epoch, channels_last=args.channels_last)
        acc = validate(fwd_model, val_loader, device, channels_last=args.channels_last)
        scheduler.step()
        lr = scheduler.get_last_lr()[0]
        print(f"epoch {epoch:3d} | loss {loss:.4f} | val_acc {acc:.4f} | lr {lr:.2e}")
        if writer:
            writer.add_scalar("train/loss", loss, epoch)
            writer.add_scalar("val/acc", acc, epoch)
            writer.add_scalar("lr", lr, epoch)

        # Track the best weights in memory; defer the upload to the throttle below
        # so we stay under HF's 128-commits/hour limit.
        if acc > best_acc:
            best_acc = acc
            best_state = {
                "epoch": epoch,
                "model": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "best_acc": best_acc,
            }
            best_dirty = True

        # Upload only every --save-every epochs (and on the final epoch).
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            state = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict(),
                "best_acc": best_acc,
            }
            try:
                ckpt.save(state, last_ckpt,
                          commit_message=f"{run} epoch {epoch} acc={acc:.4f}")
                if best_dirty:
                    ckpt.save(best_state, best_ckpt,
                              commit_message=f"{run} best epoch {best_state['epoch']} "
                                             f"acc={best_acc:.4f}")
                    best_dirty = False
            except Exception as e:  # don't let a transient rate-limit kill training
                print(f"  [warn] checkpoint upload skipped: {e}")

    if writer:
        writer.close()
    print(f"Done. Best val_acc={best_acc:.4f}")


if __name__ == "__main__":
    main()
