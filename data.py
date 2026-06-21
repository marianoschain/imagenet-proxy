"""Imagenette data loaders — a 10-class ImageNet proxy (real ImageNet images).

Imagenette is a small, easy subset of ImageNet that downloads in seconds and trains
in minutes, so you can validate the whole pipeline before scaling up. The transforms
mirror the standard ImageNet recipe so nothing changes conceptually when you move to
a larger dataset.
"""
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _to_rgb(img):
    # A handful of source images may be grayscale; force 3 channels.
    # Defined at module level (not a lambda) so it pickles for DataLoader workers.
    return img.convert("RGB")


def _build_dataset(root, split, size, transform):
    # torchvision raises if download=True and the data already exists, so fall back.
    try:
        return datasets.Imagenette(root, split=split, size=size,
                                   download=True, transform=transform)
    except RuntimeError:
        return datasets.Imagenette(root, split=split, size=size,
                                   download=False, transform=transform)


def build_loaders(root="./data", img_size=160, batch_size=64, num_workers=2):
    train_tf = transforms.Compose([
        transforms.Lambda(_to_rgb),
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    val_tf = transforms.Compose([
        transforms.Lambda(_to_rgb),
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    train_ds = _build_dataset(root, "train", "160px", train_tf)
    val_ds = _build_dataset(root, "val", "160px", val_tf)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, 10
