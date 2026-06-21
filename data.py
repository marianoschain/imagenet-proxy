"""ImageNet-proxy data loaders.

Supports three drop-in proxies of increasing size (select with `dataset=`):
  - "imagenette"     10 classes,  ~9.5k imgs, full res (torchvision)
  - "tiny-imagenet"  200 classes, 100k imgs, 64x64    (Hugging Face Hub)
  - "imagenet-100"   100 classes, ~126k imgs, full res (Hugging Face Hub)

All use the standard ImageNet transform recipe, so nothing changes conceptually
when you scale up. `build_loaders` returns the dataset's own num_classes so the
model head is sized correctly.
"""
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Datasets pulled from the Hugging Face Hub. default_img_size is just a sensible
# starting resolution; the actual size is whatever --img-size is passed.
HF_DATASETS = {
    "tiny-imagenet": {"repo": "zh-plus/tiny-imagenet", "default_img_size": 64},
    "imagenet-100": {"repo": "clane9/imagenet-100", "default_img_size": 160},
}
DATASETS = ["imagenette", *HF_DATASETS]


def _to_rgb(img):
    # A handful of source images may be grayscale; force 3 channels.
    # Defined at module level (not a lambda) so it pickles for DataLoader workers.
    return img.convert("RGB")


def _make_transforms(img_size):
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
    return train_tf, val_tf


def _build_imagenette(root, split, transform):
    # torchvision raises if download=True and the data already exists, so fall back.
    try:
        return datasets.Imagenette(root, split=split, size="160px",
                                   download=True, transform=transform)
    except RuntimeError:
        return datasets.Imagenette(root, split=split, size="160px",
                                   download=False, transform=transform)


class _HFImageDataset(Dataset):
    """Wraps a Hugging Face image dataset so it applies a torchvision transform
    and yields (tensor, label) tuples like a torchvision dataset."""

    def __init__(self, hf_split, transform):
        self.ds = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        return self.transform(item["image"]), item["label"]


def _resolve_val_split(splits):
    """Pick the validation-like split from an HF dataset's available splits."""
    for name in ("validation", "valid", "val", "test"):
        if name in splits:
            return name
    others = [s for s in splits if s != "train"]
    if others:
        return others[0]
    raise ValueError(f"No validation split among {list(splits)}")


def _build_hf_datasets(dataset, root, train_tf, val_tf):
    try:
        from datasets import load_dataset
    except ImportError as e:
        raise ImportError(
            f"The '{dataset}' proxy needs the 'datasets' library. "
            "Install it with: pip install datasets"
        ) from e

    repo = HF_DATASETS[dataset]["repo"]
    ds = load_dataset(repo, cache_dir=root)
    val_split = _resolve_val_split(ds)

    label_feat = ds["train"].features["label"]
    num_classes = getattr(label_feat, "num_classes", None)
    if num_classes is None:
        num_classes = len(set(ds["train"]["label"]))

    train_ds = _HFImageDataset(ds["train"], train_tf)
    val_ds = _HFImageDataset(ds[val_split], val_tf)
    return train_ds, val_ds, num_classes


def build_loaders(root="./data", dataset="imagenette", img_size=160,
                  batch_size=64, num_workers=2):
    if dataset not in DATASETS:
        raise ValueError(f"Unknown dataset '{dataset}'; choose from {DATASETS}")

    train_tf, val_tf = _make_transforms(img_size)

    if dataset == "imagenette":
        train_ds = _build_imagenette(root, "train", train_tf)
        val_ds = _build_imagenette(root, "val", val_tf)
        num_classes = 10
    else:
        train_ds, val_ds, num_classes = _build_hf_datasets(
            dataset, root, train_tf, val_tf)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    if num_workers > 0:
        # Keep workers alive between epochs and let them stage batches ahead so the
        # GPU isn't starved waiting on JPEG decode + augmentation.
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 4

    train_loader = DataLoader(train_ds, shuffle=True, drop_last=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, num_classes
