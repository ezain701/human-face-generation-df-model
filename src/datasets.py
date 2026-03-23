from pathlib import Path

from datasets import load_dataset
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms


class HFDatasetWrapper(Dataset):
    def __init__(self, hf_dataset, transform):
        self.ds = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        item = self.ds[idx]
        img = item["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        return self.transform(img)


class LocalImageFolderDataset(Dataset):
    def __init__(self, root_dir: str, transform):
        self.root = Path(root_dir)
        self.files = sorted([p for p in self.root.glob("**/*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
        if not self.files:
            raise FileNotFoundError(f"No images found under {root_dir}")
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        return self.transform(img)


def build_transform(image_size: int):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])


def get_dataloaders(cfg: dict):
    ds_cfg = cfg["dataset"]
    image_size = ds_cfg["image_size"]
    batch_size = ds_cfg["batch_size"]
    num_workers = ds_cfg.get("num_workers", 2)
    transform = build_transform(image_size)

    if ds_cfg["name"] == "butterflies":
        train_ds = load_dataset("huggan/smithsonian_butterflies_subset", split="train")
        dataset = HFDatasetWrapper(train_ds, transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        return loader, None

    if ds_cfg["name"] == "celeba_hq_local":
        dataset = LocalImageFolderDataset(ds_cfg["data_dir"], transform)
        train_size = ds_cfg.get("train_size", 2700)
        test_size = ds_cfg.get("test_size", 300)
        if train_size + test_size > len(dataset):
            raise ValueError(f"Requested split {train_size}+{test_size} exceeds dataset size {len(dataset)}")
        remainder = len(dataset) - train_size - test_size
        train_set, test_set, _ = random_split(
            dataset,
            [train_size, test_size, remainder],
            generator=torch.Generator().manual_seed(42),
        )
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
        test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
        return train_loader, test_loader

    raise ValueError(f"Unknown dataset name: {ds_cfg['name']}")
