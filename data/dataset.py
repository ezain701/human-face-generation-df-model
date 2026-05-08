import os
import csv
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


def build_image_transform(image_size, augment_level="basic", train=True):
    """Build image transforms for aligned face generation."""
    normalize = transforms.Normalize([0.5], [0.5])

    if not train or augment_level == "none":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ])

    if augment_level == "basic":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

    if augment_level == "face_safe":
        return transforms.Compose([
            transforms.Resize((image_size + 12, image_size + 12)),
            transforms.RandomResizedCrop(image_size, scale=(0.9, 1.0), ratio=(0.96, 1.04)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.08, hue=0.02)
            ], p=0.6),
            transforms.RandomAffine(degrees=5, translate=(0.03, 0.03), scale=(0.96, 1.04)),
            transforms.ToTensor(),
            normalize,
        ])

    if augment_level == "strong":
        return transforms.Compose([
            transforms.Resize((image_size + 24, image_size + 24)),
            transforms.RandomResizedCrop(image_size, scale=(0.82, 1.0), ratio=(0.92, 1.08)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.22, contrast=0.22, saturation=0.18, hue=0.03)
            ], p=0.75),
            transforms.RandomAffine(degrees=8, translate=(0.05, 0.05), scale=(0.92, 1.08)),
            transforms.RandomGrayscale(p=0.03),
            transforms.ToTensor(),
            normalize,
        ])

    raise ValueError(f"Unknown augment_level: {augment_level}")


def load_caption_map(caption_file):
    """Load captions keyed by filename or stem from JSON, JSONL, CSV, or TXT."""
    if caption_file is None:
        return {}

    ext = os.path.splitext(caption_file)[1].lower()
    captions = {}

    if ext == ".json":
        with open(caption_file) as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return {str(key): str(value) for key, value in data.items()}
        for row in data:
            captions[str(row["filename"])] = str(row["prompt"])
        return captions

    if ext == ".jsonl":
        with open(caption_file) as handle:
            for line in handle:
                if line.strip():
                    row = json.loads(line)
                    captions[str(row["filename"])] = str(row["prompt"])
        return captions

    if ext == ".csv":
        with open(caption_file, newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                filename = row.get("filename") or row.get("file") or row.get("image")
                prompt = row.get("prompt") or row.get("caption") or row.get("text")
                if filename and prompt:
                    captions[str(filename)] = str(prompt)
        return captions

    with open(caption_file) as handle:
        for line in handle:
            parts = line.strip().split(maxsplit=1)
            if len(parts) == 2:
                captions[parts[0]] = parts[1]
    return captions


def caption_for_path(image_path, caption_map, default_caption):
    filename = os.path.basename(image_path)
    stem = os.path.splitext(filename)[0]
    return caption_map.get(filename, caption_map.get(stem, default_caption))


class CelebAHQDataset(Dataset):
    """Dataset class for CelebA-HQ 256x256 images."""

    def __init__(
        self,
        image_dir,
        image_size=256,
        split="train",
        train_size=27000,
        caption_file=None,
        default_caption="a portrait photo of a human face",
        return_captions=False,
        augment_level="basic",
    ):
        self.image_dir = image_dir
        self.caption_map = load_caption_map(caption_file)
        self.default_caption = default_caption
        self.return_captions = return_captions
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if split == "train":
            self.image_paths = self.image_paths[:train_size]
        elif split == "test":
            self.image_paths = self.image_paths[train_size:train_size + 300]

        self.transform = build_image_transform(
            image_size,
            augment_level=augment_level,
            train=split == "train",
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(img)
        if self.return_captions:
            return image, caption_for_path(self.image_paths[idx], self.caption_map, self.default_caption)
        return image


class ButterflyDataset(Dataset):
    """Dataset class for the butterfly toy model."""

    def __init__(
        self,
        image_dir,
        image_size=128,
        caption_file=None,
        default_caption="a butterfly",
        return_captions=False,
        augment_level="basic",
    ):
        self.image_dir = image_dir
        self.caption_map = load_caption_map(caption_file)
        self.default_caption = default_caption
        self.return_captions = return_captions
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        self.transform = build_image_transform(
            image_size,
            augment_level=augment_level,
            train=True,
        )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        image = self.transform(img)
        if self.return_captions:
            return image, caption_for_path(self.image_paths[idx], self.caption_map, self.default_caption)
        return image


def get_dataloader(dataset, batch_size=16, shuffle=True, num_workers=2):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
