import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image


class CelebAHQDataset(Dataset):
    """Dataset class for CelebA-HQ 256x256 images."""

    def __init__(self, image_dir, image_size=256, split="train", train_size=2700):
        self.image_dir = image_dir
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        if split == "train":
            self.image_paths = self.image_paths[:train_size]
        elif split == "test":
            self.image_paths = self.image_paths[train_size:train_size + 300]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),  # Scale to [-1, 1]
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


class ButterflyDataset(Dataset):
    """Dataset class for the butterfly toy model."""

    def __init__(self, image_dir, image_size=128):
        self.image_dir = image_dir
        self.image_paths = sorted([
            os.path.join(image_dir, f)
            for f in os.listdir(image_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(img)


def get_dataloader(dataset, batch_size=16, shuffle=True, num_workers=2):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
