# Dataset loader for APTOS 2019 Blindness Detection.
#
# Expected layout:
#   data/aptos2019/
#       train_images/   (PNG files)
#       train.csv       (columns: id_code, diagnosis)
#
# The 'diagnosis' column holds DR grades 0–4.

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class APTOSDataset(Dataset):
    """APTOS 2019 DR grading dataset — 5 classes (grades 0–4)."""

    def __init__(self, csv_path: str, images_dir: str, transform=None):
        df = pd.read_csv(csv_path)

        # Keep only rows where the diagnosis is a valid grade 0–4
        self.records = df[df["diagnosis"].isin([0, 1, 2, 3, 4])].reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform or transforms.Compose([transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple:
        row = self.records.iloc[idx]
        image_id = row["id_code"]
        label = int(row["diagnosis"])

        # Try .png first, then .jpg (APTOS images are PNG)
        img_path = os.path.join(self.images_dir, f"{image_id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.images_dir, f"{image_id}.jpg")

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)

        return image, label


def get_aptos_transforms(train: bool = True):
    """Training transforms include flip, rotation, and colour jitter. Val/test only resize and normalise."""
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    if train:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
