"""
training/dataset_aptos.py
PyTorch Dataset class for the APTOS 2019 Blindness Detection dataset.

Expected directory layout after download and unzip:
    data/aptos2019/
        train_images/
            0a09aa7356c0.png
            0a29aac6e086.png
            ...
        train.csv          # columns: id_code, diagnosis

The 'diagnosis' column contains integer DR grades: 0, 1, 2, 3, or 4.

Usage:
    from training.dataset_aptos import APTOSDataset
    dataset = APTOSDataset(
        csv_path="data/aptos2019/train.csv",
        images_dir="data/aptos2019/train_images",
        transform=train_transform,
    )
"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class APTOSDataset(Dataset):
    """
    PyTorch Dataset for the APTOS 2019 5-class DR grading task.

    Attributes:
        records     (pd.DataFrame) Filtered DataFrame with (id_code, diagnosis).
        images_dir  (str)          Path to the folder containing training images.
        transform   (callable)     Torchvision transform pipeline applied to each image.
    """

    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        transform=None,
    ):
        """
        Args:
            csv_path:   Path to train.csv.
            images_dir: Path to the folder containing retinal images.
            transform:  Torchvision transform pipeline. If None, only ToTensor is applied.
        """
        df = pd.read_csv(csv_path)

        # Keep only rows where the diagnosis is a valid grade 0–4
        self.records = df[df["diagnosis"].isin([0, 1, 2, 3, 4])].reset_index(drop=True)
        self.images_dir = images_dir
        self.transform = transform or transforms.Compose([transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            image (torch.Tensor): Transformed image tensor.
            label (int):          DR grade label (0–4).
        """
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
    """
    Return the appropriate torchvision transform pipeline for APTOS 2019.

    Training transforms include augmentation:
        - Random horizontal flip
        - Random rotation ±15°
        - Colour jitter (brightness, contrast, saturation)
        - Resize to 224×224
        - ImageNet normalisation

    Validation/test transforms skip augmentation but still resize and normalise.

    Args:
        train: If True, return training transforms with augmentation.

    Returns:
        transforms.Compose object.
    """
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
