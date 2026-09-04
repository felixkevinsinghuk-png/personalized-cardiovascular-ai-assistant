"""
training/dataset_odir.py
PyTorch Dataset class for the ODIR-5K Ocular Disease Recognition dataset.

Expected directory layout after download and unzip:
    data/odir5k/
        ODIR-5K_Training_Images/
            0_left.jpg
            0_right.jpg
            ...
        full_df.csv      # columns include: Left-Fundus, Right-Fundus, and disease columns

ODIR-5K provides images for both eyes per patient. For CVD risk prediction,
we use the hypertension column as a proxy for hypertensive retinopathy.

Label extraction strategy:
- The CSV has separate columns for each eye (Left-Fundus, Right-Fundus)
- Disease labels are stored as keywords in columns like N, D, G, C, A, H, M, O
- We use the 'H' column (Hypertension / Hypertensive Retinopathy) as the binary label
- Each patient row generates two dataset entries (left eye + right eye)

Usage:
    from training.dataset_odir import ODIRDataset
    dataset = ODIRDataset(
        csv_path="data/odir5k/full_df.csv",
        images_dir="data/odir5k/preprocessed_images",
        transform=train_transform,
    )
"""

import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ODIRDataset(Dataset):
    """
    PyTorch Dataset for binary CVD risk classification from ODIR-5K.

    Each sample is one eye image. The label is 1 (hypertensive retinopathy
    present) or 0 (not present), drawn from the 'H' column in full_df.csv.

    Because each patient has two eyes, both are included as separate samples
    (with the same label per patient row).
    """

    def __init__(
        self,
        csv_path: str,
        images_dir: str,
        transform=None,
    ):
        """
        Args:
            csv_path:   Path to full_df.csv.
            images_dir: Path to the folder containing retinal images.
            transform:  Torchvision transform pipeline.
        """
        df = pd.read_csv(csv_path)

        # Build a flat list of (image_filename, binary_label) tuples
        records = []
        for _, row in df.iterrows():
            # ODIR 'H' column contains 1/0 or True/False for hypertension
            label = int(bool(row.get("H", 0)))

            # Left eye
            left_file = str(row.get("Left-Fundus", "")).strip()
            if left_file:
                records.append((left_file, label))

            # Right eye
            right_file = str(row.get("Right-Fundus", "")).strip()
            if right_file:
                records.append((right_file, label))

        # Filter out records whose image files don't exist on disk
        # (some Kaggle ODIR downloads have gaps in the image set)
        valid_records = [
            (fname, lbl)
            for fname, lbl in records
            if os.path.isfile(os.path.join(images_dir, fname))
        ]
        skipped = len(records) - len(valid_records)
        if skipped > 0:
            print(f"[ODIRDataset] Skipped {skipped} records with missing image files.")

        self.records = valid_records
        self.images_dir = images_dir
        self.transform = transform or transforms.Compose([transforms.ToTensor()])

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns:
            image (torch.Tensor): Transformed image tensor.
            label (float):        Binary CVD risk label (0.0 or 1.0) for BCEWithLogitsLoss.
        """
        filename, label = self.records[idx]
        img_path = os.path.join(self.images_dir, filename)
        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except (FileNotFoundError, OSError):
            # Fallback: return a black image tensor if file is unreadable
            import torch
            image = torch.zeros(3, 224, 224)

        # BCEWithLogitsLoss expects float labels
        return image, float(label)


def get_odir_transforms(train: bool = True):
    """
    Return the torchvision transform pipeline for ODIR-5K.

    Mirrors the APTOS 2019 transforms so both models train with the same
    preprocessing strategy.

    Args:
        train: If True, include augmentation transforms.

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
