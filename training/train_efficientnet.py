"""
training/train_efficientnet.py
Training script for EfficientNet-B4 on the ODIR-5K CVD risk dataset.

Architecture:
    - EfficientNet-B4 with ImageNet pre-trained weights
    - Early MBConv blocks frozen — only final blocks and classifier are trained
    - Classifier replaced with Dropout(0.5) → Linear(1792, 1)

Training config:
    - Loss: BCEWithLogitsLoss (binary cross-entropy with built-in sigmoid)
    - Optimiser: AdamW (lr=1e-4, weight_decay=1e-4)
    - Scheduler: ReduceLROnPlateau (monitors validation AUC-ROC)
    - Device: MPS (Apple M3) with CPU fallback
    - Batch size: 32
    - Early stopping: patience=5 epochs on validation AUC-ROC

Outputs:
    - models/efficientnet.pth: Best model weights
    - Console: Per-epoch loss and AUC-ROC for train and validation sets

Run from the project root:
    conda activate retinal_xai
    python training/train_efficientnet.py
"""

import os
import sys
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config
from training.dataset_odir import ODIRDataset, get_odir_transforms


# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
NUM_EPOCHS = 50
BATCH_SIZE = Config.BATCH_SIZE
LR = Config.LEARNING_RATE
PATIENCE = Config.EARLY_STOPPING_PATIENCE
SEED = 42


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model() -> nn.Module:
    """Load pre-trained EfficientNet-B4 and replace the classifier."""
    model = tv_models.efficientnet_b4(weights=tv_models.EfficientNet_B4_Weights.IMAGENET1K_V1)

    # Freeze all features blocks except the last two
    # EfficientNet-B4 has 9 feature blocks (features[0] to features[8])
    for i, block in enumerate(model.features):
        if i < 6:  # Freeze blocks 0–5, train blocks 6–8
            for param in block.parameters():
                param.requires_grad = False

    # Replace classifier: Dropout + Linear(1792, 1)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.5, inplace=True),
        nn.Linear(in_features, 1),
    )
    return model


def train_one_epoch(model, loader, criterion, optimiser, device):
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.float().to(device).unsqueeze(1)  # float32 required for MPS

        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs.detach().cpu()).squeeze(1).numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().squeeze(1).numpy())

    avg_loss = total_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    return avg_loss, auc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.float().to(device).unsqueeze(1)  # float32 required for MPS
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs.cpu()).squeeze(1).numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().squeeze(1).numpy())

    avg_loss = total_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.0

    return avg_loss, auc


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()
    print(f"[train_efficientnet] Device: {device}")

    # --- Datasets ---
    csv_path   = os.path.join(Config.ODIR_DATA_DIR, "full_df.csv")
    images_dir = os.path.join(Config.ODIR_DATA_DIR, "preprocessed_images")

    full_dataset = ODIRDataset(csv_path, images_dir, transform=None)
    n = len(full_dataset)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val

    train_data, val_data, test_data = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )

    train_data.dataset.transform = get_odir_transforms(train=True)
    val_data.dataset.transform   = get_odir_transforms(train=False)
    test_data.dataset.transform  = get_odir_transforms(train=False)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=False)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)

    print(f"[train_efficientnet] Train: {n_train} | Val: {n_val} | Test: {n_test}")

    # --- Model, Loss, Optimiser ---
    model = build_model().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR, weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="max", factor=0.5, patience=3
    )

    # --- Training loop with early stopping ---
    best_val_auc = 0.0
    patience_counter = 0
    os.makedirs("models", exist_ok=True)

    for epoch in range(1, NUM_EPOCHS + 1):
        train_loss, train_auc = train_one_epoch(model, train_loader, criterion, optimiser, device)
        val_loss,   val_auc   = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_auc)

        print(
            f"Epoch {epoch:02d}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f}, AUC: {train_auc:.4f} | "
            f"Val Loss: {val_loss:.4f}, AUC: {val_auc:.4f}"
        )

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            patience_counter = 0
            torch.save(model.state_dict(), Config.EFFICIENTNET_WEIGHTS)
            print(f"  ✓ Saved best model (Val AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping triggered after {epoch} epochs.")
                break

    print(f"\n[train_efficientnet] Training complete. Best Val AUC-ROC: {best_val_auc:.4f}")
    print(f"[train_efficientnet] Weights saved to: {Config.EFFICIENTNET_WEIGHTS}")


if __name__ == "__main__":
    main()
