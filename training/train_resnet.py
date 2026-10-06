# Training script for ResNet-50 on APTOS 2019 (5-class DR grading).
#
# Architecture: ResNet-50 with layers 1-2 frozen, final FC replaced by Dropout(0.5) → Linear(2048, 5)
# Loss: CrossEntropyLoss with inverse-frequency class weights (grades 3 and 4 are rare)
# Optimiser: AdamW lr=1e-4, ReduceLROnPlateau scheduler, early stopping patience=5
#
# Run from project root:
#   conda activate retinal_xai
#   python training/train_resnet.py

import os
import sys
import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import roc_auc_score
import numpy as np

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config
from training.dataset_aptos import APTOSDataset, get_aptos_transforms


NUM_EPOCHS = 50
BATCH_SIZE = Config.BATCH_SIZE
LR         = Config.LEARNING_RATE
PATIENCE   = Config.EARLY_STOPPING_PATIENCE
NUM_CLASSES = Config.NUM_CLASSES_RESNET
SEED       = 42


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_model(num_classes: int) -> nn.Module:
    """Load pre-trained ResNet-50 and replace the final FC layer."""
    model = tv_models.resnet50(weights=tv_models.ResNet50_Weights.IMAGENET1K_V1)

    # Freeze layers 1 and 2
    for name, param in model.named_parameters():
        if name.startswith("layer1") or name.startswith("layer2"):
            param.requires_grad = False

    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),
        nn.Linear(in_features, num_classes),
    )
    return model


def train_one_epoch(model, loader, criterion, optimiser, device):
    model.train()
    total_loss = 0.0
    all_labels, all_probs = [], []

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimiser.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimiser.step()

        total_loss += loss.item() * images.size(0)
        probs = torch.softmax(outputs.detach().cpu(), dim=1).numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    # AUC-ROC: one-vs-rest multi-class
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0

    return avg_loss, auc


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs.cpu(), dim=1).numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(loader.dataset)
    try:
        auc = roc_auc_score(all_labels, all_probs, multi_class="ovr", average="macro")
    except ValueError:
        auc = 0.0

    return avg_loss, auc


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()
    print(f"[train_resnet] Device: {device}")

    # --- Datasets ---
    csv_path = os.path.join(Config.APTOS_DATA_DIR, "train.csv")
    images_dir = os.path.join(Config.APTOS_DATA_DIR, "train_images")

    full_dataset = APTOSDataset(csv_path, images_dir, transform=None)
    n = len(full_dataset)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val

    train_data, val_data, test_data = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )

    # Apply transforms separately (Dataset wrapping trick)
    train_data.dataset.transform = get_aptos_transforms(train=True)
    val_data.dataset.transform   = get_aptos_transforms(train=False)
    test_data.dataset.transform  = get_aptos_transforms(train=False)

    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2, pin_memory=False)
    val_loader   = DataLoader(val_data,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=False)

    print(f"[train_resnet] Train: {n_train} | Val: {n_val} | Test: {n_test}")

    # --- Model, Loss, Optimiser ---
    model = build_model(NUM_CLASSES).to(device)

    # Weighted CrossEntropyLoss — fixes class imbalance in APTOS dataset.
    # Weights are inversely proportional to class frequency:
    #   Grade 0 (No DR)          ~1800 images → weight 0.5  (very common, down-weighted)
    #   Grade 1 (Mild)            ~370 images → weight 2.0
    #   Grade 2 (Moderate)        ~999 images → weight 1.0
    #   Grade 3 (Severe)          ~193 images → weight 3.5  (rare, up-weighted)
    #   Grade 4 (Proliferative)   ~295 images → weight 2.5
    class_weights = torch.tensor([0.5, 2.0, 1.0, 3.5, 2.5]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
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
            torch.save(model.state_dict(), Config.RESNET_WEIGHTS)
            print(f"  * Saved best model (Val AUC: {best_val_auc:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"  Early stopping triggered after {epoch} epochs.")
                break

    print(f"\n[train_resnet] Training complete. Best Val AUC-ROC: {best_val_auc:.4f}")
    print(f"[train_resnet] Weights saved to: {Config.RESNET_WEIGHTS}")


if __name__ == "__main__":
    main()
