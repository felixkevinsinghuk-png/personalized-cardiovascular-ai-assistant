"""
training/evaluate.py
Ablation study and evaluation script.

Evaluates three configurations on the held-out test sets:
    1. ResNet-50 only (DR grade → risk level via direct threshold)
    2. EfficientNet-B4 only (CVD score → risk level via direct threshold)
    3. Fused output (weighted combination of both models)

For each configuration, computes:
    - AUC-ROC
    - F1-Score (weighted average)
    - Sensitivity (Recall / True Positive Rate)
    - Specificity (True Negative Rate)

Results are printed as a formatted table for the dissertation.

Run from the project root after training both models:
    conda activate retinal_xai
    python training/evaluate.py
"""

import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config
from models.model_loader import load_models
from training.dataset_aptos import APTOSDataset, get_aptos_transforms
from training.dataset_odir import ODIRDataset, get_odir_transforms
from ml.fusion import fuse_scores


SEED = 42


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def compute_binary_metrics(y_true: list, y_prob: list, threshold: float = 0.5) -> dict:
    """
    Compute AUC-ROC, F1, Sensitivity, and Specificity for binary classification.

    Args:
        y_true:    Ground truth binary labels (0 or 1).
        y_prob:    Predicted probabilities for class 1.
        threshold: Decision threshold for binarising probabilities.

    Returns:
        Dictionary of metric names to float values.
    """
    y_pred = [1 if p >= threshold else 0 for p in y_prob]

    auc = roc_auc_score(y_true, y_prob)
    f1  = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    return {
        "AUC-ROC":     round(auc, 4),
        "F1-Score":    round(f1, 4),
        "Sensitivity": round(sensitivity, 4),
        "Specificity": round(specificity, 4),
    }


def evaluate_resnet_only(resnet, device) -> dict:
    """
    Evaluate ResNet-50 alone on the APTOS test split.
    Binary conversion: DR grade >= 2 → CVD risk positive (high-risk proxy).
    """
    csv_path   = os.path.join(Config.APTOS_DATA_DIR, "train.csv")
    images_dir = os.path.join(Config.APTOS_DATA_DIR, "train_images")

    full_dataset = APTOSDataset(csv_path, images_dir, transform=get_aptos_transforms(train=False))
    n = len(full_dataset)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val

    _, _, test_data = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )
    loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)

    resnet.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = resnet(images)
            probs = torch.softmax(logits.cpu(), dim=1).numpy()

            # Sum probabilities of grades 2–4 as "positive" risk probability
            positive_prob = probs[:, 2:].sum(axis=1)
            all_probs.extend(positive_prob)

            # Binary label: grade >= 2 → 1, else → 0
            binary_labels = (labels.numpy() >= 2).astype(int)
            all_labels.extend(binary_labels)

    return compute_binary_metrics(all_labels, all_probs)


def evaluate_efficientnet_only(efficientnet, device) -> dict:
    """Evaluate EfficientNet-B4 alone on the ODIR-5K test split."""
    csv_path   = os.path.join(Config.ODIR_DATA_DIR, "full_df.csv")
    images_dir = os.path.join(Config.ODIR_DATA_DIR, "ODIR-5K_Training_Images")

    full_dataset = ODIRDataset(csv_path, images_dir, transform=get_odir_transforms(train=False))
    n = len(full_dataset)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val

    _, _, test_data = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )
    loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)

    efficientnet.eval()
    all_labels, all_probs = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = efficientnet(images)
            probs = torch.sigmoid(logits.cpu()).squeeze(1).numpy()
            all_probs.extend(probs)
            all_labels.extend([int(l) for l in labels.numpy()])

    return compute_binary_metrics(all_labels, all_probs)


def evaluate_fused(resnet, efficientnet, device) -> dict:
    """
    Evaluate the fused model on a combined test evaluation.

    Uses ODIR-5K test images (which have ground truth CVD risk labels).
    The fused score is computed by running each image through both models
    and applying the weighted fusion formula.

    Note: This requires both models to process the same images.
    We approximate fusion evaluation using ODIR-5K test set only,
    as it provides reliable binary CVD risk ground truth.
    """
    csv_path   = os.path.join(Config.ODIR_DATA_DIR, "full_df.csv")
    images_dir = os.path.join(Config.ODIR_DATA_DIR, "ODIR-5K_Training_Images")

    full_dataset = ODIRDataset(csv_path, images_dir, transform=get_odir_transforms(train=False))
    n = len(full_dataset)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val

    _, _, test_data = random_split(
        full_dataset, [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(SEED),
    )
    loader = DataLoader(test_data, batch_size=32, shuffle=False, num_workers=2)

    resnet.eval()
    efficientnet.eval()
    all_labels, all_fused_scores = [], []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            # ResNet: get DR grade probabilities
            resnet_logits = resnet(images)
            resnet_probs = torch.softmax(resnet_logits.cpu(), dim=1).numpy()
            dr_grades = resnet_probs.argmax(axis=1)

            # EfficientNet: get CVD scores
            effnet_logits = efficientnet(images)
            cvd_scores = torch.sigmoid(effnet_logits.cpu()).squeeze(1).numpy()

            # Fuse scores
            for dr_grade, cvd_score in zip(dr_grades, cvd_scores):
                fused_score, _ = fuse_scores(int(dr_grade), float(cvd_score))
                all_fused_scores.append(fused_score)

            all_labels.extend([int(l) for l in labels.numpy()])

    return compute_binary_metrics(all_labels, all_fused_scores)


def print_results_table(results: dict):
    """Print ablation study results as a formatted table."""
    print("\n" + "=" * 65)
    print("ABLATION STUDY RESULTS")
    print("=" * 65)
    print(f"{'Configuration':<25} {'AUC-ROC':>10} {'F1-Score':>10} {'Sensitivity':>13} {'Specificity':>13}")
    print("-" * 65)
    for config_name, metrics in results.items():
        print(
            f"{config_name:<25} "
            f"{metrics['AUC-ROC']:>10.4f} "
            f"{metrics['F1-Score']:>10.4f} "
            f"{metrics['Sensitivity']:>13.4f} "
            f"{metrics['Specificity']:>13.4f}"
        )
    print("=" * 65)


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()
    print(f"[evaluate] Device: {device}")
    print("[evaluate] Loading models...")

    resnet, efficientnet, _, _, device = load_models()

    print("[evaluate] Running ablation study...")
    results = {}

    print("  → Evaluating ResNet-50 only...")
    results["ResNet-50 Only"] = evaluate_resnet_only(resnet, device)

    print("  → Evaluating EfficientNet-B4 only...")
    results["EfficientNet-B4 Only"] = evaluate_efficientnet_only(efficientnet, device)

    print("  → Evaluating fused output...")
    results["Fused (w1=0.4, w2=0.6)"] = evaluate_fused(resnet, efficientnet, device)

    print_results_table(results)


if __name__ == "__main__":
    main()
