# Confusion matrices and model evaluation plots — 10 figures saved to eda/output/cm_*.png
# Requires trained model weights in models/ directory.
#
# Run from project root:
#   conda activate retinal_xai
#   python eda/eda_confusion_matrix.py


import os
import sys
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, f1_score, recall_score,
    roc_curve, auc as sk_auc,
    ConfusionMatrixDisplay,
)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config
from models.model_loader import load_models
from training.dataset_aptos import APTOSDataset, get_aptos_transforms
from training.dataset_odir import ODIRDataset, get_odir_transforms
from ml.fusion import fuse_scores

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
OUT_DIR = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42

GRADE_LABELS_SHORT = ["No DR\n(0)", "Mild\n(1)", "Moderate\n(2)", "Severe\n(3)", "Prolif.\n(4)"]
GRADE_LABELS_FULL  = ["No DR (0)", "Mild DR (1)", "Moderate DR (2)", "Severe DR (3)", "Proliferative (4)"]

plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    13,
    "axes.labelsize":    11,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f8f9fa",
})


def save(fig, name: str):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Load test datasets
# ---------------------------------------------------------------------------
def get_aptos_test_loader():
    csv_path   = os.path.join(Config.APTOS_DATA_DIR, "train.csv")
    images_dir = os.path.join(Config.APTOS_DATA_DIR, "train_images")
    full = APTOSDataset(csv_path, images_dir, transform=get_aptos_transforms(train=False))
    n = len(full)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val
    _, _, test = random_split(full, [n_train, n_val, n_test],
                               generator=torch.Generator().manual_seed(SEED))
    return DataLoader(test, batch_size=32, shuffle=False, num_workers=0)


def get_odir_test_loader():
    csv_path   = os.path.join(Config.ODIR_DATA_DIR, "full_df.csv")
    images_dir = os.path.join(Config.ODIR_DATA_DIR, "preprocessed_images")
    full = ODIRDataset(csv_path, images_dir, transform=get_odir_transforms(train=False))
    n = len(full)
    n_train = int(0.70 * n)
    n_val   = int(0.15 * n)
    n_test  = n - n_train - n_val
    _, _, test = random_split(full, [n_train, n_val, n_test],
                               generator=torch.Generator().manual_seed(SEED))
    return DataLoader(test, batch_size=32, shuffle=False, num_workers=0)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def run_resnet_on_loader(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits.cpu(), dim=1).numpy()
            preds = probs.argmax(axis=1)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.numpy().astype(int))
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


def run_effnet_on_loader(model, loader, device):
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            logits = model(imgs)
            probs = torch.sigmoid(logits.cpu()).squeeze(1).numpy()
            preds = (probs >= 0.5).astype(int)
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend([int(l) for l in labels.numpy()])
    return np.array(all_labels), np.array(all_preds), np.array(all_probs)


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
print("[eda_cm] Loading models ...")
resnet, efficientnet, _, _, device = load_models()
print(f"[eda_cm] Device: {device}")

print("[eda_cm] Running ResNet-50 inference on APTOS test set ...")
aptos_loader = get_aptos_test_loader()
r_labels, r_preds, r_probs = run_resnet_on_loader(resnet, aptos_loader, device)

print("[eda_cm] Running EfficientNet-B4 inference on ODIR test set ...")
odir_loader = get_odir_test_loader()
e_labels, e_preds, e_probs = run_effnet_on_loader(efficientnet, odir_loader, device)


# ===========================================================================
# Plot 1 — ResNet-50 Normalised Confusion Matrix
# ===========================================================================
print("\n[1/10] ResNet-50 normalised confusion matrix ...")
cm_r = confusion_matrix(r_labels, r_preds, labels=[0, 1, 2, 3, 4])
cm_r_norm = cm_r.astype(float) / cm_r.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm_r_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(GRADE_LABELS_SHORT, fontsize=9)
ax.set_yticklabels(GRADE_LABELS_SHORT, fontsize=9)
for i in range(5):
    for j in range(5):
        val = cm_r_norm[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                color="white" if val > 0.5 else "black", fontsize=9, fontweight="bold")
ax.set_title("ResNet-50 — Normalised Confusion Matrix\n(APTOS 2019 Test Set)", fontweight="bold", pad=12)
ax.set_xlabel("Predicted DR Grade")
ax.set_ylabel("True DR Grade")
save(fig, "cm_01_resnet_confusion_matrix.png")


# ===========================================================================
# Plot 2 — ResNet-50 Raw Count Confusion Matrix
# ===========================================================================
print("[2/10] ResNet-50 raw count confusion matrix ...")
fig, ax = plt.subplots(figsize=(8, 7))
im = ax.imshow(cm_r, interpolation="nearest", cmap="YlOrRd")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks(range(5)); ax.set_yticks(range(5))
ax.set_xticklabels(GRADE_LABELS_SHORT, fontsize=9)
ax.set_yticklabels(GRADE_LABELS_SHORT, fontsize=9)
for i in range(5):
    for j in range(5):
        val = cm_r[i, j]
        ax.text(j, i, str(val), ha="center", va="center",
                color="white" if val > cm_r.max() * 0.6 else "black", fontsize=9, fontweight="bold")
ax.set_title("ResNet-50 — Raw Count Confusion Matrix\n(APTOS 2019 Test Set)", fontweight="bold", pad=12)
ax.set_xlabel("Predicted DR Grade")
ax.set_ylabel("True DR Grade")
save(fig, "cm_02_resnet_confusion_matrix_raw.png")


# ===========================================================================
# Plot 3 — EfficientNet-B4 Normalised Confusion Matrix
# ===========================================================================
print("[3/10] EfficientNet-B4 normalised confusion matrix ...")
cm_e = confusion_matrix(e_labels, e_preds, labels=[0, 1])
cm_e_norm = cm_e.astype(float) / cm_e.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm_e_norm, interpolation="nearest", cmap="Reds", vmin=0, vmax=1)
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted\nNegative", "Predicted\nPositive"])
ax.set_yticklabels(["Actual\nNegative", "Actual\nPositive"])
for i in range(2):
    for j in range(2):
        val = cm_e_norm[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                color="white" if val > 0.5 else "black", fontsize=12, fontweight="bold")
ax.set_title("EfficientNet-B4 — Normalised Confusion Matrix\n(ODIR-5K CVD Test Set)", fontweight="bold", pad=12)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
save(fig, "cm_03_efficientnet_confusion_matrix.png")


# ===========================================================================
# Plot 4 — EfficientNet-B4 Raw Count Confusion Matrix
# ===========================================================================
print("[4/10] EfficientNet-B4 raw count confusion matrix ...")
tn, fp, fn, tp = cm_e.ravel()
labels_2x2 = [["TN", "FP"], ["FN", "TP"]]

fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm_e, interpolation="nearest", cmap="Blues")
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted\nNegative", "Predicted\nPositive"])
ax.set_yticklabels(["Actual\nNegative", "Actual\nPositive"])
for i in range(2):
    for j in range(2):
        val = cm_e[i, j]
        ax.text(j, i, f"{labels_2x2[i][j]}\n{val}", ha="center", va="center",
                color="white" if val > cm_e.max() * 0.6 else "black",
                fontsize=12, fontweight="bold")
ax.set_title("EfficientNet-B4 — Raw Count Confusion Matrix\n(ODIR-5K CVD Test Set)", fontweight="bold", pad=12)
ax.set_xlabel("Predicted Label")
ax.set_ylabel("True Label")
save(fig, "cm_04_efficientnet_confusion_matrix_raw.png")


# ===========================================================================
# Plot 5 — ResNet-50 Per-Class Precision / Recall / F1
# ===========================================================================
print("[5/10] ResNet-50 per-class metrics ...")
report = classification_report(r_labels, r_preds, labels=[0,1,2,3,4],
                                target_names=GRADE_LABELS_FULL, output_dict=True)

precision = [report[g]["precision"] for g in GRADE_LABELS_FULL]
recall    = [report[g]["recall"]    for g in GRADE_LABELS_FULL]
f1        = [report[g]["f1-score"]  for g in GRADE_LABELS_FULL]

x = np.arange(5)
w = 0.25
GRADE_COLOURS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

fig, ax = plt.subplots(figsize=(11, 5))
ax.bar(x - w, precision, width=w, label="Precision", color="#1565C0", edgecolor="white")
ax.bar(x,     recall,    width=w, label="Recall",    color="#2E7D32", edgecolor="white")
ax.bar(x + w, f1,        width=w, label="F1-Score",  color="#E65100", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(GRADE_LABELS_FULL, fontsize=9)
ax.set_ylim(0, 1.15)
for metric_vals, offset in [(precision, -w), (recall, 0), (f1, w)]:
    for i, v in enumerate(metric_vals):
        ax.text(i + offset, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_title("ResNet-50 — Per-Class Precision / Recall / F1\n(APTOS 2019 Test Set)", fontweight="bold", pad=12)
ax.set_xlabel("DR Grade")
ax.set_ylabel("Score")
ax.legend()
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
save(fig, "cm_05_resnet_per_class_metrics.png")


# ===========================================================================
# Plot 6 — EfficientNet-B4 Summary Metrics Bar Chart
# ===========================================================================
print("[6/10] EfficientNet-B4 summary metrics ...")
sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
auc_score   = roc_auc_score(e_labels, e_probs)
f1_score_v  = f1_score(e_labels, e_preds, average="weighted", zero_division=0)

metrics = {
    "AUC-ROC":     auc_score,
    "F1-Score\n(Weighted)": f1_score_v,
    "Sensitivity\n(Recall)": sensitivity,
    "Specificity": specificity,
}
metric_colours = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A"]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(list(metrics.keys()), list(metrics.values()),
              color=metric_colours, edgecolor="white", linewidth=1.5, width=0.55)
for bar, v in zip(bars, metrics.values()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
            f"{v:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.axhline(0.5, color="red", linestyle="--", linewidth=1.2, alpha=0.7, label="Random baseline (0.5)")
ax.set_ylim(0, 1.15)
ax.set_title("EfficientNet-B4 — CVD Risk Classification Metrics\n(ODIR-5K Test Set)", fontweight="bold", pad=12)
ax.set_ylabel("Score")
ax.legend()
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
save(fig, "cm_06_efficientnet_metrics_bar.png")


# ===========================================================================
# Plot 7 — Ablation Study: ResNet-only vs EfficientNet-only vs Fused
# ===========================================================================
print("[7/10] Ablation study bar chart ...")

# ResNet-only: binary proxy (grades 2-4 as positive)
r_binary_prob  = r_probs[:, 2:].sum(axis=1)
r_binary_label = (r_labels >= 2).astype(int)
resnet_auc  = roc_auc_score(r_binary_label, r_binary_prob)
r_binary_pred = (r_binary_prob >= 0.5).astype(int)
tn_r, fp_r, fn_r, tp_r = confusion_matrix(r_binary_label, r_binary_pred, labels=[0,1]).ravel()
resnet_f1   = f1_score(r_binary_label, r_binary_pred, average="weighted", zero_division=0)
resnet_sens = tp_r / (tp_r + fn_r) if (tp_r + fn_r) > 0 else 0.0
resnet_spec = tn_r / (tn_r + fp_r) if (tn_r + fp_r) > 0 else 0.0

# EfficientNet-only (already computed)
effnet_auc  = auc_score
effnet_f1   = f1_score_v
effnet_sens = sensitivity
effnet_spec = specificity

# Fused: re-run both models on ODIR test set
print("   -> Computing fused scores on ODIR test set ...")
resnet.eval(); efficientnet.eval()
fused_scores_list, fused_labels_list = [], []
with torch.no_grad():
    for imgs, labels in odir_loader:
        imgs = imgs.to(device)
        r_logits = resnet(imgs)
        r_p      = torch.softmax(r_logits.cpu(), dim=1).numpy()
        dr_grades = r_p.argmax(axis=1)
        e_logits = efficientnet(imgs)
        cvd_s    = torch.sigmoid(e_logits.cpu()).squeeze(1).numpy()
        for dr_g, cvd_v in zip(dr_grades, cvd_s):
            fs, _ = fuse_scores(int(dr_g), float(cvd_v))
            fused_scores_list.append(fs)
        fused_labels_list.extend([int(l) for l in labels.numpy()])

fused_arr = np.array(fused_scores_list)
fused_lab = np.array(fused_labels_list)
fused_auc  = roc_auc_score(fused_lab, fused_arr)
fused_pred = (fused_arr >= 0.5).astype(int)
tn_f, fp_f, fn_f, tp_f = confusion_matrix(fused_lab, fused_pred, labels=[0,1]).ravel()
fused_f1   = f1_score(fused_lab, fused_pred, average="weighted", zero_division=0)
fused_sens = tp_f / (tp_f + fn_f) if (tp_f + fn_f) > 0 else 0.0
fused_spec = tn_f / (tn_f + fp_f) if (tn_f + fp_f) > 0 else 0.0

ablation = {
    "ResNet-50\nOnly":           [resnet_auc, resnet_f1, resnet_sens, resnet_spec],
    "EfficientNet-B4\nOnly":     [effnet_auc,  effnet_f1,  effnet_sens,  effnet_spec],
    "Fused\n(w1=0.4, w2=0.6)":  [fused_auc,   fused_f1,   fused_sens,   fused_spec],
}

metric_names = ["AUC-ROC", "F1-Score", "Sensitivity", "Specificity"]
cfg_colours  = ["#1565C0", "#E65100", "#2E7D32"]
x = np.arange(len(metric_names))
w = 0.25

fig, ax = plt.subplots(figsize=(11, 6))
for i, (cfg, vals) in enumerate(ablation.items()):
    offset = (i - 1) * w
    bars = ax.bar(x + offset, vals, width=w, label=cfg, color=cfg_colours[i], edgecolor="white")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(metric_names)
ax.set_ylim(0, 1.18)
ax.set_title("Ablation Study — ResNet-50 vs EfficientNet-B4 vs Fused Model", fontweight="bold", pad=12)
ax.set_ylabel("Score")
ax.legend(title="Configuration", fontsize=9)
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
save(fig, "cm_07_ablation_study.png")


# ===========================================================================
# Plot 8 — ResNet-50 Multi-Class ROC Curves (OvR)
# ===========================================================================
print("[8/10] ResNet-50 multi-class ROC curves ...")
GRADE_COLOURS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]
fig, ax = plt.subplots(figsize=(8, 7))

from sklearn.preprocessing import label_binarize
r_labels_bin = label_binarize(r_labels, classes=[0, 1, 2, 3, 4])

for grade in range(5):
    fpr, tpr, _ = roc_curve(r_labels_bin[:, grade], r_probs[:, grade])
    roc_auc = sk_auc(fpr, tpr)
    ax.plot(fpr, tpr, lw=2, color=GRADE_COLOURS[grade],
            label=f"{GRADE_LABELS_FULL[grade]} (AUC={roc_auc:.3f})")

ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random (AUC=0.5)")
ax.set_title("ResNet-50 — Multi-Class ROC Curves (One-vs-Rest)\n(APTOS 2019 Test Set)", fontweight="bold", pad=12)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(fontsize=9, loc="lower right")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.xaxis.grid(True, linestyle="--", alpha=0.5)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
save(fig, "cm_08_resnet_roc_curve.png")


# ===========================================================================
# Plot 9 — EfficientNet-B4 Binary ROC Curve
# ===========================================================================
print("[9/10] EfficientNet-B4 ROC curve ...")
fpr_e, tpr_e, _ = roc_curve(e_labels, e_probs)
roc_e = sk_auc(fpr_e, tpr_e)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_e, tpr_e, lw=2.5, color="#E65100", label=f"EfficientNet-B4 (AUC={roc_e:.4f})")
ax.fill_between(fpr_e, tpr_e, alpha=0.12, color="#E65100")
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random (AUC=0.5)")
# Mark operating point (threshold=0.5)
thresh_idx = np.argmin(np.abs(np.linspace(0, 1, len(fpr_e)) - 0.5))
ax.scatter([fpr_e[thresh_idx]], [tpr_e[thresh_idx]], marker="o", s=80,
           color="#E65100", zorder=5, label="Operating point (t=0.5)")
ax.set_title("EfficientNet-B4 — Binary ROC Curve (CVD Risk)\n(ODIR-5K Test Set)", fontweight="bold", pad=12)
ax.set_xlabel("False Positive Rate (1 - Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.legend(fontsize=10, loc="lower right")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.xaxis.grid(True, linestyle="--", alpha=0.5)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
save(fig, "cm_09_efficientnet_roc_curve.png")


# ===========================================================================
# Plot 10 — Fused Model ROC Curve
# ===========================================================================
print("[10/10] Fused model ROC curve ...")
fpr_f, tpr_f, _ = roc_curve(fused_lab, fused_arr)
roc_f = sk_auc(fpr_f, tpr_f)

fig, ax = plt.subplots(figsize=(7, 6))
ax.plot(fpr_e, tpr_e, lw=2, color="#E65100", linestyle="--", alpha=0.7, label=f"EfficientNet-only (AUC={roc_e:.4f})")
ax.plot(fpr_f, tpr_f, lw=2.5, color="#1565C0", label=f"Fused Model (AUC={roc_f:.4f})")
ax.fill_between(fpr_f, tpr_f, alpha=0.1, color="#1565C0")
ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random (AUC=0.5)")
ax.set_title("Fused Model — ROC Curve (CVD Risk)\n(ODIR-5K Test Set)", fontweight="bold", pad=12)
ax.set_xlabel("False Positive Rate (1 - Specificity)")
ax.set_ylabel("True Positive Rate (Sensitivity)")
ax.legend(fontsize=10, loc="lower right")
ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
ax.xaxis.grid(True, linestyle="--", alpha=0.5)
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.set_axisbelow(True)
save(fig, "cm_10_fused_roc_curve.png")


print(f"\n[eda_cm] All 10 plots saved to: {OUT_DIR}")
print(f"\nFinal metric summary:")
print(f"  ResNet-50:      AUC={resnet_auc:.4f}, F1={resnet_f1:.4f}, Sens={resnet_sens:.4f}, Spec={resnet_spec:.4f}")
print(f"  EfficientNet:   AUC={effnet_auc:.4f},  F1={effnet_f1:.4f},  Sens={effnet_sens:.4f},  Spec={effnet_spec:.4f}")
print(f"  Fused:          AUC={fused_auc:.4f},  F1={fused_f1:.4f},  Sens={fused_sens:.4f},  Spec={fused_spec:.4f}")
