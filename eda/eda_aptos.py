# EDA for APTOS 2019 Blindness Detection — generates 12 plots saved to eda/output/aptos_*.png
#
# Run from project root:
#   conda activate retinal_xai
#   python eda/eda_aptos.py


import os
import sys
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # Non-interactive backend — no display needed
import matplotlib.pyplot as plt
import cv2
from PIL import Image

# Allow imports from project root
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_PATH   = os.path.join(Config.APTOS_DATA_DIR, "train.csv")
IMAGES_DIR = os.path.join(Config.APTOS_DATA_DIR, "train_images")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

GRADE_LABELS_SHORT = {
    0: "No DR (0)",
    1: "Mild (1)",
    2: "Moderate (2)",
    3: "Severe (3)",
    4: "Proliferative (4)",
}

# Dissertation colour palette
GRADE_COLOURS = ["#2196F3", "#4CAF50", "#FF9800", "#F44336", "#9C27B0"]

plt.rcParams.update({
    "figure.dpi":        150,
    "savefig.dpi":       150,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    14,
    "axes.labelsize":    12,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "#f8f9fa",
    "grid.color":        "white",
    "grid.linewidth":    1.2,
})


def save(fig, name: str):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def load_image_rgb(img_path: str, size: int = 224):
    try:
        bgr = cv2.imread(img_path)
        if bgr is None:
            return None
        bgr = cv2.resize(bgr, (size, size))
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Load CSV
# ---------------------------------------------------------------------------
print("[eda_aptos] Loading CSV ...")
df = pd.read_csv(CSV_PATH)
df = df[df["diagnosis"].isin([0, 1, 2, 3, 4])].reset_index(drop=True)
counts = df["diagnosis"].value_counts().sort_index()
grades = counts.index.tolist()
n_total = len(df)

print(f"[eda_aptos] Total samples: {n_total}")
for g, c in counts.items():
    print(f"  Grade {g} ({GRADE_LABELS_SHORT[g]}): {c}  ({100*c/n_total:.1f}%)")


# ===========================================================================
# Plot 1 — Class Distribution Bar Chart
# ===========================================================================
print("\n[1/12] Class distribution bar chart ...")
fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(
    [GRADE_LABELS_SHORT[g] for g in grades],
    [counts[g] for g in grades],
    color=GRADE_COLOURS, edgecolor="white", linewidth=1.5, width=0.65
)
for bar, g in zip(bars, grades):
    v = counts[g]
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 25,
            f"{v}\n({100*v/n_total:.1f}%)", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_title("APTOS 2019 — Class Distribution (DR Grade)", fontweight="bold", pad=12)
ax.set_xlabel("Diabetic Retinopathy Grade")
ax.set_ylabel("Number of Images")
ax.set_ylim(0, counts.max() * 1.18)
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
fig.tight_layout()
save(fig, "aptos_01_class_distribution_bar.png")


# ===========================================================================
# Plot 2 — Class Distribution Pie Chart
# ===========================================================================
print("[2/12] Class distribution pie chart ...")
fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts, autotexts = ax.pie(
    [counts[g] for g in grades],
    labels=[GRADE_LABELS_SHORT[g] for g in grades],
    autopct="%1.1f%%",
    colors=GRADE_COLOURS,
    startangle=140,
    pctdistance=0.78,
    wedgeprops=dict(edgecolor="white", linewidth=2),
)
for at in autotexts:
    at.set_fontsize(10)
    at.set_fontweight("bold")
ax.set_title("APTOS 2019 — Class Proportions", fontweight="bold", pad=16)
save(fig, "aptos_02_class_distribution_pie.png")


# ===========================================================================
# Plot 3 — Class Imbalance Ratio vs. Majority Class
# ===========================================================================
print("[3/12] Imbalance ratio bar chart ...")
majority = counts.max()
imbalance = {g: majority / counts[g] for g in grades}

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(
    [GRADE_LABELS_SHORT[g] for g in grades],
    [imbalance[g] for g in grades],
    color=GRADE_COLOURS, edgecolor="white", linewidth=1.5, width=0.65
)
for bar, g in zip(bars, grades):
    v = imbalance[g]
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{v:.2f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.axhline(1.0, color="red", linestyle="--", linewidth=1.5, label="Majority class (1x)")
ax.set_title("APTOS 2019 — Class Imbalance Ratio (Majority / Grade)", fontweight="bold", pad=12)
ax.set_xlabel("DR Grade")
ax.set_ylabel("Imbalance Ratio (x)")
ax.legend()
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
save(fig, "aptos_03_class_imbalance_ratio.png")


# ===========================================================================
# Plot 4 — Cumulative Distribution
# ===========================================================================
print("[4/12] Cumulative distribution ...")
cumulative = [counts[g] for g in grades]
cum_pct = np.cumsum(cumulative) / n_total * 100

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar([GRADE_LABELS_SHORT[g] for g in grades], cum_pct,
       color=GRADE_COLOURS, edgecolor="white", alpha=0.85, width=0.65)
ax.plot([GRADE_LABELS_SHORT[g] for g in grades], cum_pct,
        "o-", color="#333", linewidth=2, markersize=8)
for g, cp in zip(grades, cum_pct):
    ax.text(g, cp + 1.5, f"{cp:.1f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_title("APTOS 2019 — Cumulative Class Distribution", fontweight="bold", pad=12)
ax.set_xlabel("DR Grade (ordered)")
ax.set_ylabel("Cumulative % of Dataset")
ax.set_ylim(0, 115)
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
save(fig, "aptos_04_cumulative_distribution.png")


# ===========================================================================
# Plot 5 — Sample Image Grid (5 samples per grade)
# ===========================================================================
print("[5/12] Sample image grid (5 per grade) ...")
N_SAMPLES = 5
fig, axes = plt.subplots(len(grades), N_SAMPLES, figsize=(N_SAMPLES * 2.5, len(grades) * 2.5))
fig.suptitle("APTOS 2019 — Sample Retinal Images per DR Grade", fontsize=14, fontweight="bold", y=1.01)

for row, g in enumerate(grades):
    ids = df[df["diagnosis"] == g]["id_code"].tolist()
    chosen = random.sample(ids, min(N_SAMPLES, len(ids)))
    for col in range(N_SAMPLES):
        ax = axes[row, col]
        ax.axis("off")
        if col < len(chosen):
            img_path = os.path.join(IMAGES_DIR, f"{chosen[col]}.png")
            if not os.path.exists(img_path):
                img_path = os.path.join(IMAGES_DIR, f"{chosen[col]}.jpg")
            rgb = load_image_rgb(img_path)
            if rgb is not None:
                ax.imshow(rgb)
        if col == 0:
            ax.set_ylabel(GRADE_LABELS_SHORT[g], fontsize=10, fontweight="bold", labelpad=6)

plt.subplots_adjust(wspace=0.04, hspace=0.08)
save(fig, "aptos_05_sample_image_grid.png")


# ===========================================================================
# Plot 6 — Mean Image per Grade
# ===========================================================================
print("[6/12] Mean image per grade (loads images — may take ~1 min) ...")
N_FOR_MEAN = 100

fig, axes = plt.subplots(1, len(grades), figsize=(len(grades) * 3, 3.5))
fig.suptitle("APTOS 2019 — Mean Retinal Image per DR Grade", fontsize=13, fontweight="bold")

for ax, g in zip(axes, grades):
    ids = df[df["diagnosis"] == g]["id_code"].tolist()
    chosen = random.sample(ids, min(N_FOR_MEAN, len(ids)))
    imgs = []
    for iid in chosen:
        p = os.path.join(IMAGES_DIR, f"{iid}.png")
        if not os.path.exists(p):
            p = os.path.join(IMAGES_DIR, f"{iid}.jpg")
        rgb = load_image_rgb(p)
        if rgb is not None:
            imgs.append(rgb.astype(np.float32))
    if imgs:
        mean_img = np.mean(imgs, axis=0).astype(np.uint8)
        ax.imshow(mean_img)
    ax.set_title(GRADE_LABELS_SHORT[g], fontsize=10, fontweight="bold")
    ax.axis("off")
save(fig, "aptos_06_mean_image_per_grade.png")


# ===========================================================================
# Plot 7 — RGB Channel Histograms (all grades overlaid)
# ===========================================================================
print("[7/12] RGB channel histograms ...")
N_HIST = 40
channels = {"Red": 0, "Green": 1, "Blue": 2}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("APTOS 2019 — RGB Channel Intensity Histograms per DR Grade", fontsize=13, fontweight="bold")

for ax, (ch_name, ch_idx) in zip(axes, channels.items()):
    for g in grades:
        ids = df[df["diagnosis"] == g]["id_code"].tolist()
        chosen = random.sample(ids, min(N_HIST, len(ids)))
        all_pixels = []
        for iid in chosen:
            p = os.path.join(IMAGES_DIR, f"{iid}.png")
            if not os.path.exists(p):
                p = os.path.join(IMAGES_DIR, f"{iid}.jpg")
            rgb = load_image_rgb(p)
            if rgb is not None:
                all_pixels.extend(rgb[:, :, ch_idx].flatten().tolist())
        if all_pixels:
            ax.hist(all_pixels, bins=64, range=(0, 255), alpha=0.55,
                    label=GRADE_LABELS_SHORT[g], color=GRADE_COLOURS[g], density=True)
    ax.set_title(f"{ch_name} Channel", fontweight="bold")
    ax.set_xlabel("Pixel Intensity (0-255)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, loc="upper left")
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
save(fig, "aptos_07_rgb_histograms.png")


# ===========================================================================
# Plot 8 — Mean Channel Intensity per Grade (grouped bar)
# ===========================================================================
print("[8/12] Mean channel intensity per grade ...")
N_SAMPLE = 60
mean_r, mean_g_ch, mean_b = [], [], []

for g in grades:
    ids = df[df["diagnosis"] == g]["id_code"].tolist()
    chosen = random.sample(ids, min(N_SAMPLE, len(ids)))
    rs, gs, bs = [], [], []
    for iid in chosen:
        p = os.path.join(IMAGES_DIR, f"{iid}.png")
        if not os.path.exists(p):
            p = os.path.join(IMAGES_DIR, f"{iid}.jpg")
        rgb = load_image_rgb(p)
        if rgb is not None:
            rs.append(rgb[:, :, 0].mean())
            gs.append(rgb[:, :, 1].mean())
            bs.append(rgb[:, :, 2].mean())
    mean_r.append(np.mean(rs) if rs else 0)
    mean_g_ch.append(np.mean(gs) if gs else 0)
    mean_b.append(np.mean(bs) if bs else 0)

x = np.arange(len(grades))
w = 0.25
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w, mean_r,    width=w, label="Red",   color="#e53935", edgecolor="white")
ax.bar(x,     mean_g_ch, width=w, label="Green", color="#43a047", edgecolor="white")
ax.bar(x + w, mean_b,    width=w, label="Blue",  color="#1e88e5", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([GRADE_LABELS_SHORT[g] for g in grades])
ax.set_title("APTOS 2019 — Mean RGB Channel Intensity per DR Grade", fontweight="bold", pad=12)
ax.set_xlabel("DR Grade")
ax.set_ylabel("Mean Pixel Intensity (0-255)")
ax.legend()
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
save(fig, "aptos_08_rgb_per_grade_histogram.png")


# ===========================================================================
# Plot 9 — Image Dimensions Scatter Plot
# ===========================================================================
print("[9/12] Image dimensions scatter plot ...")
N_DIM = 200
dim_data = {g: {"w": [], "h": []} for g in grades}

sample_df = df.sample(min(N_DIM * len(grades), len(df)), random_state=SEED)
for _, row in sample_df.iterrows():
    g = int(row["diagnosis"])
    p = os.path.join(IMAGES_DIR, f"{row['id_code']}.png")
    if not os.path.exists(p):
        p = os.path.join(IMAGES_DIR, f"{row['id_code']}.jpg")
    try:
        with Image.open(p) as img:
            w, h = img.size
            dim_data[g]["w"].append(w)
            dim_data[g]["h"].append(h)
    except Exception:
        pass

fig, ax = plt.subplots(figsize=(8, 7))
for g in grades:
    if dim_data[g]["w"]:
        ax.scatter(dim_data[g]["w"], dim_data[g]["h"],
                   label=GRADE_LABELS_SHORT[g], color=GRADE_COLOURS[g],
                   alpha=0.55, s=30, edgecolors="white", linewidths=0.5)
ax.set_title("APTOS 2019 — Image Dimensions (Width x Height)", fontweight="bold", pad=12)
ax.set_xlabel("Image Width (px)")
ax.set_ylabel("Image Height (px)")
ax.legend(title="DR Grade", fontsize=9)
ax.yaxis.grid(True, linestyle="--", alpha=0.6)
ax.xaxis.grid(True, linestyle="--", alpha=0.6)
ax.set_axisbelow(True)
save(fig, "aptos_09_image_dimensions.png")


# ===========================================================================
# Plot 10 — Train / Val / Test Split per Grade
# ===========================================================================
print("[10/12] Train/Val/Test split per grade ...")
from sklearn.model_selection import train_test_split

n = len(df)
n_train_approx = int(0.70 * n)
n_val_approx   = int(0.15 * n)
n_test_approx  = n - n_train_approx - n_val_approx

idx_all = df.index.tolist()
idx_train_val, idx_test = train_test_split(idx_all, test_size=n_test_approx/n, random_state=SEED)
idx_train, idx_val      = train_test_split(idx_train_val, test_size=n_val_approx/(n_train_approx+n_val_approx), random_state=SEED)

n_train_actual = len(idx_train)
n_val_actual   = len(idx_val)
n_test_actual  = len(idx_test)

split_counts = {}
for g in grades:
    split_counts[g] = {
        "Train": len([i for i in idx_train if df.loc[i, "diagnosis"] == g]),
        "Val":   len([i for i in idx_val   if df.loc[i, "diagnosis"] == g]),
        "Test":  len([i for i in idx_test  if df.loc[i, "diagnosis"] == g]),
    }

x = np.arange(len(grades))
w = 0.28
train_c = [split_counts[g]["Train"] for g in grades]
val_c   = [split_counts[g]["Val"]   for g in grades]
test_c  = [split_counts[g]["Test"]  for g in grades]

fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - w, train_c, width=w, label=f"Train ({n_train_actual})", color="#1565C0", edgecolor="white")
ax.bar(x,     val_c,   width=w, label=f"Val ({n_val_actual})",     color="#F57F17", edgecolor="white")
ax.bar(x + w, test_c,  width=w, label=f"Test ({n_test_actual})",   color="#2E7D32", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([GRADE_LABELS_SHORT[g] for g in grades])
ax.set_title("APTOS 2019 — Train / Val / Test Split per DR Grade (70/15/15)", fontweight="bold", pad=12)
ax.set_xlabel("DR Grade")
ax.set_ylabel("Number of Images")
ax.legend()
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
save(fig, "aptos_10_train_val_test_split.png")


# ===========================================================================
# Plot 11 — Class Weights Used During Training
# ===========================================================================
print("[11/12] Class weights visualisation ...")
class_weights = {0: 0.5, 1: 2.0, 2: 1.0, 3: 3.5, 4: 2.5}

fig, ax = plt.subplots(figsize=(9, 5))
bars = ax.bar(
    [GRADE_LABELS_SHORT[g] for g in grades],
    [class_weights[g] for g in grades],
    color=GRADE_COLOURS, edgecolor="white", linewidth=1.5, width=0.65
)
for bar, g in zip(bars, grades):
    v = class_weights[g]
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
            f"{v}x", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.axhline(1.0, color="#555", linestyle="--", linewidth=1.5, label="Neutral weight (1.0x)")
ax.set_title("APTOS 2019 — CrossEntropyLoss Class Weights (Inverse Frequency)", fontweight="bold", pad=12)
ax.set_xlabel("DR Grade")
ax.set_ylabel("Loss Weight")
ax.legend()
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
save(fig, "aptos_11_class_weights.png")


# ===========================================================================
# Plot 12 — Brightness Distribution per Grade (box plot)
# ===========================================================================
print("[12/12] Brightness distribution per grade ...")
N_BRIGHT = 80
brightness_per_grade = {g: [] for g in grades}

for g in grades:
    ids = df[df["diagnosis"] == g]["id_code"].tolist()
    chosen = random.sample(ids, min(N_BRIGHT, len(ids)))
    for iid in chosen:
        p = os.path.join(IMAGES_DIR, f"{iid}.png")
        if not os.path.exists(p):
            p = os.path.join(IMAGES_DIR, f"{iid}.jpg")
        rgb = load_image_rgb(p)
        if rgb is not None:
            brightness_per_grade[g].append(rgb.mean())

fig, ax = plt.subplots(figsize=(10, 5))
bp = ax.boxplot(
    [brightness_per_grade[g] for g in grades],
    labels=[GRADE_LABELS_SHORT[g] for g in grades],
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    flierprops=dict(marker="o", markersize=4, alpha=0.5),
)
for patch, colour in zip(bp["boxes"], GRADE_COLOURS):
    patch.set_facecolor(colour)
    patch.set_alpha(0.85)

ax.set_title("APTOS 2019 — Image Brightness Distribution per DR Grade", fontweight="bold", pad=12)
ax.set_xlabel("DR Grade")
ax.set_ylabel("Mean Pixel Brightness (0-255)")
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
save(fig, "aptos_12_brightness_per_grade.png")


print(f"\n[eda_aptos] All 12 plots saved to: {OUT_DIR}")
