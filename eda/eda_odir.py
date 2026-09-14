# EDA for ODIR-5K Ocular Disease Recognition — generates 12 plots saved to eda/output/odir_*.png
# Label used: 'H' column (Hypertensive Retinopathy) — binary 0/1
#
# Run from project root:
#   conda activate retinal_xai
#   python eda/eda_odir.py

import os
import sys
import random
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import Config

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
CSV_PATH   = os.path.join(Config.ODIR_DATA_DIR, "full_df.csv")
IMAGES_DIR = os.path.join(Config.ODIR_DATA_DIR, "preprocessed_images")
OUT_DIR    = os.path.join(os.path.dirname(__file__), "output")
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Disease column mapping
DISEASE_COLS = {
    "N": "Normal",
    "D": "Diabetic Retinopathy",
    "G": "Glaucoma",
    "C": "Cataract",
    "A": "AMD",
    "H": "Hypertension (CVD target)",
    "M": "Myopia",
    "O": "Other",
}

# Colours
LABEL_COLOURS  = ["#2196F3", "#F44336"]   # 0=blue, 1=red
DISEASE_COLOURS = [
    "#4CAF50", "#F44336", "#9C27B0",
    "#FF9800", "#2196F3", "#E91E63",
    "#009688", "#795548"
]

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
print("[eda_odir] Loading CSV ...")
df = pd.read_csv(CSV_PATH)
df["H"] = df["H"].astype(int)
n_total = len(df)

print(f"[eda_odir] Total patient rows: {n_total}")
print(f"  CVD Positive (H=1): {df['H'].sum()}")
print(f"  CVD Negative (H=0): {(df['H'] == 0).sum()}")

# Build flat image records matching ODIRDataset logic
records = []
for _, row in df.iterrows():
    label = int(bool(row.get("H", 0)))
    for col in ["Left-Fundus", "Right-Fundus"]:
        fname = str(row.get(col, "")).strip()
        if fname:
            fpath = os.path.join(IMAGES_DIR, fname)
            if os.path.isfile(fpath):
                records.append({"filename": fname, "path": fpath, "label": label,
                                 "age": row.get("Patient Age", np.nan),
                                 "sex": row.get("Patient Sex", "Unknown")})

records_df = pd.DataFrame(records)
print(f"[eda_odir] Valid image records: {len(records_df)}")


# ===========================================================================
# Plot 1 — CVD Label Distribution (bar)
# ===========================================================================
print("\n[1/12] CVD label distribution bar ...")
label_counts = records_df["label"].value_counts().sort_index()
labels_str   = ["CVD Negative (H=0)", "CVD Positive (H=1)"]

fig, ax = plt.subplots(figsize=(7, 5))
bars = ax.bar(labels_str, [label_counts.get(0, 0), label_counts.get(1, 0)],
              color=LABEL_COLOURS, edgecolor="white", linewidth=1.5, width=0.55)
total_rec = len(records_df)
for bar, lbl in zip(bars, [0, 1]):
    v = label_counts.get(lbl, 0)
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
            f"{v}\n({100*v/total_rec:.1f}%)", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("ODIR-5K — CVD (Hypertension) Label Distribution", fontweight="bold", pad=12)
ax.set_ylabel("Number of Eye Images")
ax.set_ylim(0, label_counts.max() * 1.18)
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
save(fig, "odir_01_cvd_label_distribution_bar.png")


# ===========================================================================
# Plot 2 — CVD Label Distribution (pie)
# ===========================================================================
print("[2/12] CVD label distribution pie ...")
fig, ax = plt.subplots(figsize=(7, 7))
ax.pie([label_counts.get(0, 0), label_counts.get(1, 0)],
       labels=labels_str,
       autopct="%1.1f%%",
       colors=LABEL_COLOURS,
       startangle=90,
       pctdistance=0.78,
       wedgeprops=dict(edgecolor="white", linewidth=2))
ax.set_title("ODIR-5K — CVD Label Proportions", fontweight="bold", pad=16)
save(fig, "odir_02_cvd_label_distribution_pie.png")


# ===========================================================================
# Plot 3 — All 8 Disease Class Counts
# ===========================================================================
print("[3/12] All disease class counts ...")
disease_sums = {col: int(df[col].sum()) for col in DISEASE_COLS}
fig, ax = plt.subplots(figsize=(11, 5))
bars = ax.bar(
    list(DISEASE_COLS.values()),
    list(disease_sums.values()),
    color=DISEASE_COLOURS, edgecolor="white", linewidth=1.5, width=0.7
)
for bar, (col, v) in zip(bars, disease_sums.items()):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 15,
            f"{v}", ha="center", va="bottom", fontsize=9, fontweight="bold")
# Highlight the CVD (H) bar
for bar, col in zip(bars, DISEASE_COLS):
    if col == "H":
        bar.set_edgecolor("black")
        bar.set_linewidth(2.5)
ax.set_title("ODIR-5K — All 8 Disease Class Counts (Patient-Level)", fontweight="bold", pad=12)
ax.set_xlabel("Disease Category")
ax.set_ylabel("Number of Patients")
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
plt.xticks(rotation=20, ha="right")
save(fig, "odir_03_all_disease_counts.png")


# ===========================================================================
# Plot 4 — Patient Age Distribution
# ===========================================================================
print("[4/12] Patient age distribution ...")
ages = df["Patient Age"].dropna()
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(ages, bins=30, color="#5C6BC0", edgecolor="white", linewidth=1.2)
ax.axvline(ages.mean(), color="#F44336", linestyle="--", linewidth=2,
           label=f"Mean: {ages.mean():.1f} yrs")
ax.axvline(ages.median(), color="#FF9800", linestyle="--", linewidth=2,
           label=f"Median: {ages.median():.1f} yrs")
ax.set_title("ODIR-5K — Patient Age Distribution", fontweight="bold", pad=12)
ax.set_xlabel("Age (years)")
ax.set_ylabel("Number of Patients")
ax.legend()
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
save(fig, "odir_04_age_distribution.png")


# ===========================================================================
# Plot 5 — Age by CVD Label (box plot)
# ===========================================================================
print("[5/12] Age by CVD label box plot ...")
age_neg = df[df["H"] == 0]["Patient Age"].dropna()
age_pos = df[df["H"] == 1]["Patient Age"].dropna()

fig, ax = plt.subplots(figsize=(7, 5))
bp = ax.boxplot([age_neg, age_pos],
                labels=["CVD Negative (H=0)", "CVD Positive (H=1)"],
                patch_artist=True,
                medianprops=dict(color="white", linewidth=2.5),
                whiskerprops=dict(linewidth=1.5),
                capprops=dict(linewidth=1.5),
                flierprops=dict(marker="o", markersize=4, alpha=0.4))
for patch, colour in zip(bp["boxes"], LABEL_COLOURS):
    patch.set_facecolor(colour)
    patch.set_alpha(0.85)
ax.set_title("ODIR-5K — Patient Age by CVD Label", fontweight="bold", pad=12)
ax.set_ylabel("Age (years)")
# Annotate medians
for i, grp in enumerate([age_neg, age_pos], 1):
    ax.text(i, grp.median() + 0.8, f"Med={grp.median():.0f}", ha="center", fontsize=9, fontweight="bold")
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
save(fig, "odir_05_age_by_cvd_label.png")


# ===========================================================================
# Plot 6 — Patient Sex Distribution
# ===========================================================================
print("[6/12] Sex distribution ...")
sex_counts = df["Patient Sex"].value_counts()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("ODIR-5K — Patient Sex Distribution", fontsize=13, fontweight="bold")

sex_colours = ["#5C6BC0", "#E91E63"]
ax1.bar(sex_counts.index, sex_counts.values, color=sex_colours, edgecolor="white", width=0.55)
for i, (sex, v) in enumerate(sex_counts.items()):
    ax1.text(i, v + 20, f"{v}\n({100*v/n_total:.1f}%)",
             ha="center", va="bottom", fontsize=11, fontweight="bold")
ax1.set_ylabel("Number of Patients")
ax1.yaxis.grid(True, linestyle="--", alpha=0.7)
ax1.set_axisbelow(True)

ax2.pie(sex_counts.values, labels=sex_counts.index, autopct="%1.1f%%",
        colors=sex_colours, startangle=90, wedgeprops=dict(edgecolor="white", linewidth=2))
save(fig, "odir_06_sex_distribution.png")


# ===========================================================================
# Plot 7 — Sex by CVD Label (stacked bar)
# ===========================================================================
print("[7/12] Sex by CVD label stacked bar ...")
crosstab = pd.crosstab(df["Patient Sex"], df["H"])
crosstab.columns = ["CVD Negative", "CVD Positive"]

fig, ax = plt.subplots(figsize=(7, 5))
crosstab.plot(kind="bar", ax=ax, color=[LABEL_COLOURS[0], LABEL_COLOURS[1]],
              edgecolor="white", width=0.6)
ax.set_title("ODIR-5K — Sex vs. CVD Label", fontweight="bold", pad=12)
ax.set_xlabel("Patient Sex")
ax.set_ylabel("Number of Patients")
ax.legend(title="CVD Label")
plt.xticks(rotation=0)
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
save(fig, "odir_07_sex_by_cvd_label.png")


# ===========================================================================
# Plot 8 — Sample Image Grid (5 positive, 5 negative)
# ===========================================================================
print("[8/12] Sample image grid ...")
N_SAMPLES = 5
pos_paths = records_df[records_df["label"] == 1]["path"].tolist()
neg_paths = records_df[records_df["label"] == 0]["path"].tolist()
chosen_pos = random.sample(pos_paths, min(N_SAMPLES, len(pos_paths)))
chosen_neg = random.sample(neg_paths, min(N_SAMPLES, len(neg_paths)))

fig, axes = plt.subplots(2, N_SAMPLES, figsize=(N_SAMPLES * 2.5, 5.5))
fig.suptitle("ODIR-5K — Sample Retinal Images by CVD Label", fontsize=14, fontweight="bold")

row_labels = ["CVD Positive (H=1)", "CVD Negative (H=0)"]
for row, (paths, row_lbl) in enumerate([(chosen_pos, row_labels[0]), (chosen_neg, row_labels[1])]):
    for col in range(N_SAMPLES):
        ax = axes[row, col]
        ax.axis("off")
        if col < len(paths):
            rgb = load_image_rgb(paths[col])
            if rgb is not None:
                ax.imshow(rgb)
        if col == 0:
            ax.set_ylabel(row_lbl, fontsize=10, fontweight="bold", labelpad=6)

plt.subplots_adjust(wspace=0.04, hspace=0.1)
save(fig, "odir_08_sample_image_grid.png")


# ===========================================================================
# Plot 9 — RGB Channel Histograms by CVD Label
# ===========================================================================
print("[9/12] RGB histograms by CVD label ...")
N_HIST = 80
channels = {"Red": 0, "Green": 1, "Blue": 2}

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
fig.suptitle("ODIR-5K — RGB Channel Histograms by CVD Label", fontsize=13, fontweight="bold")

for ax, (ch_name, ch_idx) in zip(axes, channels.items()):
    for lbl, colour, lbl_str in [(0, LABEL_COLOURS[0], "CVD Negative"), (1, LABEL_COLOURS[1], "CVD Positive")]:
        paths = records_df[records_df["label"] == lbl]["path"].tolist()
        chosen = random.sample(paths, min(N_HIST, len(paths)))
        all_pixels = []
        for p in chosen:
            rgb = load_image_rgb(p)
            if rgb is not None:
                all_pixels.extend(rgb[:, :, ch_idx].flatten().tolist())
        if all_pixels:
            ax.hist(all_pixels, bins=64, range=(0, 255), alpha=0.6,
                    label=lbl_str, color=colour, density=True)
    ax.set_title(f"{ch_name} Channel", fontweight="bold")
    ax.set_xlabel("Pixel Intensity (0-255)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle="--", alpha=0.5)
    ax.set_axisbelow(True)
save(fig, "odir_09_rgb_histograms.png")


# ===========================================================================
# Plot 10 — Mean Channel Intensity by CVD Label
# ===========================================================================
print("[10/12] Mean channel intensity by label ...")
N_SAMPLE = 100
data = {0: {"R": [], "G": [], "B": []}, 1: {"R": [], "G": [], "B": []}}

for lbl in [0, 1]:
    paths = records_df[records_df["label"] == lbl]["path"].tolist()
    chosen = random.sample(paths, min(N_SAMPLE, len(paths)))
    for p in chosen:
        rgb = load_image_rgb(p)
        if rgb is not None:
            data[lbl]["R"].append(rgb[:, :, 0].mean())
            data[lbl]["G"].append(rgb[:, :, 1].mean())
            data[lbl]["B"].append(rgb[:, :, 2].mean())

x = np.array([0, 1])
w = 0.2
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(x - w, [np.mean(data[0]["R"]), np.mean(data[1]["R"])], width=w, label="Red",   color="#e53935", edgecolor="white")
ax.bar(x,     [np.mean(data[0]["G"]), np.mean(data[1]["G"])], width=w, label="Green", color="#43a047", edgecolor="white")
ax.bar(x + w, [np.mean(data[0]["B"]), np.mean(data[1]["B"])], width=w, label="Blue",  color="#1e88e5", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(["CVD Negative (H=0)", "CVD Positive (H=1)"])
ax.set_title("ODIR-5K — Mean RGB Channel Intensity by CVD Label", fontweight="bold", pad=12)
ax.set_ylabel("Mean Pixel Intensity (0-255)")
ax.legend()
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
save(fig, "odir_10_mean_channel_by_label.png")


# ===========================================================================
# Plot 11 — Brightness Box Plot by CVD Label
# ===========================================================================
print("[11/12] Brightness box plot by CVD label ...")
brightness = {0: [], 1: []}
N_BRIGHT = 150

for lbl in [0, 1]:
    paths = records_df[records_df["label"] == lbl]["path"].tolist()
    chosen = random.sample(paths, min(N_BRIGHT, len(paths)))
    for p in chosen:
        rgb = load_image_rgb(p)
        if rgb is not None:
            brightness[lbl].append(rgb.mean())

fig, ax = plt.subplots(figsize=(7, 5))
bp = ax.boxplot(
    [brightness[0], brightness[1]],
    labels=["CVD Negative (H=0)", "CVD Positive (H=1)"],
    patch_artist=True,
    medianprops=dict(color="white", linewidth=2.5),
    whiskerprops=dict(linewidth=1.5),
    capprops=dict(linewidth=1.5),
    flierprops=dict(marker="o", markersize=4, alpha=0.4),
)
for patch, colour in zip(bp["boxes"], LABEL_COLOURS):
    patch.set_facecolor(colour)
    patch.set_alpha(0.85)
ax.set_title("ODIR-5K — Image Brightness Distribution by CVD Label", fontweight="bold", pad=12)
ax.set_ylabel("Mean Pixel Brightness (0-255)")
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
save(fig, "odir_11_brightness_by_label.png")


# ===========================================================================
# Plot 12 — Train / Val / Test Split by CVD Label
# ===========================================================================
print("[12/12] Train/Val/Test split by CVD label ...")
from sklearn.model_selection import train_test_split

n_rec = len(records_df)
n_train_approx = int(0.70 * n_rec)
n_val_approx   = int(0.15 * n_rec)
n_test_approx  = n_rec - n_train_approx - n_val_approx

idx_all = list(range(n_rec))
idx_tv, idx_test = train_test_split(idx_all, test_size=n_test_approx/n_rec, random_state=SEED)
idx_train, idx_val = train_test_split(idx_tv, test_size=n_val_approx/(n_train_approx+n_val_approx), random_state=SEED)

split_results = {}
for split_name, idxs in [("Train", idx_train), ("Val", idx_val), ("Test", idx_test)]:
    sub = records_df.iloc[idxs]
    split_results[split_name] = {
        "CVD Negative": int((sub["label"] == 0).sum()),
        "CVD Positive": int((sub["label"] == 1).sum()),
        "total": len(sub),
    }

x = np.arange(3)
w = 0.35
splits = ["Train", "Val", "Test"]
neg_counts = [split_results[s]["CVD Negative"] for s in splits]
pos_counts = [split_results[s]["CVD Positive"] for s in splits]
totals     = [split_results[s]["total"] for s in splits]

fig, ax = plt.subplots(figsize=(9, 5))
ax.bar(x - w/2, neg_counts, width=w, label="CVD Negative", color=LABEL_COLOURS[0], edgecolor="white")
ax.bar(x + w/2, pos_counts, width=w, label="CVD Positive", color=LABEL_COLOURS[1], edgecolor="white")
for i, total in enumerate(totals):
    ax.text(i, max(neg_counts[i], pos_counts[i]) + 50,
            f"n={total}", ha="center", fontsize=9, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([f"{s} ({totals[i]})" for i, s in enumerate(splits)])
ax.set_title("ODIR-5K — Train / Val / Test Split by CVD Label (70/15/15)", fontweight="bold", pad=12)
ax.set_ylabel("Number of Images")
ax.legend()
ax.yaxis.grid(True, linestyle="--", alpha=0.7)
ax.set_axisbelow(True)
save(fig, "odir_12_train_val_test_split.png")


print(f"\n[eda_odir] All 12 plots saved to: {OUT_DIR}")
