#!/usr/bin/env bash
SSD_ROOT="$(cd "$(dirname "$0")" && pwd)"

CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate retinal_xai

echo "Running Flask app..."
cd "$SSD_ROOT"
python app.py
