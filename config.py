"""
config.py
Central configuration for the Retinal XAI CVD Risk Assessment application.
All tunable values live here so no other module needs hard-coded constants.
"""

import os

# ---------------------------------------------------------------------------
# Base directory — root of the project
# ---------------------------------------------------------------------------
BASE_DIR = os.path.abspath(os.path.dirname(__file__))


# ---------------------------------------------------------------------------
# Flask configuration
# ---------------------------------------------------------------------------
class Config:
    # Secret key for session signing — change this in production
    SECRET_KEY = os.environ.get("SECRET_KEY", "retinal-xai-secret-key-change-me")

    # Maximum upload size: 16 MB
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024

    # Allowed image extensions
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

    # ---------------------------------------------------------------------------
    # File storage paths
    # ---------------------------------------------------------------------------
    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    HEATMAP_FOLDER = os.path.join(BASE_DIR, "static", "heatmaps")

    # ---------------------------------------------------------------------------
    # MySQL Database URI
    # Format: mysql+pymysql://<user>:<password>@<host>/<database>
    # Update the password field to match your local MySQL root password.
    # ---------------------------------------------------------------------------
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "mysql+pymysql://root:@127.0.0.1/retinal_xai_db"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # ---------------------------------------------------------------------------
    # Model weight file paths
    # ---------------------------------------------------------------------------
    RESNET_WEIGHTS = os.path.join(BASE_DIR, "models", "resnet50.pth")
    EFFICIENTNET_WEIGHTS = os.path.join(BASE_DIR, "models", "efficientnet.pth")

    # ---------------------------------------------------------------------------
    # Fusion weights and risk thresholds
    # ---------------------------------------------------------------------------
    # w1 applied to normalised DR grade (0–1), w2 applied to CVD score (0–1)
    FUSION_W1 = 0.4   # DR grade weight
    FUSION_W2 = 0.6   # CVD risk score weight

    # Risk level thresholds (applied after 1.25× sensitivity multiplier)
    # Low threshold lowered from 0.33 → 0.25 to penalise false negatives.
    RISK_LOW_THRESHOLD = 0.25
    RISK_HIGH_THRESHOLD = 0.67

    # ---------------------------------------------------------------------------
    # LLM configuration
    # ---------------------------------------------------------------------------
    # Mistral via Ollama (locally downloaded)
    OLLAMA_URL = "http://localhost:11434/api/generate"
    OLLAMA_MODEL = "mistral"
    OLLAMA_TIMEOUT = 300  # seconds (increased for initial SSD load)

    # MedGemma via MLX
    MEDGEMMA_MODEL = "google/medgemma-4b-it"
    MEDGEMMA_MAX_TOKENS = 512

    # Which LLM to use as primary: "biomistral" or "medgemma"
    PRIMARY_LLM = "biomistral"

    # ---------------------------------------------------------------------------
    # Training data paths (used only by training scripts)
    # ---------------------------------------------------------------------------
    APTOS_DATA_DIR = os.path.join(BASE_DIR, "data", "aptos2019")
    ODIR_DATA_DIR = os.path.join(BASE_DIR, "data", "odir5k")

    # Training hyperparameters
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EARLY_STOPPING_PATIENCE = 5
    NUM_CLASSES_RESNET = 5    # DR grades 0–4
    NUM_CLASSES_EFFICIENTNET = 1  # Binary CVD risk output
