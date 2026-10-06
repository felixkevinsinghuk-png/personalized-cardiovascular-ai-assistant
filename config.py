# Central config — all tunable values live here so nothing is hard-coded elsewhere.

import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))


class Config:
    SECRET_KEY = os.environ.get("SECRET_KEY", "retinal-xai-secret-key-change-me")
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB upload limit
    ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}

    UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
    HEATMAP_FOLDER = os.path.join(BASE_DIR, "static", "heatmaps")

    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL",
        "mysql+pymysql://root:@127.0.0.1/retinal_xai_db"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    RESNET_WEIGHTS      = os.path.join(BASE_DIR, "models", "resnet50.pth")
    EFFICIENTNET_WEIGHTS = os.path.join(BASE_DIR, "models", "efficientnet.pth")

    # Fusion weights: w1 = DR grade contribution, w2 = CVD score contribution
    FUSION_W1 = 0.4
    FUSION_W2 = 0.6

    # Risk thresholds applied after the 1.25× safety multiplier
    RISK_LOW_THRESHOLD  = 0.20   # matches fusion.py deployed threshold (< 0.20 = Low)
    RISK_HIGH_THRESHOLD = 0.65   # matches fusion.py deployed threshold (> 0.65 = High)

    # Ollama (BioMistral locally via REST API)
    OLLAMA_URL     = "http://localhost:11434/api/generate"
    OLLAMA_MODEL   = "mistral"
    OLLAMA_TIMEOUT = 300  # seconds — increased for first-load from SSD

    # MedGemma via MLX (Apple Silicon)
    MEDGEMMA_MODEL      = "google/medgemma-4b-it"
    MEDGEMMA_MAX_TOKENS = 512

    PRIMARY_LLM = "biomistral"  # switch to "medgemma" if preferred

    # Training data paths — only used by training scripts, not the app
    APTOS_DATA_DIR = os.path.join(BASE_DIR, "data", "aptos2019")
    ODIR_DATA_DIR  = os.path.join(BASE_DIR, "data", "odir5k")

    # Training hyperparameters
    BATCH_SIZE               = 32
    LEARNING_RATE            = 1e-4
    EARLY_STOPPING_PATIENCE  = 5
    NUM_CLASSES_RESNET       = 5   # DR grades 0–4
    NUM_CLASSES_EFFICIENTNET = 1   # Binary CVD risk output
