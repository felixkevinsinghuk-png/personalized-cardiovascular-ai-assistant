"""
app.py
Main Flask application for the Retinal XAI CVD Risk Assessment tool.

Routes:
    GET  /               — Home page: image upload form
    POST /predict        — Run full pipeline and save to MySQL
    GET  /results/<id>   — Results page for a specific prediction
    GET  /history        — Table of all past predictions
    POST /chat           — Chatbot Q&A endpoint (returns JSON)

Start the app:
    conda activate retinal_xai
    flask run
    # or: python app.py
"""

import os
import uuid
import logging
from flask import (
    Flask, render_template, request, redirect,
    url_for, flash, jsonify,
)
from werkzeug.utils import secure_filename

from config import Config
from database.db import db
from database.models import Prediction, Heatmap, LLMReport, ChatHistory
from ml.preprocess import preprocess_image
from ml.predict import run_resnet, run_efficientnet
from ml.fusion import fuse_scores, get_dr_label
from ml.gradcam import generate_heatmap
from llm.prompt_builder import build_prompt, build_chat_prompt
from llm.biomistral import query_biomistral
from llm.medgemma import query_medgemma
from llm.clinical_context_extractor import extract_clinical_context_score
from llm.safety_filter import apply_safety_filter
from models.model_loader import load_models
from ml.caption_inference import generate_retinal_caption

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config.from_object(Config)

# Initialise SQLAlchemy with this app
db.init_app(app)

# Create upload and heatmap directories if they do not exist
os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.HEATMAP_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Load models at startup (once per process)
# ---------------------------------------------------------------------------
resnet_model = None
efficientnet_model = None
resnet_target_layer = None
effnet_target_layer = None
_device = None


def get_models():
    """Lazy-load models on first request (avoids loading during testing)."""
    global resnet_model, efficientnet_model, resnet_target_layer, effnet_target_layer, _device
    if resnet_model is None:
        logger.info("Loading CNN models...")
        resnet_model, efficientnet_model, resnet_target_layer, effnet_target_layer, _device = load_models()
        logger.info("Models loaded successfully.")
    return resnet_model, efficientnet_model, resnet_target_layer, effnet_target_layer, _device


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def allowed_file(filename: str) -> bool:
    """Return True if the file extension is in the allowed set."""
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS
    )


def query_llm(prompt: str) -> str:
    """
    Send prompt to Mistral via Ollama.
    """
    response = query_biomistral(prompt)
    if response.startswith("[BioMistral"):
        logger.error("Mistral unavailable. Is Ollama running? Run: ollama serve")
    return response


# ---------------------------------------------------------------------------
# Route 1 — GET /
# ---------------------------------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    """Home page — display the image upload form."""
    return render_template("index.html")


# ---------------------------------------------------------------------------
# Route 2 — POST /predict
# ---------------------------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    """
    Receive the uploaded retinal image and run the full analysis pipeline.
    Saves all results to MySQL and redirects to the results page.
    """
    # 1. Validate file upload
    if "file" not in request.files:
        flash("No file was uploaded. Please select a retinal image.", "danger")
        return redirect(url_for("index"))

    file = request.files["file"]

    if file.filename == "":
        flash("No file selected. Please choose a retinal image.", "danger")
        return redirect(url_for("index"))

    if not allowed_file(file.filename):
        flash("Invalid file type. Please upload a .jpg or .png image.", "danger")
        return redirect(url_for("index"))

    # 2. Save uploaded file with a unique name to avoid collisions
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_filename = f"{uuid.uuid4().hex}.{ext}"
    upload_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
    file.save(upload_path)
    logger.info(f"Image saved: {upload_path}")

    try:
        # 3. Preprocess image
        tensor, original_rgb = preprocess_image(upload_path)
        
        # 3.5 Extract clinical context (if provided)
        clinical_history = request.form.get("clinical_history", "").strip()
        if clinical_history:
            ccs = extract_clinical_context_score(clinical_history)
        else:
            clinical_history = None
            ccs = None

        # 4. Load models (lazy, cached after first call)
        resnet, efficientnet, resnet_tl, effnet_tl, device = get_models()

        # 5. Run inference
        dr_grade, dr_probability = run_resnet(tensor, resnet, device)
        cvd_score = run_efficientnet(tensor, efficientnet, device)

        # 6. Generate Retinal Caption
        retinal_caption = generate_retinal_caption(tensor, dr_grade)

        # 7. Fusion
        fused_score, risk_level = fuse_scores(dr_grade, cvd_score, ccs)
        dr_label = get_dr_label(dr_grade)

        logger.info(
            f"Prediction: DR={dr_grade} ({dr_label}), CVD={cvd_score:.4f}, "
            f"Fused={fused_score:.4f}, Risk={risk_level}"
        )

        # 7. Save prediction to DB (need ID before generating heatmap filename)
        prediction = Prediction(
            image_filename=unique_filename,
            dr_grade=dr_grade,
            dr_label=dr_label,
            dr_probability=dr_probability,
            cvd_score=cvd_score,
            fused_score=fused_score,
            clinical_history_text=clinical_history,
            clinical_context_score=ccs,
            risk_level=risk_level,
            retinal_caption=retinal_caption,
        )
        db.session.add(prediction)
        db.session.flush()  # Assigns prediction.id without committing
        prediction_id = prediction.id

        # 8. Generate Grad-CAM heatmap (ResNet-50)
        heatmap_rel_path = generate_heatmap(
            model=resnet,
            target_layer=resnet_tl,
            tensor=tensor,
            original_rgb=original_rgb,
            prediction_id=prediction_id,
            model_name="resnet50",
            device=device,
        )

        heatmap_record = Heatmap(
            prediction_id=prediction_id,
            model_name="resnet50",
            heatmap_path=heatmap_rel_path,
        )
        db.session.add(heatmap_record)

        # 9. Generate LLM report
        prompt = build_prompt(
            dr_grade, 
            dr_label, 
            cvd_score, 
            fused_score, 
            risk_level,
            clinical_history=clinical_history,
            ccs=ccs,
            retinal_caption=retinal_caption
        )
        raw_response = query_llm(prompt)
        safe_report, was_filtered = apply_safety_filter(raw_response)

        llm_report = LLMReport(
            prediction_id=prediction_id,
            llm_model=Config.PRIMARY_LLM,
            report_text=safe_report,
            was_filtered=was_filtered,
        )
        db.session.add(llm_report)

        # 10. Commit everything to the database
        db.session.commit()
        logger.info(f"All records committed. Prediction ID: {prediction_id}")

        return redirect(url_for("results", prediction_id=prediction_id))

    except Exception as e:
        db.session.rollback()
        logger.error(f"Pipeline error: {e}", exc_info=True)
        flash(f"An error occurred during analysis: {str(e)}", "danger")
        return redirect(url_for("index"))


# ---------------------------------------------------------------------------
# Route 3 — GET /results/<id>
# ---------------------------------------------------------------------------
@app.route("/results/<int:prediction_id>", methods=["GET"])
def results(prediction_id: int):
    """Display the analysis results for a specific prediction."""
    prediction = Prediction.query.get_or_404(prediction_id)
    heatmap = Heatmap.query.filter_by(
        prediction_id=prediction_id, model_name="resnet50"
    ).first()
    report = LLMReport.query.filter_by(prediction_id=prediction_id).first()

    return render_template(
        "results.html",
        prediction=prediction,
        heatmap=heatmap,
        report=report,
    )


# ---------------------------------------------------------------------------
# Route 4 — GET /history
# ---------------------------------------------------------------------------
@app.route("/history", methods=["GET"])
def history():
    """Display a table of all past predictions, newest first."""
    predictions = Prediction.query.order_by(Prediction.created_at.desc()).all()
    return render_template("history.html", predictions=predictions)


# ---------------------------------------------------------------------------
# Route 5 — POST /chat
# ---------------------------------------------------------------------------
@app.route("/chat", methods=["POST"])
def chat():
    """
    Handle chatbot Q&A messages.
    Expects JSON: { "message": "...", "prediction_id": 42 }
    Returns JSON: { "response": "...", "was_filtered": false }
    """
    data = request.get_json(silent=True)
    if not data or "message" not in data or "prediction_id" not in data:
        return jsonify({"error": "Invalid request body."}), 400

    user_message = data["message"].strip()
    prediction_id = int(data["prediction_id"])

    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    # Fetch prediction context
    prediction = Prediction.query.get_or_404(prediction_id)

    # Build contextualised chatbot prompt
    prompt = build_chat_prompt(
        user_message=user_message,
        dr_grade=prediction.dr_grade,
        dr_label=prediction.dr_label,
        cvd_score=prediction.cvd_score,
        fused_score=prediction.fused_score,
        risk_level=prediction.risk_level,
    )

    # Generate and filter response
    raw_response = query_llm(prompt)
    safe_response, was_filtered = apply_safety_filter(raw_response)

    # Save exchange to chat_history
    chat_entry = ChatHistory(
        prediction_id=prediction_id,
        user_message=user_message,
        llm_response=safe_response,
        was_filtered=was_filtered,
    )
    db.session.add(chat_entry)
    db.session.commit()

    return jsonify({"response": safe_response, "was_filtered": was_filtered})


# ---------------------------------------------------------------------------
# Database initialisation helper
# ---------------------------------------------------------------------------
@app.cli.command("init-db")
def init_db():
    """Flask CLI command: flask init-db — creates all tables."""
    with app.app_context():
        db.create_all()
        print("Database tables created.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, host="0.0.0.0", port=5001)
