"""
app.py — main Flask application for the Retinal XAI CVD Risk Assessment tool.

Routes:
    GET  /               Upload form
    POST /predict        Run the full analysis pipeline, save to MySQL, redirect to results
    GET  /results/<id>   Results page for a given prediction
    GET  /history        Table of all past predictions
    POST /chat           Chatbot Q&A (returns JSON)

Start the app:
    conda activate retinal_xai
    python app.py
"""

import os
import uuid
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename

from config import Config
from database.db import db
from database.models import Prediction, Heatmap, LLMReport, ChatHistory, ValidationFailure
from ml.preprocess import preprocess_image
from ml.retinal_validator import validate_retinal_image
from ml.predict import run_resnet, run_efficientnet
from ml.fusion import fuse_scores, get_dr_label
from ml.gradcam import generate_heatmap
from llm.prompt_builder import build_prompt, build_chat_prompt
from llm.biomistral import query_biomistral
from llm.clinical_context_extractor import extract_clinical_context_score
from llm.safety_filter import apply_safety_filter
from models.model_loader import load_models
from ml.caption_inference import generate_retinal_caption

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)

os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
os.makedirs(Config.HEATMAP_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Models are loaded once on first request and cached here
resnet_model = None
efficientnet_model = None
resnet_target_layer = None
effnet_target_layer = None
_device = None


def get_models():
    """Lazy-load both CNN models — only runs on the first prediction request."""
    global resnet_model, efficientnet_model, resnet_target_layer, effnet_target_layer, _device
    if resnet_model is None:
        logger.info("Loading CNN models...")
        resnet_model, efficientnet_model, resnet_target_layer, effnet_target_layer, _device = load_models()
        logger.info("Models loaded.")
    return resnet_model, efficientnet_model, resnet_target_layer, effnet_target_layer, _device


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in Config.ALLOWED_EXTENSIONS


def query_llm(prompt: str) -> str:
    response = query_biomistral(prompt)
    if response.startswith("[BioMistral"):
        logger.error("Mistral unavailable — is Ollama running? Run: ollama serve")
    return response


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Receive an uploaded retinal image and run the full analysis pipeline."""
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

    # Save with a UUID to avoid filename collisions
    ext = file.filename.rsplit(".", 1)[1].lower()
    unique_filename = f"{uuid.uuid4().hex}.{ext}"
    upload_path = os.path.join(Config.UPLOAD_FOLDER, unique_filename)
    file.save(upload_path)
    logger.info(f"Image saved: {upload_path}")

    # Validate before any model inference — delete the file immediately if it fails
    is_valid, confidence, rejection_msg = validate_retinal_image(upload_path)
    if not is_valid:
        os.remove(upload_path)
        logger.warning(f"Rejected upload (confidence={confidence:.3f}): {file.filename!r}")
        # Log the rejection separately — it must never appear in the predictions history
        try:
            failure = ValidationFailure(
                original_filename=file.filename[:255] if file.filename else None,
                rejection_reason=rejection_msg[:100],
                confidence_score=confidence,
            )
            db.session.add(failure)
            db.session.commit()
        except Exception as log_err:
            logger.error(f"Could not log validation failure: {log_err}")
        flash(rejection_msg, "danger")
        return redirect(url_for("index"))


    try:
        tensor, original_rgb = preprocess_image(upload_path)

        clinical_history = request.form.get("clinical_history", "").strip()
        if clinical_history:
            ccs = extract_clinical_context_score(clinical_history)
        else:
            clinical_history = None
            ccs = None

        resnet, efficientnet, resnet_tl, effnet_tl, device = get_models()

        dr_grade, dr_probability = run_resnet(tensor, resnet, device)
        cvd_score = run_efficientnet(tensor, efficientnet, device)
        retinal_caption = generate_retinal_caption(tensor, dr_grade, validated=True)

        fused_score, risk_level = fuse_scores(dr_grade, cvd_score, ccs)
        dr_label = get_dr_label(dr_grade)

        logger.info(
            f"Prediction: DR={dr_grade} ({dr_label}), CVD={cvd_score:.4f}, "
            f"Fused={fused_score:.4f}, Risk={risk_level}"
        )

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
        db.session.flush()  # get the ID before committing
        prediction_id = prediction.id

        heatmap_rel_path = generate_heatmap(
            model=resnet,
            target_layer=resnet_tl,
            tensor=tensor,
            original_rgb=original_rgb,
            prediction_id=prediction_id,
            model_name="resnet50",
            device=device,
        )

        db.session.add(Heatmap(
            prediction_id=prediction_id,
            model_name="resnet50",
            heatmap_path=heatmap_rel_path,
        ))

        prompt = build_prompt(
            dr_grade, dr_label, cvd_score, fused_score, risk_level,
            clinical_history=clinical_history,
            ccs=ccs,
            retinal_caption=retinal_caption,
        )
        raw_response = query_llm(prompt)
        safe_report, was_filtered = apply_safety_filter(raw_response)

        db.session.add(LLMReport(
            prediction_id=prediction_id,
            llm_model=Config.PRIMARY_LLM,
            report_text=safe_report,
            was_filtered=was_filtered,
        ))

        db.session.commit()
        logger.info(f"All records committed. Prediction ID: {prediction_id}")

        return redirect(url_for("results", prediction_id=prediction_id))

    except Exception as e:
        db.session.rollback()
        logger.error(f"Pipeline error: {e}", exc_info=True)
        flash(f"An error occurred during analysis: {str(e)}", "danger")
        return redirect(url_for("index"))


@app.route("/results/<int:prediction_id>", methods=["GET"])
def results(prediction_id: int):
    prediction = Prediction.query.get_or_404(prediction_id)
    heatmap = Heatmap.query.filter_by(prediction_id=prediction_id, model_name="resnet50").first()
    report = LLMReport.query.filter_by(prediction_id=prediction_id).first()
    return render_template("results.html", prediction=prediction, heatmap=heatmap, report=report)


@app.route("/history", methods=["GET"])
def history():
    predictions = Prediction.query.order_by(Prediction.created_at.desc()).all()
    return render_template("history.html", predictions=predictions)


@app.route("/chat", methods=["POST"])
def chat():
    """
    Chatbot endpoint for follow-up questions on a prediction.
    Expects JSON: { "message": "...", "prediction_id": 42 }
    """
    data = request.get_json(silent=True)
    if not data or "message" not in data or "prediction_id" not in data:
        return jsonify({"error": "Invalid request body."}), 400

    user_message = data["message"].strip()
    prediction_id = int(data["prediction_id"])

    if not user_message:
        return jsonify({"error": "Message cannot be empty."}), 400

    prediction = Prediction.query.get_or_404(prediction_id)

    prompt = build_chat_prompt(
        user_message=user_message,
        dr_grade=prediction.dr_grade,
        dr_label=prediction.dr_label,
        cvd_score=prediction.cvd_score,
        fused_score=prediction.fused_score,
        risk_level=prediction.risk_level,
    )

    raw_response = query_llm(prompt)
    safe_response, was_filtered = apply_safety_filter(raw_response)

    db.session.add(ChatHistory(
        prediction_id=prediction_id,
        user_message=user_message,
        llm_response=safe_response,
        was_filtered=was_filtered,
    ))
    db.session.commit()

    return jsonify({"response": safe_response, "was_filtered": was_filtered})


@app.cli.command("init-db")
def init_db():
    """Flask CLI: flask init-db — creates all tables."""
    with app.app_context():
        db.create_all()
        print("Database tables created.")


if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True, host="0.0.0.0", port=5001)
