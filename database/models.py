# SQLAlchemy ORM models — one class per table.
# If you change these, update database/schema.sql to match.

from datetime import datetime
from database.db import db


class Prediction(db.Model):
    """One row per uploaded retinal image. Everything else links back to this."""
    __tablename__ = "predictions"

    id                     = db.Column(db.Integer, primary_key=True, autoincrement=True)
    image_filename         = db.Column(db.String(255), nullable=False)
    dr_grade               = db.Column(db.SmallInteger, nullable=False)
    dr_label               = db.Column(db.String(50), nullable=False)
    dr_probability         = db.Column(db.Float, nullable=False)
    cvd_score              = db.Column(db.Float, nullable=False)
    fused_score            = db.Column(db.Float, nullable=False)
    clinical_history_text  = db.Column(db.Text, nullable=True)
    clinical_context_score = db.Column(db.Float, nullable=True)
    risk_level             = db.Column(db.String(50), nullable=False)
    retinal_caption        = db.Column(db.Text, nullable=True)
    created_at             = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    heatmaps     = db.relationship("Heatmap",     backref="prediction", lazy=True, cascade="all, delete-orphan")
    llm_reports  = db.relationship("LLMReport",   backref="prediction", lazy=True, cascade="all, delete-orphan")
    chat_history = db.relationship("ChatHistory", backref="prediction", lazy=True, cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id":                     self.id,
            "image_filename":         self.image_filename,
            "dr_grade":               self.dr_grade,
            "dr_label":               self.dr_label,
            "dr_probability":         round(self.dr_probability, 4),
            "cvd_score":              round(self.cvd_score, 4),
            "fused_score":            round(self.fused_score, 4),
            "clinical_history_text":  self.clinical_history_text,
            "clinical_context_score": round(self.clinical_context_score, 4) if self.clinical_context_score is not None else None,
            "risk_level":             self.risk_level,
            "retinal_caption":        self.retinal_caption,
            "created_at":             self.created_at.strftime("%Y-%m-%d %H:%M:%S"),
        }


class Heatmap(db.Model):
    """Stores the file path of a Grad-CAM overlay. One prediction can have several (one per model)."""
    __tablename__ = "heatmaps"

    id            = db.Column(db.Integer, primary_key=True, autoincrement=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey("predictions.id"), nullable=False)
    model_name    = db.Column(db.String(50), nullable=False)
    heatmap_path  = db.Column(db.String(512), nullable=False)
    created_at    = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


class LLMReport(db.Model):
    """Full text of the LLM-generated cardiovascular risk report."""
    __tablename__ = "llm_reports"

    id            = db.Column(db.Integer, primary_key=True, autoincrement=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey("predictions.id"), nullable=False)
    llm_model     = db.Column(db.String(100), nullable=False)
    report_text   = db.Column(db.Text, nullable=False)
    was_filtered  = db.Column(db.Boolean, nullable=False, default=False)
    created_at    = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


class ChatHistory(db.Model):
    """Every chatbot message exchange linked to a prediction."""
    __tablename__ = "chat_history"

    id            = db.Column(db.Integer, primary_key=True, autoincrement=True)
    prediction_id = db.Column(db.Integer, db.ForeignKey("predictions.id"), nullable=False)
    user_message  = db.Column(db.Text, nullable=False)
    llm_response  = db.Column(db.Text, nullable=False)
    was_filtered  = db.Column(db.Boolean, nullable=False, default=False)
    created_at    = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)


class ValidationFailure(db.Model):
    """
    Log of uploads rejected by the retinal image validation gate.
    Stored separately from predictions — these never appear in history.
    Useful for reviewing what kinds of non-retinal images users are uploading.
    """
    __tablename__ = "validation_failures"

    id               = db.Column(db.Integer, primary_key=True, autoincrement=True)
    original_filename = db.Column(db.String(255), nullable=True)   # user's original filename (not saved to disk)
    rejection_reason  = db.Column(db.String(100), nullable=False)  # short key e.g. "green", "low_confidence"
    confidence_score  = db.Column(db.Float, nullable=False, default=0.0)
    created_at        = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
