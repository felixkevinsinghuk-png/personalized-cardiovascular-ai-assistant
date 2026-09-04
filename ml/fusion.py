"""
ml/fusion.py
Score fusion and risk level assignment.

Combines the DR grade from ResNet-50 and the CVD score from EfficientNet-B4
into a single fused cardiovascular risk score using a weighted formula.

    fused_score = (0.40 × norm_dr) + (0.60 × cvd_score)

If clinical_context_score (ccs) is provided:
    fused_score = (0.30 × norm_dr) + (0.45 × cvd_score) + (0.25 × ccs)

Clinical Safety Multiplier:
    A 1.25× sensitivity multiplier is applied to the raw fused score before
    threshold assignment. This deliberately biases the system toward caution,
    penalising false negatives (telling a sick person they are fine is worse
    than telling a healthy person to consult a doctor).

Risk thresholds (applied after multiplier):
    adjusted_score < 0.20            → Low Risk
    0.20 ≤ adjusted_score ≤ 0.35     → Borderline Risk
    0.35 < adjusted_score ≤ 0.65     → Moderate Risk
    adjusted_score > 0.65            → High Risk
"""

from config import Config


# DR grade index → human-readable label
DR_LABELS = {
    0: "No DR (Normal)",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}


def get_dr_label(dr_grade: int) -> str:
    """
    Convert a DR grade index (0–4) to its human-readable label.

    Args:
        dr_grade: Integer grade predicted by ResNet-50.

    Returns:
        String label, e.g. "Mild DR".
    """
    return DR_LABELS.get(dr_grade, "Unknown")


def fuse_scores(dr_grade: int, cvd_score: float, ccs: float = None) -> tuple[float, str]:
    """
    Fuse the DR grade, CVD score, and optional CCS into a single risk score.

    Args:
        dr_grade:  Predicted DR grade index from ResNet-50 (0–4).
        cvd_score: CVD risk probability from EfficientNet-B4 (0.0–1.0).
        ccs:       Clinical Context Score from Mistral-7B (0.0-1.0). Optional.

    Returns:
        fused_score (float) Weighted combination clamped to [0.0, 1.0].
        risk_level  (str)   One of "Low", "Moderate", or "High".
    """
    # Normalise DR grade to [0, 1] range
    normalised_dr = dr_grade / 4.0

    # Weighted fusion
    if ccs is not None:
        fused = (0.30 * normalised_dr) + (0.45 * cvd_score) + (0.25 * ccs)
    else:
        fused = (0.40 * normalised_dr) + (0.60 * cvd_score)

    # Clinical Safety Multiplier — penalises false negatives.
    # A false negative (telling a sick person they are safe) is clinically
    # far more dangerous than a false positive. This 1.25× multiplier
    # deliberately pushes borderline scores into a higher risk tier,
    # erring on the side of caution.
    SENSITIVITY_MULTIPLIER = 1.25
    adjusted_fused = fused * SENSITIVITY_MULTIPLIER

    # Clamp to valid range
    adjusted_fused = max(0.0, min(1.0, adjusted_fused))

    # Assign risk level based on adjusted score.
    if adjusted_fused < 0.20:
        risk_level = "Low"
    elif adjusted_fused <= 0.35:
        risk_level = "Borderline"
    elif adjusted_fused <= 0.65:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    return round(fused, 4), risk_level
