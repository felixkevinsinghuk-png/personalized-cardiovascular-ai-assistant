# Score fusion and risk level assignment.
#
# The raw fused score is a weighted average of the DR grade and CVD score:
#   fused = (0.40 × norm_dr) + (0.60 × cvd_score)
#
# If clinical history is provided:
#   fused = (0.30 × norm_dr) + (0.45 × cvd_score) + (0.25 × ccs)
#
# A 1.25× sensitivity multiplier is applied before threshold assignment.
# This deliberately biases the system toward caution — telling a sick person
# they're fine is far worse than a false positive.
#
# Risk thresholds (after multiplier):
#   < 0.20            → Low
#   0.20 – 0.35       → Borderline
#   0.35 – 0.65       → Moderate
#   > 0.65            → High

from config import Config


DR_LABELS = {
    0: "No DR (Normal)",
    1: "Mild DR",
    2: "Moderate DR",
    3: "Severe DR",
    4: "Proliferative DR",
}


def get_dr_label(dr_grade: int) -> str:
    return DR_LABELS.get(dr_grade, "Unknown")


def fuse_scores(dr_grade: int, cvd_score: float, ccs: float = None) -> tuple[float, str]:
    """
    Combine DR grade, CVD score, and optional clinical context into a single risk score.
    Returns the raw fused score (before multiplier) and the final risk level string.
    """
    norm_dr = dr_grade / 4.0

    if ccs is not None:
        fused = (0.30 * norm_dr) + (0.45 * cvd_score) + (0.25 * ccs)
    else:
        fused = (0.40 * norm_dr) + (0.60 * cvd_score)

    # 1.25× multiplier to penalise false negatives
    adjusted = max(0.0, min(1.0, fused * 1.25))

    if adjusted < 0.20:
        risk_level = "Low"
    elif adjusted <= 0.35:
        risk_level = "Borderline"
    elif adjusted <= 0.65:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    return round(fused, 4), risk_level
