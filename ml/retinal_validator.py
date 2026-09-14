# Strict retinal fundus image validation gate.
#
# Returns (is_valid, confidence, message) where:
#   is_valid   — True only if the image is confidently retinal
#   confidence — float 0.0–1.0 built from a multi-feature scoring system
#   message    — user-facing rejection text if is_valid is False
#
# Confidence threshold is conservative (0.45) to prefer false rejects
# over false accepts — letting a leaf through is far worse than blocking
# an unusually dark fundus image.
#
# Scoring approach:
#   - Each positive retinal signal adds to a score (max ~1.0)
#   - Hard-fail checks immediately return False for unambiguous non-retinal inputs
#   - Soft signals accumulate; if the total score < CONFIDENCE_THRESHOLD, reject

import cv2
import numpy as np

CONFIDENCE_THRESHOLD = 0.45  # conservative — err on the side of rejecting

_REJECTION_MESSAGES = {
    "unreadable":   "The uploaded file could not be read as an image. Please upload a valid JPEG or PNG.",
    "green":        "This system accepts retinal fundus photographs only. The uploaded image "
                    "appears to be a plant, leaf, or outdoor scene and cannot be analysed.",
    "green_hue":    "This system accepts retinal fundus photographs only. The uploaded image "
                    "contains predominantly green tones and does not resemble a retinal fundus image.",
    "blue":         "This system accepts retinal fundus photographs only. The uploaded image "
                    "appears to be a sky, water, or blue-dominant scene.",
    "white":        "This system accepts retinal fundus photographs only. The uploaded image "
                    "appears to be a document, screenshot, or overexposed photo.",
    "black":        "The uploaded image is too dark to analyse. "
                    "Please upload a well-lit colour retinal fundus photograph.",
    "grayscale":    "This system accepts retinal fundus photographs only. The uploaded image "
                    "appears to be grayscale or an X-ray. A colour retinal fundus image is required.",
    "low_confidence": (
        "This system accepts retinal fundus photographs only. "
        "The uploaded image does not appear to be a retinal fundus photograph — "
        "it does not show the expected retinal colour profile, circular fundus field, "
        "or warm-toned retinal background. "
        "Please upload a colour fundus photograph taken with a fundus camera."
    ),
}


def validate_retinal_image(image_path: str) -> tuple[bool, float, str]:
    """
    Validate whether an uploaded image is a retinal fundus photograph.

    Returns:
        is_valid   (bool)  — True only if confidence >= CONFIDENCE_THRESHOLD
        confidence (float) — 0.0–1.0 retinal likelihood score
        message    (str)   — empty string if valid, user-facing reason if not
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        return False, 0.0, _REJECTION_MESSAGES["unreadable"]

    img  = cv2.resize(bgr, (224, 224))
    rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv  = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    r = rgb[:, :, 0].astype(float)
    g = rgb[:, :, 1].astype(float)
    b = rgb[:, :, 2].astype(float)

    mean_r = r.mean()
    mean_g = g.mean()
    mean_b = b.mean()

    # ── HARD FAILS ──────────────────────────────────────────────────────────
    # These are unambiguous non-retinal inputs; return immediately.

    # Leaves / plants — green channel strongly dominates
    if mean_g > mean_r * 1.25 and mean_g > mean_b * 1.1:
        return False, 0.0, _REJECTION_MESSAGES["green"]

    # Sky / water / blue objects
    if mean_b > mean_r * 1.4 and mean_b > mean_g * 1.15:
        return False, 0.0, _REJECTION_MESSAGES["blue"]

    # Documents / overexposed — all channels very high
    if mean_r > 200 and mean_g > 200 and mean_b > 200:
        return False, 0.0, _REJECTION_MESSAGES["white"]

    # Near-black — not enough image content
    if mean_r < 12 and mean_g < 12 and mean_b < 12:
        return False, 0.0, _REJECTION_MESSAGES["black"]

    # Grayscale / X-ray — all channels nearly equal
    channel_spread = max(mean_r, mean_g, mean_b) - min(mean_r, mean_g, mean_b)
    if channel_spread < 8:
        return False, 0.0, _REJECTION_MESSAGES["grayscale"]

    # ── SOFT SCORING ────────────────────────────────────────────────────────
    # Each check contributes a score. Retinal images accumulate a high total.
    score = 0.0

    # Signal 1: Red channel dominance (retinal tissue is warm/orange)
    # Ideal: R is the highest channel
    if mean_r >= mean_g and mean_r >= mean_b:
        score += 0.20
    elif mean_r >= mean_g * 0.85:  # R close to G is still plausible
        score += 0.10

    # Signal 2: Warm colour temperature
    # Fundus images: R > B is nearly universal
    if mean_r > mean_b * 1.1:
        score += 0.10

    # Signal 3: HSV hue analysis on bright (non-border) pixels
    bright_mask = grey > 45
    n_bright = bright_mask.sum()
    if n_bright > 500:
        hues = hsv[:, :, 0][bright_mask]
        # Warm hues in OpenCV HSV (0–180): 0–22 (red/orange) and 158–180 (red wrap)
        warm_ratio  = float(((hues <= 22) | (hues >= 158)).mean())
        green_ratio = float(((hues >= 35) & (hues <= 85)).mean())
        cyan_ratio  = float(((hues >= 85) & (hues <= 130)).mean())

        # Hard fail: predominantly green-hued bright pixels
        if green_ratio > 0.35:
            return False, 0.0, _REJECTION_MESSAGES["green_hue"]

        # Reward warm hues
        if warm_ratio > 0.25:
            score += 0.20
        elif warm_ratio > 0.10:
            score += 0.10

        # Penalise cyan/teal (not typical of fundus)
        if cyan_ratio > 0.3:
            score -= 0.10

    # Signal 4: Dark circular border (fundus camera characteristic)
    # Retinal images: significant fraction of near-black pixels at the border
    dark_ratio        = float((grey < 25).mean())
    bright_ratio      = float((grey > 40).mean())

    if dark_ratio > 0.10:
        score += 0.15  # large dark border is a strong positive signal
    elif dark_ratio > 0.05:
        score += 0.08

    if bright_ratio < 0.05:
        # Almost no bright pixels — unusable image
        return False, 0.0, _REJECTION_MESSAGES["black"]

    # Signal 5: Bright region concentrated in the centre (fundus structure)
    # Check if brightness in the inner 50% of the image exceeds the outer ring
    h, w = grey.shape
    inner_radius = min(h, w) // 4
    centre_mask  = np.zeros_like(grey, dtype=bool)
    cv2.circle(centre_mask.view(np.uint8), (w // 2, h // 2), inner_radius, 1, -1)
    centre_mask = centre_mask.astype(bool)

    inner_brightness = grey[centre_mask].mean()
    outer_brightness = grey[~centre_mask].mean()

    if inner_brightness > outer_brightness * 1.15:
        score += 0.15  # bright centre, dark ring — characteristic of fundus
    elif inner_brightness > outer_brightness:
        score += 0.05

    # Signal 6: Colour saturation in the bright region (fundus has rich colour)
    saturation = hsv[:, :, 1][bright_mask] if n_bright > 500 else np.array([0])
    mean_sat   = float(saturation.mean())
    if mean_sat > 60:
        score += 0.10  # well-saturated colour — not a faded or greyscale photo
    elif mean_sat < 15:
        score -= 0.10  # very low saturation — likely greyscale or washed out

    # Clamp to [0, 1]
    confidence = max(0.0, min(1.0, round(score, 3)))

    if confidence < CONFIDENCE_THRESHOLD:
        return False, confidence, _REJECTION_MESSAGES["low_confidence"]

    return True, confidence, ""
