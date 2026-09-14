# Caption inference with a clinical Safety Override.
#
# First tries the trained LSTM captioner. If the generated caption implies a LOWER
# severity than the ResNet-predicted DR grade, it overrides with a guaranteed-correct
# template. This mirrors the 1.25× safety multiplier in the fusion score — we always
# err on the side of flagging, never understating.
# Falls back to templates entirely if no model weights are found.

import logging
import os
import torch
import pickle
from ml.retinal_caption_model import EncoderCNN, DecoderRNN

logger = logging.getLogger(__name__)

_vocab   = None
_encoder = None
_decoder = None
_device  = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Grade-appropriate fallback captions — used both as the Safety Override and as
# the plain fallback when model weights aren't available.
_SAFE_TEMPLATES = {
    0: (
        "Fundus image appears normal. Optic disc and macula are grossly preserved. "
        "No obvious microaneurysms, haemorrhages, or hard exudates are visible. "
        "Vascular caliber is within normal limits."
    ),
    1: (
        "Fundus image shows mild vascular changes. Scattered microaneurysms are present. "
        "No obvious haemorrhages or hard exudates. Optic disc appears normal."
    ),
    2: (
        "Fundus image shows moderate vascular abnormalities. "
        "Microaneurysms and dot-blot haemorrhages are visible. "
        "Mild vascular tortuosity is present. Hard exudates are beginning to form."
    ),
    3: (
        "Fundus image reveals severe vascular damage. "
        "Numerous haemorrhages and hard exudates are present throughout the retina. "
        "Significant vascular tortuosity and venous beading are visible."
    ),
    4: (
        "Fundus image shows proliferative changes. "
        "Neovascularisation is evident at the disc or elsewhere. "
        "Pre-retinal haemorrhage and severe exudation are present, indicating high risk."
    ),
}

# Keywords used to infer the severity implied by a generated caption
_TEMPLATE_SEVERITY = {
    0: ["normal", "grossly preserved", "no obvious", "within normal limits"],
    1: ["mild", "scattered microaneurysms", "no obvious haemorrhages"],
    2: ["moderate", "dot-blot haemorrhages", "beginning to form"],
    3: ["severe", "numerous haemorrhages", "venous beading"],
    4: ["proliferative", "neovascularisation", "pre-retinal haemorrhage"],
}


def _infer_caption_severity(caption: str) -> int:
    """Scan a caption for severity keywords, most-severe first. Returns -1 if none match."""
    caption_lower = caption.lower()
    for grade in [4, 3, 2, 1, 0]:
        if any(kw in caption_lower for kw in _TEMPLATE_SEVERITY[grade]):
            return grade
    return -1


def _apply_safety_override(caption: str, dr_grade: int) -> str:
    """If the LSTM caption understates severity relative to the DR grade, swap in the template."""
    inferred = _infer_caption_severity(caption)
    if inferred < dr_grade:
        logger.warning(
            f"Safety Override: LSTM implied grade {inferred} but DR grade is {dr_grade}. "
            "Using safe template."
        )
        return _SAFE_TEMPLATES.get(dr_grade, caption)
    return caption


def load_caption_models() -> bool:
    global _vocab, _encoder, _decoder

    vocab_path = "models/weights/vocab.pkl"
    model_path = "models/weights/retinal_captioner.pth"

    if not os.path.exists(vocab_path) or not os.path.exists(model_path):
        return False

    try:
        from training.build_vocab import Vocabulary  # noqa: needed for pickle
        with open(vocab_path, "rb") as f:
            _vocab = pickle.load(f)

        ckpt     = torch.load(model_path, map_location=_device)
        _encoder = EncoderCNN(ckpt["embed_size"]).to(_device)
        _decoder = DecoderRNN(ckpt["embed_size"], ckpt["hidden_size"], ckpt["vocab_size"]).to(_device)

        _encoder.load_state_dict(ckpt["encoder_state_dict"])
        _decoder.load_state_dict(ckpt["decoder_state_dict"])
        _encoder.eval()
        _decoder.eval()
        return True
    except Exception as e:
        logger.error(f"Failed to load caption models: {e}")
        return False


def generate_retinal_caption(image_tensor, dr_grade: int, validated: bool = False) -> str:
    """
    Generate a retinal findings caption, safety-checked against the DR grade.

    validated must be True — set by app.py only after the retinal validator passes.
    This prevents the captioner from ever running on a non-retinal image.
    """
    if not validated:
        # This should never happen in normal flow because app.py gates on
        # validate_retinal_image() before calling this. Raising here is a
        # belt-and-suspenders guard against future code changes.
        raise ValueError(
            "generate_retinal_caption called without retinal validation. "
            "Only call this function after validate_retinal_image() returns True."
        )
    global _encoder, _decoder

    if _encoder is None or _decoder is None:
        models_loaded = load_caption_models()
    else:
        models_loaded = True

    if models_loaded:
        logger.info("Generating caption using trained Vision-Language model.")
        with torch.no_grad():
            features    = _encoder(image_tensor.to(_device))
            sampled_ids = _decoder.sample(features)

        words = []
        for word_id in sampled_ids:
            word = _vocab.idx2word[word_id]
            if word == "<end>":
                break
            if word not in ["<start>", "<pad>", "<unk>"]:
                words.append(word)

        sentence = " ".join(words).replace(" .", ".").replace(" ,", ",").capitalize()
        return _apply_safety_override(sentence, dr_grade)

    logger.info("Caption model weights not found — using template fallback.")
    return _SAFE_TEMPLATES.get(dr_grade, "Retinal features could not be clearly resolved from the image.")
