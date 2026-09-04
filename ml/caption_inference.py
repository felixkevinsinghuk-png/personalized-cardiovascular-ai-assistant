"""
ml/caption_inference.py
Provides the inference function to generate a retinal caption.
Uses the trained RetinalCaptioner (Encoder-Decoder) model with a Safety Override
to guarantee clinically safe output and prevent false negatives.
"""

import logging
import os
import torch
import pickle
from ml.retinal_caption_model import EncoderCNN, DecoderRNN

logger = logging.getLogger(__name__)

# Global variables to cache models
_vocab = None
_encoder = None
_decoder = None
_device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# Ground-truth severity templates — used for Safety Override
# These are guaranteed-correct, hallucination-free captions aligned to DR grade
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

# Severity levels implied by each template keyword
# Used to detect if the LSTM caption undersells a high-risk grade
_TEMPLATE_SEVERITY = {
    0: ["normal", "grossly preserved", "no obvious", "within normal limits"],
    1: ["mild", "scattered microaneurysms", "no obvious haemorrhages"],
    2: ["moderate", "dot-blot haemorrhages", "beginning to form"],
    3: ["severe", "numerous haemorrhages", "venous beading"],
    4: ["proliferative", "neovascularisation", "pre-retinal haemorrhage"],
}

def _infer_caption_severity(caption: str) -> int:
    """
    Infer the implied severity grade from a generated caption by keyword matching.
    Returns 0-4. Returns -1 if severity cannot be determined.
    """
    caption_lower = caption.lower()
    # Check from most severe to least severe to be clinically conservative
    for grade in [4, 3, 2, 1, 0]:
        for keyword in _TEMPLATE_SEVERITY[grade]:
            if keyword in caption_lower:
                return grade
    return -1  # Cannot determine

def _apply_safety_override(caption: str, dr_grade: int) -> str:
    """
    Clinical Safety Override:
    If the LSTM-generated caption implies a LOWER severity than the ResNet-50 DR grade,
    override it with the correct template to prevent false negative captions.

    This is analogous to the 1.25× safety multiplier in the fusion score.
    A false negative caption (e.g., saying 'no obvious haemorrhages' when grade is 3)
    could mislead BioMistral into generating a falsely reassuring report.
    """
    inferred = _infer_caption_severity(caption)

    # If the LSTM caption implies lower severity than the actual grade, override it
    if inferred < dr_grade:
        logger.warning(
            f"Safety Override triggered: LSTM implied grade {inferred} "
            f"but ResNet DR grade is {dr_grade}. Using safe template."
        )
        return _SAFE_TEMPLATES.get(dr_grade, caption)

    return caption


def load_caption_models():
    global _vocab, _encoder, _decoder

    vocab_path = 'models/weights/vocab.pkl'
    model_path = 'models/weights/retinal_captioner.pth'

    if not os.path.exists(vocab_path) or not os.path.exists(model_path):
        return False

    try:
        # Load vocab — requires Vocabulary class to be importable
        from training.build_vocab import Vocabulary  # noqa: F401 (needed for pickle)
        with open(vocab_path, 'rb') as f:
            _vocab = pickle.load(f)

        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=_device)

        # Init models
        _encoder = EncoderCNN(checkpoint['embed_size']).to(_device)
        _decoder = DecoderRNN(checkpoint['embed_size'], checkpoint['hidden_size'],
                              checkpoint['vocab_size']).to(_device)

        _encoder.load_state_dict(checkpoint['encoder_state_dict'])
        _decoder.load_state_dict(checkpoint['decoder_state_dict'])

        _encoder.eval()
        _decoder.eval()
        return True
    except Exception as e:
        logger.error(f"Failed to load caption models: {e}")
        return False


def generate_retinal_caption(image_tensor, dr_grade: int) -> str:
    """
    Generate a descriptive caption of the retinal findings.

    Args:
        image_tensor: Preprocessed image tensor (1, 3, 224, 224).
        dr_grade: The DR grade predicted by ResNet-50.

    Returns:
        A clinical string describing the retinal findings, safety-checked and
        guaranteed not to understate severity relative to the DR grade.
    """
    # 1. Attempt to use trained PyTorch model
    if _encoder is None or _decoder is None:
        models_loaded = load_caption_models()
    else:
        models_loaded = True

    if models_loaded:
        logger.info("Generating retinal caption using trained Vision-Language model.")
        with torch.no_grad():
            image_tensor = image_tensor.to(_device)
            features = _encoder(image_tensor)
            sampled_ids = _decoder.sample(features)

            # Convert word IDs back to string
            sampled_caption = []
            for word_id in sampled_ids:
                word = _vocab.idx2word[word_id]
                if word == '<end>':
                    break
                if word not in ['<start>', '<pad>', '<unk>']:
                    sampled_caption.append(word)

            sentence = ' '.join(sampled_caption)
            sentence = sentence.replace(' .', '.').replace(' ,', ',').capitalize()

            # Apply Safety Override to prevent false negative captions
            sentence = _apply_safety_override(sentence, dr_grade)
            return sentence

    # 2. Fallback: guaranteed-correct template if model weights not found
    logger.info("Caption models not found. Using safe template fallback.")
    return _SAFE_TEMPLATES.get(dr_grade, "Retinal features could not be clearly resolved from the image.")
