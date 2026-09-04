"""
ml/predict.py
Model inference functions — run preprocessed tensors through each CNN model.

Functions:
    run_resnet(tensor, model, device)       → (dr_grade, dr_probability)
    run_efficientnet(tensor, model, device) → cvd_score
"""

import torch
import torch.nn.functional as F


def run_resnet(
    tensor: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
) -> tuple[int, float]:
    """
    Run inference through the ResNet-50 DR grading model.

    Args:
        tensor: Preprocessed image tensor of shape [1, 3, 224, 224].
        model:  Loaded ResNet-50 model in eval mode.
        device: The compute device (mps / cuda / cpu).

    Returns:
        dr_grade        (int)   Predicted DR grade index, 0–4.
        dr_probability  (float) Softmax probability of the predicted grade.
    """
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)                          # Shape: [1, 5]
        probabilities = F.softmax(logits, dim=1)        # Shape: [1, 5]

    dr_grade = int(probabilities.argmax(dim=1).item())
    dr_probability = float(probabilities[0, dr_grade].item())

    return dr_grade, dr_probability


def run_efficientnet(
    tensor: torch.Tensor,
    model: torch.nn.Module,
    device: torch.device,
) -> float:
    """
    Run inference through the EfficientNet-B4 CVD risk model.

    Args:
        tensor: Preprocessed image tensor of shape [1, 3, 224, 224].
        model:  Loaded EfficientNet-B4 model in eval mode.
        device: The compute device (mps / cuda / cpu).

    Returns:
        cvd_score (float) CVD risk probability in range [0.0, 1.0].
                          Higher values indicate greater cardiovascular risk.
    """
    tensor = tensor.to(device)

    with torch.no_grad():
        logits = model(tensor)                          # Shape: [1, 1]
        cvd_score = float(torch.sigmoid(logits).item())

    return cvd_score
