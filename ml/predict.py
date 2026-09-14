# Inference wrappers for ResNet-50 and EfficientNet-B4.
# Both functions expect a preprocessed [1, 3, 224, 224] tensor from preprocess.py.

import torch
import torch.nn.functional as F


def run_resnet(tensor: torch.Tensor, model: torch.nn.Module, device: torch.device) -> tuple[int, float]:
    """
    Run the DR grading model and return the predicted grade (0–4) and its softmax probability.
    """
    tensor = tensor.to(device)
    with torch.no_grad():
        probs = F.softmax(model(tensor), dim=1)

    dr_grade       = int(probs.argmax(dim=1).item())
    dr_probability = float(probs[0, dr_grade].item())
    return dr_grade, dr_probability


def run_efficientnet(tensor: torch.Tensor, model: torch.nn.Module, device: torch.device) -> float:
    """
    Run the CVD risk model and return a probability in [0.0, 1.0].
    Higher means more cardiovascular risk.
    """
    tensor = tensor.to(device)
    with torch.no_grad():
        cvd_score = float(torch.sigmoid(model(tensor)).item())
    return cvd_score
