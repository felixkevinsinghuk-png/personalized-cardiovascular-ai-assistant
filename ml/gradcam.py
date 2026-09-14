# Grad-CAM heatmap generation using the pytorch-grad-cam library.
#
# Hooks into a target conv layer, runs a forward pass, overlays the activation
# map on the original image, and saves it to static/heatmaps/.
# Returns a relative path (from static/) for storing in the DB and referencing in templates.
#
# Note: pytorch_grad_cam can be unreliable on MPS (Apple Silicon) for certain model
# architectures. We run the CAM computation on CPU to guarantee correctness,
# then move the model back to the original device immediately after.

import logging
import os
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from config import Config

logger = logging.getLogger(__name__)


def generate_heatmap(
    model: torch.nn.Module,
    target_layer: torch.nn.Module,
    tensor: torch.Tensor,
    original_rgb: np.ndarray,
    prediction_id: int,
    model_name: str,
    device: torch.device,
) -> str:
    """
    Generate and save a Grad-CAM overlay for the given model and prediction.
    Returns the relative path from static/, e.g. "heatmaps/42_resnet50.png".
    Falls back to saving the plain original image if CAM fails.
    """
    os.makedirs(Config.HEATMAP_FOLDER, exist_ok=True)
    filename  = f"{prediction_id}_{model_name}.png"
    save_path = os.path.join(Config.HEATMAP_FOLDER, filename)

    try:
        # Run Grad-CAM on CPU — avoids MPS precision/op-support issues
        model.cpu()
        input_cpu = tensor.cpu()

        cam = GradCAM(model=model, target_layers=[target_layer])
        # targets=None → uses the class with the highest score automatically
        grayscale_cam = cam(input_tensor=input_cpu, targets=None)[0]

        # show_cam_on_image expects float32 RGB normalised to [0, 1]
        rgb_float     = original_rgb.astype(np.float32) / 255.0
        visualisation = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

        # show_cam_on_image returns RGB — OpenCV imwrite expects BGR
        cv2.imwrite(save_path, cv2.cvtColor(visualisation, cv2.COLOR_RGB2BGR))
        logger.info(f"Grad-CAM saved: {save_path}")

    except Exception as e:
        logger.warning(f"Grad-CAM failed ({e}). Saving plain image as fallback.")
        # Save the original image so the results page still has something to show
        cv2.imwrite(save_path, cv2.cvtColor(original_rgb, cv2.COLOR_RGB2BGR))

    finally:
        # Always move the model back to its original device
        model.to(device)

    return os.path.join("heatmaps", filename)
