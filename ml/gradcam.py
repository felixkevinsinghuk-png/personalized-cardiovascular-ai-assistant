"""
ml/gradcam.py
Grad-CAM heatmap generation using the pytorch-grad-cam library.

For each model, this module:
1. Hooks into the identified target convolutional layer
2. Runs a forward pass to capture the activation map
3. Overlays the coloured heatmap on the original RGB image
4. Saves the result to static/heatmaps/

The saved filename follows the pattern:
    <prediction_id>_<model_name>.png
e.g.: 42_resnet50.png

The function returns the relative path (from the static/ folder root) so it
can be stored in the database and referenced in Jinja2 templates via url_for.
"""

import os
import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from config import Config


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
    Generate and save a Grad-CAM heatmap overlay for the given model.

    Args:
        model:          Loaded CNN model in eval mode.
        target_layer:   The target convolutional layer to hook into (e.g. resnet.layer4[-1]).
        tensor:         Preprocessed input tensor of shape [1, 3, 224, 224].
        original_rgb:   Original resized uint8 RGB image array of shape [224, 224, 3].
        prediction_id:  Database ID of the parent Prediction row (used in filename).
        model_name:     Short name string used in filename, e.g. "resnet50" or "efficientnet".
        device:         The compute device.

    Returns:
        heatmap_path (str): Relative path from the static/ folder,
                            e.g. "heatmaps/42_resnet50.png".
                            Store this string in the Heatmap DB record.
    """
    # Move tensor to the correct device
    input_tensor = tensor.to(device)

    # --- Build Grad-CAM ---
    # GradCAM requires the model and a list of target layers
    cam = GradCAM(model=model, target_layers=[target_layer])

    # targets=None tells the library to use the highest-scoring class automatically
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)

    # grayscale_cam has shape [batch, H, W]; take the first (and only) item
    grayscale_cam = grayscale_cam[0, :]  # Shape: [224, 224]

    # --- Overlay heatmap on original image ---
    # show_cam_on_image expects:
    #   img:        float32 RGB array normalised to [0, 1], shape [H, W, 3]
    #   grayscale_cam: float32 array in [0, 1], shape [H, W]
    rgb_float = original_rgb.astype(np.float32) / 255.0
    visualisation = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

    # --- Save heatmap image ---
    os.makedirs(Config.HEATMAP_FOLDER, exist_ok=True)
    filename = f"{prediction_id}_{model_name}.png"
    save_path = os.path.join(Config.HEATMAP_FOLDER, filename)

    # show_cam_on_image returns RGB — convert to BGR for OpenCV imwrite
    bgr_visualisation = cv2.cvtColor(visualisation, cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_path, bgr_visualisation)

    # Return relative path from static/ for use in templates and DB storage
    return os.path.join("heatmaps", filename)
