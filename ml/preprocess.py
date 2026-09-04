"""
ml/preprocess.py
Retinal fundus image preprocessing pipeline.

Steps performed:
1. Load image from disk via OpenCV
2. Crop the large black circular border using contour detection
3. Resize to 224×224 pixels
4. Convert BGR → RGB
5. Apply ImageNet normalisation
6. Convert to a PyTorch tensor with batch dimension [1, 3, 224, 224]

Returns:
    tensor      (torch.Tensor)  Shape [1, 3, 224, 224] — model-ready input
    original_rgb (np.ndarray)   Shape [H, W, 3] uint8 — for Grad-CAM overlay
"""

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


# ImageNet normalisation constants
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

# Torchvision transform pipeline (applied after manual preprocessing)
_to_tensor_and_normalise = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def _crop_black_border(bgr_image: np.ndarray) -> np.ndarray:
    """
    Detect and crop the black circular border common in retinal fundus images.

    Strategy:
    - Convert to greyscale and apply a binary threshold
    - Find all external contours
    - Select the largest contour (the fundus circle itself)
    - Crop the image to the bounding rectangle of that contour

    If no large contour is found (the image has no notable border), the
    original image is returned unchanged.

    Args:
        bgr_image: OpenCV BGR image as a numpy array.

    Returns:
        Cropped BGR image.
    """
    grey = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Binary threshold: everything above intensity 10 is considered content
    _, thresh = cv2.threshold(grey, 10, 255, cv2.THRESH_BINARY)

    # Find external contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return bgr_image

    # Select the largest contour by area
    largest = max(contours, key=cv2.contourArea)

    # Get bounding rectangle and add a small margin (2 px) to avoid clipping
    x, y, w, h = cv2.boundingRect(largest)
    margin = 2
    x = max(0, x - margin)
    y = max(0, y - margin)
    x2 = min(bgr_image.shape[1], x + w + margin)
    y2 = min(bgr_image.shape[0], y + h + margin)

    cropped = bgr_image[y:y2, x:x2]
    return cropped


def preprocess_image(image_path: str) -> tuple[torch.Tensor, np.ndarray]:
    """
    Full preprocessing pipeline for a retinal fundus image.

    Args:
        image_path: Absolute or relative path to the image file (.jpg or .png).

    Returns:
        tensor       (torch.Tensor) Shape [1, 3, 224, 224], normalised for ImageNet models.
        original_rgb (np.ndarray)   Shape [224, 224, 3], uint8 RGB image for Grad-CAM overlay.

    Raises:
        FileNotFoundError: If the image file cannot be loaded by OpenCV.
        ValueError:        If the loaded array is not a valid image.
    """
    # 1. Load image
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not load image from: {image_path}")

    # 2. Crop black circular border
    bgr = _crop_black_border(bgr)

    # 3. Resize to 224×224
    bgr_resized = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA)

    # 4. Convert BGR → RGB
    rgb = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB)

    # Keep a copy of the uint8 RGB image for Grad-CAM overlay
    original_rgb = rgb.copy()

    # 5. Apply ImageNet normalisation via torchvision transforms
    pil_image = Image.fromarray(rgb)
    tensor = _to_tensor_and_normalise(pil_image)

    # 6. Add batch dimension: [3, 224, 224] → [1, 3, 224, 224]
    tensor = tensor.unsqueeze(0)

    return tensor, original_rgb
