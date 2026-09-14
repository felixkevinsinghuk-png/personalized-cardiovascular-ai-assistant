# Preprocessing pipeline for retinal fundus images.
#
# Steps:
#   1. Load image via OpenCV
#   2. Crop the large black circular border
#   3. Resize to 224×224
#   4. Convert BGR → RGB
#   5. Normalise with ImageNet stats and convert to a PyTorch tensor [1, 3, 224, 224]

import cv2
import numpy as np
import torch
from torchvision import transforms
from PIL import Image


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_to_tensor_and_normalise = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
])


def _crop_black_border(bgr_image: np.ndarray) -> np.ndarray:
    """
    Crops the large black circular border that surrounds most retinal fundus images.
    Uses contour detection — finds the largest bright region and crops to its bounding box.
    Falls back to the original image if no suitable contour is found.
    """
    grey = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(grey, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return bgr_image

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)

    # Small margin to avoid clipping the disc edge
    margin = 2
    x  = max(0, x - margin)
    y  = max(0, y - margin)
    x2 = min(bgr_image.shape[1], x + w + margin)
    y2 = min(bgr_image.shape[0], y + h + margin)

    return bgr_image[y:y2, x:x2]


def preprocess_image(image_path: str) -> tuple[torch.Tensor, np.ndarray]:
    """
    Full preprocessing pipeline for a single retinal fundus image.

    Returns:
        tensor       — shape [1, 3, 224, 224], normalised for ImageNet models
        original_rgb — shape [224, 224, 3], uint8 RGB copy for Grad-CAM overlay
    """
    bgr = cv2.imread(image_path)
    if bgr is None:
        raise FileNotFoundError(f"Could not load image from: {image_path}")

    bgr         = _crop_black_border(bgr)
    bgr_resized = cv2.resize(bgr, (224, 224), interpolation=cv2.INTER_AREA)
    rgb         = cv2.cvtColor(bgr_resized, cv2.COLOR_BGR2RGB)
    original_rgb = rgb.copy()

    pil_image = Image.fromarray(rgb)
    tensor    = _to_tensor_and_normalise(pil_image)
    tensor    = tensor.unsqueeze(0)  # add batch dimension: [3,224,224] → [1,3,224,224]

    return tensor, original_rgb
