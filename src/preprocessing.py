"""Image preprocessing shared by training and camera inference."""

from __future__ import annotations

import numpy as np


IMAGE_SIZE = 28


def normalize_pixels(image: np.ndarray) -> np.ndarray:
    """Convert pixel values to float32 in the 0..1 range."""
    array = np.asarray(image, dtype=np.float32)
    if array.size == 0:
        raise ValueError("Cannot normalize an empty image")
    if float(array.max()) > 1.0:
        array = array / 255.0
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def reshape_flat_image(flat_pixels: np.ndarray) -> np.ndarray:
    """Convert one 784-pixel CSV row into a CNN-ready 28x28x1 image."""
    array = np.asarray(flat_pixels, dtype=np.float32)
    if array.size != IMAGE_SIZE * IMAGE_SIZE:
        raise ValueError(f"Expected 784 pixels, got {array.size}")
    image = array.reshape((IMAGE_SIZE, IMAGE_SIZE, 1))
    return normalize_pixels(image)


def preprocess_camera_crop(crop: np.ndarray) -> np.ndarray:
    """Prepare an OpenCV hand crop for the CNN.

    The model is trained on grayscale 28x28 Sign MNIST images, so live camera
    crops are converted to grayscale, resized, normalized, and batched.
    """
    import cv2

    if crop is None or crop.size == 0:
        raise ValueError("Cannot preprocess an empty camera crop")

    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    equalized = cv2.equalizeHist(resized.astype(np.uint8))
    image = normalize_pixels(equalized).reshape((1, IMAGE_SIZE, IMAGE_SIZE, 1))
    return image
