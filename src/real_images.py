"""Load real camera images organized by ASL letter folders."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from src.labels import LETTER_TO_LABEL, label_to_model_index
from src.preprocessing import IMAGE_SIZE, preprocess_camera_crop


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}
Cropper = Callable[[np.ndarray], np.ndarray | None]


class MediaPipeCropper:
    """Crop the detected hand from full camera frames."""

    def __init__(self) -> None:
        from src.hand_tracking import create_hand_tracker

        self._tracker = create_hand_tracker()

    def __call__(self, frame: np.ndarray) -> np.ndarray | None:
        detection = self._tracker.detect(frame)
        if detection is None:
            return None
        return detection.crop

    def close(self) -> None:
        self._tracker.close()


def _empty_dataset() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.empty((0, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32),
        np.empty((0,), dtype=np.int64),
    )


def iter_labeled_image_paths(root: Path) -> list[tuple[Path, int]]:
    """Return image paths paired with compact model label indexes."""
    compact_map = label_to_model_index()
    labeled_paths: list[tuple[Path, int]] = []
    if not root.exists():
        return labeled_paths

    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        letter = folder.name.strip().upper()
        sign_label = LETTER_TO_LABEL.get(letter)
        if sign_label is None:
            continue
        model_index = compact_map[sign_label]
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                labeled_paths.append((path, model_index))
    return labeled_paths


def load_real_image_dataset(root: Path, cropper: Cropper | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Load real hand images from letter folders for CNN training."""
    paths = iter_labeled_image_paths(root)
    if not paths:
        return _empty_dataset()

    owns_cropper = cropper is None
    active_cropper = cropper or MediaPipeCropper()
    images: list[np.ndarray] = []
    labels: list[int] = []

    try:
        for path, label in paths:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            crop = active_cropper(frame)
            if crop is None or crop.size == 0:
                continue
            try:
                prepared = preprocess_camera_crop(crop)[0]
            except ValueError:
                continue
            images.append(prepared)
            labels.append(label)
    finally:
        if owns_cropper and hasattr(active_cropper, "close"):
            active_cropper.close()  # type: ignore[attr-defined]

    if not images:
        return _empty_dataset()
    return np.stack(images).astype(np.float32), np.asarray(labels, dtype=np.int64)
