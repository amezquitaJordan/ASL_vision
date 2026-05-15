"""
ARCHIVO: real_images.py
MÓDULO: Imágenes Reales
DESCRIPCIÓN: Carga imágenes reales de la cámara organizadas en subcarpetas por letra del ASL.
PARTE DE LA APP QUE CONTROLA: Preparación del dataset complementario de fotos propias para el entrenamiento.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from src.labels import LETTER_TO_LABEL, label_to_model_index
from src.preprocessing import IMAGE_SIZE, preprocess_camera_crop


# Extensiones de imagen aceptadas al leer la carpeta de imágenes reales
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}

# Tipo de función que recibe un fotograma y devuelve un recorte de mano (o None)
Cropper = Callable[[np.ndarray], np.ndarray | None]


class MediaPipeCropper:
    """
    Clase: Recorta la región de la mano detectada en un fotograma usando MediaPipe.
    Sirve para extraer automáticamente la mano de cada imagen del dataset real.
    """

    def __init__(self) -> None:
        from src.hand_tracking import create_hand_tracker
        self._tracker = create_hand_tracker()

    def __call__(self, frame: np.ndarray) -> np.ndarray | None:
        """
        Función: Detecta la mano en el fotograma y devuelve el recorte.
        Retorna None si no se encuentra ninguna mano.
        """
        detection = self._tracker.detect(frame)
        if detection is None:
            return None
        return detection.crop

    def close(self) -> None:
        """Función: Libera los recursos del rastreador de MediaPipe."""
        self._tracker.close()


def _empty_dataset() -> tuple[np.ndarray, np.ndarray]:
    """
    Función: Devuelve arreglos vacíos con la forma correcta.
    Se usa cuando no se encuentran imágenes válidas.
    """
    return (
        np.empty((0, IMAGE_SIZE, IMAGE_SIZE, 1), dtype=np.float32),
        np.empty((0,), dtype=np.int64),
    )


def iter_labeled_image_paths(root: Path) -> list[tuple[Path, int]]:
    """
    Función: Recorre subcarpetas y asocia cada imagen a su índice del modelo.
    Cada subcarpeta debe llamarse con la letra que contiene (ej: "A", "B", "C").
    Parámetros: root (Path) - ruta raíz del dataset.
    """
    compact_map = label_to_model_index()
    labeled_paths: list[tuple[Path, int]] = []

    if not root.exists():
        return labeled_paths

    # Itera sobre cada subcarpeta válida
    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        letter = folder.name.strip().upper()
        sign_label = LETTER_TO_LABEL.get(letter)

        # Salta carpetas cuyo nombre no corresponde a una letra válida
        if sign_label is None:
            continue

        model_index = compact_map[sign_label]

        # Agrega cada archivo de imagen junto a su índice de clase
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                labeled_paths.append((path, model_index))

    return labeled_paths


def load_real_image_dataset(root: Path, cropper: Cropper | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Función: Carga y preprocesa todas las imágenes reales para usarlas en entrenamiento.
    Usa MediaPipe para detectar y recortar la mano de cada imagen automáticamente.
    Parámetros:
      - root (Path): carpeta raíz con subcarpetas por letra.
      - cropper (Cropper | None): función opcional de recorte personalizada.
    """
    paths = iter_labeled_image_paths(root)

    # Si no hay imágenes, devuelve datasets vacíos
    if not paths:
        return _empty_dataset()

    # Crea un recortador si no se proporcionó uno externo
    owns_cropper = cropper is None
    active_cropper = cropper or MediaPipeCropper()

    images: list[np.ndarray] = []
    labels: list[int] = []

    try:
        for path, label in paths:
            # Lee la imagen desde el disco
            frame = cv2.imread(str(path))
            if frame is None:
                continue

            # Extrae el recorte de la mano
            crop = active_cropper(frame)
            if crop is None or crop.size == 0:
                continue

            # Preprocesa para que sea compatible con la CNN
            try:
                prepared = preprocess_camera_crop(crop)[0]
            except ValueError:
                continue

            images.append(prepared)
            labels.append(label)

    finally:
        # Cierra el recortador solo si fue creado en esta función
        if owns_cropper and hasattr(active_cropper, "close"):
            active_cropper.close()  # type: ignore[attr-defined]

    if not images:
        return _empty_dataset()

    return np.stack(images).astype(np.float32), np.asarray(labels, dtype=np.int64)
