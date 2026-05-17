"""
ARCHIVO: landmarks.py
MÓDULO: Landmarks Estáticos
DESCRIPCIÓN: Extrae y normaliza puntos de MediaPipe para reconocer letras estáticas del ASL.
PARTE DE LA APP QUE CONTROLA: Conversión de una mano detectada en un vector numérico para el clasificador.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


# MediaPipe Hands entrega 21 puntos; cada punto aporta coordenadas x e y.
LANDMARK_COUNT = 21
LANDMARK_VECTOR_SIZE = LANDMARK_COUNT * 2
_EPSILON = 1e-6


class LandmarkListLike(Protocol):
    """Interfaz mínima de una lista de landmarks compatible con MediaPipe."""

    landmark: list[object]


@dataclass
class LandmarkExtraction:
    """
    Clase: Resultado de extraer landmarks de una imagen.
    - features: vector normalizado de 42 valores.
    - bbox: caja delimitadora de la mano en píxeles.
    - raw_landmarks: landmarks originales de MediaPipe para dibujarlos si hace falta.
    """

    features: np.ndarray
    bbox: tuple[int, int, int, int]
    raw_landmarks: object


def landmarks_to_feature_vector(hand_landmarks: LandmarkListLike) -> np.ndarray:
    """
    Función: Convierte 21 landmarks de MediaPipe en un vector normalizado de 42 valores.
    La normalización resta el mínimo de la caja y divide por su ancho/alto para reducir
    el efecto de posición y tamaño de la mano dentro de la cámara.
    """
    points = hand_landmarks.landmark
    if len(points) != LANDMARK_COUNT:
        raise ValueError(f"Se esperaban {LANDMARK_COUNT} landmarks, se obtuvieron {len(points)}")

    xs = np.asarray([float(point.x) for point in points], dtype=np.float32)
    ys = np.asarray([float(point.y) for point in points], dtype=np.float32)

    width = max(float(xs.max() - xs.min()), _EPSILON)
    height = max(float(ys.max() - ys.min()), _EPSILON)

    features: list[float] = []
    for x, y in zip(xs, ys):
        features.append(float((x - xs.min()) / width))
        features.append(float((y - ys.min()) / height))

    return np.asarray(features, dtype=np.float32)


def bbox_from_landmarks(hand_landmarks: LandmarkListLike, frame_width: int, frame_height: int, padding: int = 35) -> tuple[int, int, int, int]:
    """
    Función: Calcula una caja delimitadora en píxeles a partir de landmarks normalizados.
    El margen ayuda a que la visualización no quede pegada a los dedos.
    """
    points = hand_landmarks.landmark
    if len(points) != LANDMARK_COUNT:
        raise ValueError(f"Se esperaban {LANDMARK_COUNT} landmarks, se obtuvieron {len(points)}")

    xs = [float(point.x) for point in points]
    ys = [float(point.y) for point in points]
    x1 = max(0, int(min(xs) * frame_width) - padding)
    y1 = max(0, int(min(ys) * frame_height) - padding)
    x2 = min(frame_width, int(max(xs) * frame_width) + padding)
    y2 = min(frame_height, int(max(ys) * frame_height) + padding)
    return x1, y1, x2, y2


class MediaPipeLandmarkExtractor:
    """
    Clase: Adaptador de MediaPipe Hands para imágenes estáticas o fotogramas de cámara.
    Devuelve solo una mano para mantener el detector simple y consistente.
    """

    def __init__(self, static_image_mode: bool = True, min_detection_confidence: float = 0.45) -> None:
        """Función: Inicializa MediaPipe con parámetros adecuados para extracción de landmarks."""
        import mediapipe as mp

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=0.50,
        )

    def extract(self, frame: np.ndarray) -> LandmarkExtraction | None:
        """
        Función: Extrae landmarks normalizados desde un fotograma BGR.
        Retorna None cuando MediaPipe no encuentra una mano.
        """
        if frame is None or frame.size == 0:
            return None

        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None

        hand_landmarks = result.multi_hand_landmarks[0]
        features = landmarks_to_feature_vector(hand_landmarks)
        bbox = bbox_from_landmarks(hand_landmarks, width, height)
        return LandmarkExtraction(features=features, bbox=bbox, raw_landmarks=hand_landmarks)

    def close(self) -> None:
        """Función: Libera los recursos internos de MediaPipe."""
        self._hands.close()
