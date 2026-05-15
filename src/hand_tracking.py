"""
ARCHIVO: hand_tracking.py
MÓDULO: Seguimiento de Manos
DESCRIPCIÓN: Adaptadores de seguimiento de manos utilizando MediaPipe y un modo alternativo basado en OpenCV.
PARTE DE LA APP QUE CONTROLA: Detección y recorte de la región de la mano en cada fotograma de la cámara.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np


# Tipos de datos: caja delimitadora y punto de coordenadas
BBox = tuple[int, int, int, int]
Point = tuple[float, float]


@dataclass
class HandDetection:
    """
    Clase: Estructura de datos que guarda el resultado de una detección de mano.
    - bbox: coordenadas de la caja delimitadora (x1, y1, x2, y2)
    - crop: imagen recortada de la mano
    - motion_point: punta del dedo índice para detectar J o Z
    - status: texto descriptivo del modo de detección activo
    """
    bbox: BBox
    crop: np.ndarray
    motion_point: Point | None
    status: str


class HandTracker(Protocol):
    """
    Clase: Protocolo (interfaz) que deben cumplir todos los detectores de manos.
    Define los métodos obligatorios: detect, draw y close.
    """
    status: str

    def detect(self, frame: np.ndarray) -> HandDetection | None:
        ...

    def draw(self, frame: np.ndarray, detection: HandDetection | None) -> None:
        ...

    def close(self) -> None:
        ...


def fixed_roi_bbox(frame_width: int, frame_height: int) -> BBox:
    """
    Función: Calcula las coordenadas de un cuadro fijo centrado dentro del fotograma.
    Se usa como zona de interés (ROI) cuando MediaPipe no está disponible.
    Parámetros: frame_width (int), frame_height (int) - dimensiones del cuadro.
    """
    # El cuadro mide el 50% del lado más corto del fotograma
    size = int(min(frame_width, frame_height) * 0.50)
    
    # Centra el cuadro horizontalmente y lo sube ligeramente en vertical
    x1 = max(0, (frame_width - size) // 2)
    y1 = max(0, (frame_height - size) // 2 - int(frame_height * 0.08))
    x2 = min(frame_width, x1 + size)
    y2 = min(frame_height, y1 + size)
    return x1, y1, x2, y2


def fingertip_from_mask(mask: np.ndarray, roi_origin: tuple[int, int], frame_size: tuple[int, int]) -> Point | None:
    """
    Función: Estima la posición de la punta del dedo buscando el punto más alto del contorno mayor.
    Se normaliza la posición (0 a 1) respecto al tamaño total del fotograma.
    Parámetros:
      - mask (np.ndarray): máscara binaria del color de piel.
      - roi_origin (tuple): esquina superior izquierda del ROI en el fotograma original.
      - frame_size (tuple): ancho y alto del fotograma completo.
    """
    # Encuentra contornos en la máscara
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    # Toma el contorno más grande (se asume que es la mano)
    largest = max(contours, key=cv2.contourArea)
    
    # Descarta contornos muy pequeños que probablemente son ruido
    if cv2.contourArea(largest) < 120:
        return None

    # Identifica el punto más alto del contorno (la punta del dedo)
    topmost = tuple(largest[largest[:, :, 1].argmin()][0])
    frame_width, frame_height = frame_size
    
    # Convierte la coordenada local del ROI a coordenadas normalizadas del fotograma
    x = (roi_origin[0] + int(topmost[0])) / frame_width
    y = (roi_origin[1] + int(topmost[1])) / frame_height
    return x, y


class MediaPipeHandTracker:
    """
    Clase: Detector de manos usando la librería MediaPipe (modo principal).
    Detecta landmarks del esqueleto de la mano con alta precisión.
    """

    def __init__(self) -> None:
        """
        Inicializa el modelo de MediaPipe configurado para detectar máximo 1 mano.
        """
        import mediapipe as mp

        self._mp_hands = mp.solutions.hands
        self._mp_drawing = mp.solutions.drawing_utils
        
        # Configura el modelo con umbrales de confianza mínimos del 65%
        self._hands = self._mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.65,
        )
        self.status = "MediaPipe"
        self._last_landmarks = None

    def detect(self, frame: np.ndarray) -> HandDetection | None:
        """
        Función: Procesa un fotograma y devuelve la detección de la mano con su recorte.
        Retorna None si no hay ninguna mano visible.
        Parámetros: frame (np.ndarray) - fotograma en formato BGR.
        """
        height, width = frame.shape[:2]
        
        # Convierte de BGR a RGB porque MediaPipe requiere ese formato
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        self._last_landmarks = None
        
        # Si no hay resultados, no hay mano detectada
        if not result.multi_hand_landmarks:
            return None

        # Toma los landmarks de la primera mano encontrada
        landmarks = result.multi_hand_landmarks[0]
        self._last_landmarks = landmarks
        
        # Calcula la caja delimitadora a partir de los landmarks con un margen extra
        xs = [landmark.x for landmark in landmarks.landmark]
        ys = [landmark.y for landmark in landmarks.landmark]
        padding = 35
        x1 = max(0, int(min(xs) * width) - padding)
        y1 = max(0, int(min(ys) * height) - padding)
        x2 = min(width, int(max(xs) * width) + padding)
        y2 = min(height, int(max(ys) * height) + padding)
        
        # El punto de movimiento es la punta del dedo índice (landmark 8)
        index_tip = landmarks.landmark[8]
        return HandDetection(
            bbox=(x1, y1, x2, y2),
            crop=frame[y1:y2, x1:x2],
            motion_point=(index_tip.x, index_tip.y),
            status=self.status,
        )

    def draw(self, frame: np.ndarray, detection: HandDetection | None) -> None:
        """
        Función: Dibuja la caja delimitadora y el esqueleto de los landmarks sobre el fotograma.
        Parámetros: frame (np.ndarray), detection (HandDetection | None).
        """
        if detection:
            x1, y1, x2, y2 = detection.bbox
            # Dibuja la caja en verde (color en BGR)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (39, 174, 96), 2)
        if self._last_landmarks is not None:
            # Dibuja el esqueleto de conexiones de la mano
            self._mp_drawing.draw_landmarks(frame, self._last_landmarks, self._mp_hands.HAND_CONNECTIONS)

    def close(self) -> None:
        """
        Función: Libera los recursos del modelo de MediaPipe.
        """
        self._hands.close()


class FixedRoiHandTracker:
    """
    Clase: Detector alternativo basado solo en OpenCV cuando MediaPipe no puede cargarse.
    Usa una zona fija en el centro del fotograma y segmentación por color de piel.
    """

    def __init__(self, reason: str = "") -> None:
        self.status = "ROI fijo"
        self.reason = reason

    def detect(self, frame: np.ndarray) -> HandDetection | None:
        """
        Función: Detecta la mano usando el color de piel en un ROI fijo central.
        Parámetros: frame (np.ndarray) - fotograma en formato BGR.
        """
        height, width = frame.shape[:2]
        
        # Calcula las coordenadas del cuadro fijo
        x1, y1, x2, y2 = fixed_roi_bbox(width, height)
        crop = frame[y1:y2, x1:x2]
        
        # Convierte el recorte al espacio de color HSV para segmentar la piel
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        
        # Rango de color de piel en HSV
        lower_skin = np.array([0, 25, 35], dtype=np.uint8)
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Aplica operaciones morfológicas para limpiar el ruido en la máscara
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Estima la punta del dedo a partir de la máscara de piel
        point = fingertip_from_mask(mask, roi_origin=(x1, y1), frame_size=(width, height))
        return HandDetection(
            bbox=(x1, y1, x2, y2),
            crop=crop,
            motion_point=point,
            status=self.status,
        )

    def draw(self, frame: np.ndarray, detection: HandDetection | None) -> None:
        """
        Función: Dibuja el cuadro fijo y un mensaje de guía en el fotograma.
        Parámetros: frame (np.ndarray), detection (HandDetection | None).
        """
        if not detection:
            return
        x1, y1, x2, y2 = detection.bbox
        
        # Dibuja el cuadro en color cian y añade texto instructivo
        cv2.rectangle(frame, (x1, y1), (x2, y2), (92, 225, 230), 2)
        cv2.putText(frame, "Coloca la mano aqui", (x1 + 12, y1 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (92, 225, 230), 2)

    def close(self) -> None:
        """
        Función: Sin recursos que liberar en este modo. Método incluido por compatibilidad.
        """
        return None


def create_hand_tracker() -> HandTracker:
    """
    Función: Crea el detector de manos usando MediaPipe como primera opción.
    Si MediaPipe falla al inicializarse, vuelve automáticamente al modo de ROI fijo.
    """
    try:
        return MediaPipeHandTracker()
    except Exception as exc:
        print(f"MediaPipe no pudo iniciar; usando ROI fijo. Detalle: {exc}")
        return FixedRoiHandTracker(reason=str(exc))
