"""
ARCHIVO: app.py
MÓDULO: Aplicación Principal
DESCRIPCIÓN: Detector en tiempo real del alfabeto ASL estático usando landmarks de MediaPipe.
PARTE DE LA APP QUE CONTROLA: Integración de la cámara, MediaPipe, modelo de landmarks y visualización.
"""

from __future__ import annotations

import os
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

# Asegura que el paquete raíz esté en la ruta del sistema
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.hand_tracking import create_hand_tracker
from src.labels import class_names


# Rutas base y ruta del modelo estático principal
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "asl_landmarks.joblib"


@dataclass(frozen=True)
class PredictionGateConfig:
    """
    Clase: Configuración de rechazo conservador para evitar mostrar letras dudosas.
    - confidence_threshold: probabilidad mínima de la mejor clase.
    - margin_threshold: diferencia mínima entre la mejor y segunda mejor clase.
    - stable_frames: cantidad de fotogramas consecutivos requeridos.
    """

    confidence_threshold: float = 0.80
    margin_threshold: float = 0.15
    stable_frames: int = 4


@dataclass(frozen=True)
class PredictionGateResult:
    """
    Clase: Resultado del filtro de predicción.
    - letter: letra aceptada o None si el sistema debe mostrar silencio.
    - confidence: confianza de la clase más probable.
    - margin: separación entre las dos clases más probables.
    """

    letter: str | None
    confidence: float
    margin: float


def env_float(name: str, default: float) -> float:
    """
    Función: Lee una variable de entorno como tipo flotante.
    Parámetros: name (str) - nombre de la variable, default (float) - valor por defecto.
    """
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    """
    Función: Lee una variable de entorno como tipo entero.
    Parámetros: name (str) - nombre de la variable, default (int) - valor por defecto.
    """
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def accept_stable_prediction(
    probabilities: np.ndarray,
    names: list[str],
    history: deque[str],
    config: PredictionGateConfig,
) -> PredictionGateResult:
    """
    Función: Decide si una predicción es suficientemente clara y estable para mostrarse.
    Si la confianza, el margen o la estabilidad no alcanzan el mínimo, devuelve letter=None.
    """
    probs = np.asarray(probabilities, dtype=np.float32)
    if probs.size == 0 or probs.size != len(names):
        history.clear()
        return PredictionGateResult(letter=None, confidence=0.0, margin=0.0)

    sorted_indices = np.argsort(probs)
    best_index = int(sorted_indices[-1])
    second_index = int(sorted_indices[-2]) if probs.size > 1 else best_index
    confidence = float(probs[best_index])
    margin = float(confidence - float(probs[second_index]))

    # Si el modelo duda, se limpia el historial para que no arrastre letras viejas.
    if confidence < config.confidence_threshold or margin < config.margin_threshold:
        history.clear()
        return PredictionGateResult(letter=None, confidence=confidence, margin=margin)

    predicted_letter = names[best_index]
    history.append(predicted_letter)
    stable_frames = max(1, config.stable_frames)
    recent_predictions = list(history)[-stable_frames:]

    if len(recent_predictions) < stable_frames or any(letter != predicted_letter for letter in recent_predictions):
        return PredictionGateResult(letter=None, confidence=confidence, margin=margin)

    return PredictionGateResult(letter=predicted_letter, confidence=confidence, margin=margin)


def run() -> None:
    """
    Función: Inicia y controla el bucle principal de la aplicación.
    Integra la captura de video, la detección de manos y las predicciones del modelo
    para mostrar en pantalla las 24 letras estáticas del ASL.
    """
    load_dotenv(ROOT / ".env")

    # Verifica que el modelo de landmarks haya sido entrenado y guardado previamente
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No existe models/asl_landmarks.joblib. Ejecuta primero: python src/train_landmarks.py")

    # Carga el clasificador de landmarks y la lista de nombres de clases
    import joblib

    model = joblib.load(MODEL_PATH)
    names = class_names()

    # Parámetros de configuración leídos desde variables de entorno
    camera_index = env_int("CAMERA_INDEX", 0)
    gate_config = PredictionGateConfig(
        confidence_threshold=env_float("CONFIDENCE_THRESHOLD", 0.80),
        margin_threshold=env_float("MARGIN_THRESHOLD", 0.15),
        stable_frames=env_int("STABLE_FRAMES", 4),
    )

    # Cola de historial de predicciones para estabilizar la letra mostrada
    prediction_history: deque[str] = deque(maxlen=max(gate_config.stable_frames * 2, 6))
    hand_tracker = create_hand_tracker()

    # Inicialización de la cámara
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara con CAMERA_INDEX={camera_index}")

    try:
        while True:
            # Lee el cuadro actual de la cámara
            ok, frame = cap.read()
            if not ok:
                break

            # Invierte el fotograma horizontalmente (efecto espejo)
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]

            letter = None
            confidence = 0.0
            margin = 0.0
            status = "Sin mano"

            # Detecta si hay una mano visible en el fotograma
            detection = hand_tracker.detect(frame)

            if detection is not None:
                # Dibuja la caja delimitadora o esqueleto de la mano
                hand_tracker.draw(frame, detection)

                if detection.landmark_features is None:
                    prediction_history.clear()
                    status = f"{detection.status} - sin landmarks"
                else:
                    # Predice con el vector geométrico de la mano y aplica rechazo conservador
                    probs = model.predict_proba([detection.landmark_features])[0]
                    gate_result = accept_stable_prediction(probs, names, prediction_history, gate_config)
                    letter = gate_result.letter
                    confidence = gate_result.confidence
                    margin = gate_result.margin
                    status = detection.status if letter else f"{detection.status} - esperando seña clara"
            else:
                # Si no hay mano, limpia el historial de predicciones
                prediction_history.clear()

            # Prepara los textos informativos para mostrar en pantalla
            label_text = f"Letra: {letter or '-'}"
            conf_text = f"Confianza: {confidence:.2f}"
            margin_text = f"Margen: {margin:.2f}"

            # Dibuja la franja superior oscura de información
            cv2.rectangle(frame, (0, 0), (width, 92), (20, 20, 20), -1)
            cv2.putText(frame, label_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, conf_text, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
            cv2.putText(frame, margin_text, (230, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
            cv2.putText(frame, status, (width - 260, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (92, 225, 230), 2)

            # Muestra el fotograma procesado
            cv2.imshow("Detector ASL estatico", frame)

            # Escucha 'q' o Escape para cerrar la aplicación
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        # Cierre correcto de recursos al terminar
        hand_tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
