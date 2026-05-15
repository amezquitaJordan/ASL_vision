"""
ARCHIVO: app.py
MÓDULO: Aplicación Principal
DESCRIPCIÓN: Detector en tiempo real del alfabeto ASL estático (24 letras) usando la cámara del equipo.
PARTE DE LA APP QUE CONTROLA: Inicialización de la interfaz e integración del modelo CNN con la cámara.
"""

from __future__ import annotations

import os
import sys
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from tensorflow import keras

# Asegura que el paquete raíz esté en la ruta del sistema
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.hand_tracking import create_hand_tracker
from src.labels import class_names
from src.preprocessing import preprocess_camera_crop


# Rutas base y ruta del modelo entrenado
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "asl_cnn.keras"


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


def dominant_prediction(history: deque[str]) -> str | None:
    """
    Función: Obtiene la predicción más frecuente dentro del historial de fotogramas recientes.
    Ayuda a evitar fluctuaciones rápidas en la letra detectada.
    Parámetros: history (deque[str]) - cola con las últimas predicciones.
    """
    if not history:
        return None
    return Counter(history).most_common(1)[0][0]


def run() -> None:
    """
    Función: Inicia y controla el bucle principal de la aplicación.
    Integra la captura de video, la detección de manos y las predicciones del modelo
    para mostrar en pantalla las 24 letras estáticas del ASL.
    """
    load_dotenv(ROOT / ".env")

    # Verifica que el modelo haya sido entrenado y guardado previamente
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No existe models/asl_cnn.keras. Ejecuta primero: python src/train.py")

    # Carga el modelo preentrenado y la lista de nombres de clases
    model = keras.models.load_model(MODEL_PATH)
    names = class_names()

    # Parámetros de configuración leídos desde variables de entorno
    threshold = env_float("CONFIDENCE_THRESHOLD", 0.75)
    camera_index = env_int("CAMERA_INDEX", 0)

    # Cola de historial de predicciones para estabilizar la letra mostrada
    prediction_history: deque[str] = deque(maxlen=6)
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
            status = "Sin mano"

            # Detecta si hay una mano visible en el fotograma
            detection = hand_tracker.detect(frame)

            if detection is not None:
                # Dibuja la caja delimitadora o esqueleto de la mano
                hand_tracker.draw(frame, detection)

                # Preprocesa el recorte de la mano para la CNN
                try:
                    prepared = preprocess_camera_crop(detection.crop)

                    # Realiza la predicción con el modelo
                    probs = model.predict(prepared, verbose=0)[0]
                    model_index = int(np.argmax(probs))
                    confidence = float(probs[model_index])

                    # Solo acepta la predicción si supera el umbral de confianza
                    if confidence >= threshold:
                        prediction_history.append(names[model_index])
                        letter = dominant_prediction(prediction_history)
                        status = detection.status
                    else:
                        prediction_history.clear()
                        status = f"{detection.status} - confianza baja"

                except ValueError:
                    status = "Recorte inválido"
            else:
                # Si no hay mano, limpia el historial de predicciones
                prediction_history.clear()

            # Prepara los textos informativos para mostrar en pantalla
            label_text = f"Letra: {letter or '-'}"
            conf_text = f"Confianza: {confidence:.2f}"

            # Dibuja la franja superior oscura de información
            cv2.rectangle(frame, (0, 0), (width, 92), (20, 20, 20), -1)
            cv2.putText(frame, label_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, conf_text, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
            cv2.putText(frame, status, (width - 260, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (92, 225, 230), 2)

            # Muestra el fotograma procesado
            cv2.imshow("Detector ASL A-Z (estatico)", frame)

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
