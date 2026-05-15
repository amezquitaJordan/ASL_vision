"""
ARCHIVO: app.py
MÓDULO: Aplicación Principal
DESCRIPCIÓN: Detector en tiempo real del alfabeto ASL usando la cámara del equipo.
PARTE DE LA APP QUE CONTROLA: Inicialización de la interfaz, integración del modelo CNN y detectores de movimiento.
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
from src.motion_letters import MotionLetterDetector
from src.preprocessing import preprocess_camera_crop


# Rutas base y ruta del modelo
ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "asl_cnn.keras"


def env_float(name: str, default: float) -> float:
    """
    Función: Lee una variable de entorno como tipo flotante.
    """
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    """
    Función: Lee una variable de entorno como tipo entero.
    """
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def dominant_prediction(history: deque[str]) -> str | None:
    """
    Función: Obtiene la predicción más frecuente dentro de los últimos fotogramas.
    Ayuda a evitar fluctuaciones rápidas en la letra detectada.
    """
    if not history:
        return None
    return Counter(history).most_common(1)[0][0]


def run() -> None:
    """
    Función: Inicia y controla el bucle principal de la aplicación, integrando 
    la captura de video, la detección de manos y las predicciones del modelo.
    """
    load_dotenv(ROOT / ".env")
    
    # Verifica que el modelo haya sido entrenado y guardado
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No existe models/asl_cnn.keras. Ejecuta primero: python src/train.py")

    # Carga el modelo preentrenado de Keras y la lista de nombres
    model = keras.models.load_model(MODEL_PATH)
    names = class_names()
    
    # Parámetros por defecto obtenidos de variables de entorno
    threshold = env_float("CONFIDENCE_THRESHOLD", 0.75)
    camera_index = env_int("CAMERA_INDEX", 0)
    
    # Historial de predicciones y detectores
    prediction_history: deque[str] = deque(maxlen=6)
    motion_detector = MotionLetterDetector()
    hand_tracker = create_hand_tracker()

    # Inicialización de cámara
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara con CAMERA_INDEX={camera_index}")

    try:
        while True:
            # Lee el cuadro actual
            ok, frame = cap.read()
            if not ok:
                break

            # Invierte el cuadro horizontalmente (efecto espejo)
            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            
            letter = None
            confidence = 0.0
            status = "Sin mano"
            
            # Detecta la mano dentro del cuadro actual
            detection = hand_tracker.detect(frame)

            if detection is not None:
                # Dibuja la caja o esqueleto sobre la mano detectada
                hand_tracker.draw(frame, detection)
                
                # Envía la posición al detector de movimiento (para J y Z)
                motion_letter = motion_detector.update(detection.motion_point)
                
                # Si se detecta un movimiento dinámico, se asume 100% de confianza
                if motion_letter:
                    letter = motion_letter
                    confidence = 1.0
                    prediction_history.clear()
                    status = "Movimiento"
                else:
                    # De lo contrario, se usa la CNN para predecir sobre la imagen estática
                    try:
                        # Preprocesa la imagen para el modelo
                        prepared = preprocess_camera_crop(detection.crop)
                        
                        # Realiza la predicción con el modelo
                        probs = model.predict(prepared, verbose=0)[0]
                        model_index = int(np.argmax(probs))
                        confidence = float(probs[model_index])
                        
                        # Evalúa si la predicción cumple el umbral mínimo
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
                # Si no se detecta la mano, se reinician los sistemas
                motion_detector.reset()
                prediction_history.clear()

            # Configuración de los textos en pantalla
            label_text = f"Letra: {letter or '-'}"
            conf_text = f"Confianza: {confidence:.2f}"
            
            # Dibuja la franja superior oscura para albergar los textos
            cv2.rectangle(frame, (0, 0), (width, 92), (20, 20, 20), -1)
            cv2.putText(frame, label_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, conf_text, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
            cv2.putText(frame, status, (width - 260, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (92, 225, 230), 2)
            
            # Muestra el cuadro final
            cv2.imshow("Detector ASL A-Z", frame)

            # Escucha interrupciones del teclado ('q' o Escape para cerrar)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        # Limpieza y cierre correcto de los recursos
        hand_tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
