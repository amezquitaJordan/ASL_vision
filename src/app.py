"""Real-time ASL alphabet detector with camera and OpenAI TTS."""

from __future__ import annotations

import os
import sys
import time
from collections import Counter, deque
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv
from tensorflow import keras

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.hand_tracking import create_hand_tracker
from src.labels import class_names
from src.motion_letters import MotionLetterDetector
from src.preprocessing import preprocess_camera_crop
from src.tts import TTSConfig, ensure_audio, play_audio, should_speak


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "asl_cnn.keras"


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, default))
    except ValueError:
        return default


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, default))
    except ValueError:
        return default


def dominant_prediction(history: deque[str]) -> str | None:
    if not history:
        return None
    return Counter(history).most_common(1)[0][0]


def run() -> None:
    load_dotenv(ROOT / ".env")
    if not MODEL_PATH.exists():
        raise FileNotFoundError("No existe models/asl_cnn.keras. Ejecuta primero: python src/train.py")

    model = keras.models.load_model(MODEL_PATH)
    names = class_names()
    threshold = env_float("CONFIDENCE_THRESHOLD", 0.75)
    camera_index = env_int("CAMERA_INDEX", 0)
    tts_config = TTSConfig(
        voice=os.getenv("TTS_VOICE", "coral"),
        stable_frames=env_int("TTS_STABLE_FRAMES", 8),
        cooldown_seconds=env_float("TTS_COOLDOWN_SECONDS", 1.5),
        audio_dir=ROOT / "audio_cache",
    )
    tts_state = {"letter": None, "count": 0, "last_spoken": {}}
    prediction_history: deque[str] = deque(maxlen=6)
    motion_detector = MotionLetterDetector()
    hand_tracker = create_hand_tracker()

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la camara con CAMERA_INDEX={camera_index}")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            letter = None
            confidence = 0.0
            status = "Sin mano"
            detection = hand_tracker.detect(frame)

            if detection is not None:
                hand_tracker.draw(frame, detection)
                motion_letter = motion_detector.update(detection.motion_point)
                if motion_letter:
                    letter = motion_letter
                    confidence = 1.0
                    prediction_history.clear()
                    status = "Movimiento"
                else:
                    try:
                        prepared = preprocess_camera_crop(detection.crop)
                        probs = model.predict(prepared, verbose=0)[0]
                        model_index = int(np.argmax(probs))
                        confidence = float(probs[model_index])
                        if confidence >= threshold:
                            prediction_history.append(names[model_index])
                            letter = dominant_prediction(prediction_history)
                            status = detection.status
                        else:
                            prediction_history.clear()
                            status = f"{detection.status} - confianza baja"
                    except ValueError:
                        status = "Recorte invalido"
            else:
                motion_detector.reset()
                prediction_history.clear()

            tts_state["now"] = time.time()
            if should_speak(letter, tts_state, tts_config):
                audio_path = ensure_audio(letter, tts_config)
                play_audio(audio_path)

            label_text = f"Letra: {letter or '-'}"
            conf_text = f"Confianza: {confidence:.2f}"
            cv2.rectangle(frame, (0, 0), (width, 92), (20, 20, 20), -1)
            cv2.putText(frame, label_text, (20, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            cv2.putText(frame, conf_text, (20, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2)
            cv2.putText(frame, status, (width - 260, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (92, 225, 230), 2)
            cv2.imshow("Detector ASL A-Z", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break

    finally:
        hand_tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run()
