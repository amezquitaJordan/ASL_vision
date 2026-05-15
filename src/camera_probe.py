"""Probe available OpenCV camera indexes on Windows."""

from __future__ import annotations

import argparse

import cv2


def probe(max_index: int, seconds: float) -> None:
    print("Probando camaras con OpenCV...")
    print("Cierra cada ventana con cualquier tecla para pasar a la siguiente.")
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            print(f"[{index}] no abre")
            cap.release()
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[{index}] abre, pero no entrega imagen")
            cap.release()
            continue

        height, width = frame.shape[:2]
        mean_brightness = frame.mean()
        print(f"[{index}] OK: {width}x{height}, brillo promedio {mean_brightness:.1f}")
        cv2.putText(
            frame,
            f"CAMERA_INDEX={index}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
        window_name = f"Prueba camara {index}"
        cv2.imshow(window_name, frame)
        cv2.waitKey(int(seconds * 1000))
        cv2.destroyWindow(window_name)
        cap.release()

    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe OpenCV camera indexes.")
    parser.add_argument("--max-index", type=int, default=4)
    parser.add_argument("--seconds", type=float, default=2.0)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    probe(max_index=args.max_index, seconds=args.seconds)
