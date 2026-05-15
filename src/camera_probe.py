"""
ARCHIVO: camera_probe.py
MÓDULO: Pruebas de Cámara
DESCRIPCIÓN: Prueba los índices de cámaras disponibles utilizando OpenCV en Windows.
PARTE DE LA APP QUE CONTROLA: Script utilitario para detectar la cámara correcta a usar.
"""

from __future__ import annotations

import argparse
import cv2


def probe(max_index: int, seconds: float) -> None:
    """
    Función: Intenta abrir y leer imágenes de los primeros N índices de cámara.
    Parámetros:
      - max_index (int): El índice máximo de cámara a probar.
      - seconds (float): Tiempo en segundos para mostrar cada cámara exitosa.
    """
    print("Probando cámaras con OpenCV...")
    print("Cierra cada ventana con cualquier tecla para pasar a la siguiente.")
    
    # Itera sobre cada posible índice de cámara
    for index in range(max_index + 1):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        
        # Verifica si el índice de cámara se pudo abrir
        if not cap.isOpened():
            print(f"[{index}] no abre")
            cap.release()
            continue

        # Intenta leer un fotograma de la cámara
        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"[{index}] abre, pero no entrega imagen")
            cap.release()
            continue

        # Obtiene las dimensiones y el brillo de la imagen
        height, width = frame.shape[:2]
        mean_brightness = frame.mean()
        print(f"[{index}] OK: {width}x{height}, brillo promedio {mean_brightness:.1f}")
        
        # Dibuja el texto informativo sobre la imagen
        cv2.putText(
            frame,
            f"CAMERA_INDEX={index}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2,
        )
        
        # Muestra la ventana con la imagen de prueba
        window_name = f"Prueba cámara {index}"
        cv2.imshow(window_name, frame)
        cv2.waitKey(int(seconds * 1000))
        cv2.destroyWindow(window_name)
        
        # Libera la cámara
        cap.release()

    # Cierra todas las ventanas de OpenCV creadas
    cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    """
    Función: Parsea los argumentos de línea de comandos.
    """
    parser = argparse.ArgumentParser(description="Prueba los índices de cámara de OpenCV.")
    parser.add_argument("--max-index", type=int, default=4, help="Índice máximo a probar")
    parser.add_argument("--seconds", type=float, default=2.0, help="Segundos por prueba")
    return parser.parse_args()


if __name__ == "__main__":
    # Lee los argumentos y ejecuta la prueba
    args = parse_args()
    probe(max_index=args.max_index, seconds=args.seconds)
