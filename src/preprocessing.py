"""
ARCHIVO: preprocessing.py
MÓDULO: Preprocesamiento
DESCRIPCIÓN: Funciones compartidas de preprocesamiento de imágenes para el entrenamiento y la cámara.
PARTE DE LA APP QUE CONTROLA: Adecuación de los píxeles e imágenes antes de pasar por el modelo CNN.
"""

from __future__ import annotations

import numpy as np

# Tamaño de imagen requerido por el modelo CNN
IMAGE_SIZE = 28


def normalize_pixels(image: np.ndarray) -> np.ndarray:
    """
    Función: Convierte los valores de los píxeles a tipo float32 en el rango de 0 a 1.
    Parámetros: image (np.ndarray) - Matriz de píxeles de la imagen.
    """
    array = np.asarray(image, dtype=np.float32)
    
    # Verifica si la imagen está vacía
    if array.size == 0:
        raise ValueError("No se puede normalizar una imagen vacía")
        
    # Si los píxeles superan el valor de 1.0, se dividen entre 255 para normalizarlos
    if float(array.max()) > 1.0:
        array = array / 255.0
        
    # Asegura que todos los valores estén estrictamente entre 0.0 y 1.0
    return np.clip(array, 0.0, 1.0).astype(np.float32)


def reshape_flat_image(flat_pixels: np.ndarray) -> np.ndarray:
    """
    Función: Convierte una fila CSV de 784 píxeles en una imagen 28x28x1 lista para la CNN.
    Parámetros: flat_pixels (np.ndarray) - Arreglo plano de píxeles.
    """
    array = np.asarray(flat_pixels, dtype=np.float32)
    
    # Valida que el número de píxeles coincida con el tamaño esperado
    if array.size != IMAGE_SIZE * IMAGE_SIZE:
        raise ValueError(f"Se esperaban 784 píxeles, se obtuvieron {array.size}")
        
    # Redimensiona a 28x28 con 1 canal (escala de grises)
    image = array.reshape((IMAGE_SIZE, IMAGE_SIZE, 1))
    return normalize_pixels(image)


def preprocess_camera_crop(crop: np.ndarray) -> np.ndarray:
    """
    Función: Prepara un recorte de mano obtenido por OpenCV para ingresarlo a la CNN.
    El modelo está entrenado con imágenes de 28x28 en escala de grises, por lo que 
    los recortes de la cámara en vivo se convierten a grises, se redimensionan, 
    se normalizan y se ajustan al formato por lotes.
    Parámetros: crop (np.ndarray) - Imagen recortada de la mano.
    """
    import cv2

    # Verifica si el recorte es válido
    if crop is None or crop.size == 0:
        raise ValueError("No se puede preprocesar un recorte de cámara vacío")

    # Convierte a escala de grises si la imagen tiene 3 canales (RGB/BGR)
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    # Redimensiona la imagen al tamaño requerido (28x28)
    resized = cv2.resize(gray, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    
    # Ecualiza el histograma para mejorar el contraste
    equalized = cv2.equalizeHist(resized.astype(np.uint8))
    
    # Normaliza la imagen y le da la forma de un lote de 1 elemento (1, 28, 28, 1)
    image = normalize_pixels(equalized).reshape((1, IMAGE_SIZE, IMAGE_SIZE, 1))
    return image
