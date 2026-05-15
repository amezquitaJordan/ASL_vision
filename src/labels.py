"""
ARCHIVO: labels.py
MÓDULO: Etiquetas
DESCRIPCIÓN: Funciones de ayuda para mapear etiquetas numéricas a letras del dataset Sign MNIST.
PARTE DE LA APP QUE CONTROLA: Traducción de las salidas del modelo a letras del alfabeto estático.
"""

from __future__ import annotations

from string import ascii_uppercase

# Etiquetas excluidas del modelo porque Sign MNIST no tiene imágenes estáticas para ellas.
# J (9) y Z (25) requieren movimiento y quedan fuera del alcance de este detector.
_ETIQUETAS_EXCLUIDAS = {9, 25}

# Mapea un índice numérico del dataset a su letra estática (A-Y, sin J ni Z)
LABEL_TO_LETTER = {
    index: letter
    for index, letter in enumerate(ascii_uppercase)
    if index not in _ETIQUETAS_EXCLUIDAS
}

# Mapea una letra estática a su etiqueta numérica original del dataset
LETTER_TO_LABEL = {letter: label for label, letter in LABEL_TO_LETTER.items()}


def letter_from_label(label: int) -> str:
    """
    Función: Devuelve la letra estática correspondiente a un índice numérico del dataset.
    Parámetros: label (int) - El índice numérico de la clase.
    """
    try:
        return LABEL_TO_LETTER[int(label)]
    except KeyError as exc:
        raise ValueError(f"Etiqueta estática de Sign MNIST no soportada: {label}") from exc


def static_class_indices() -> list[int]:
    """
    Función: Devuelve una lista ordenada con las etiquetas numéricas de las 24 señas estáticas.
    """
    return sorted(LABEL_TO_LETTER)


def class_names() -> list[str]:
    """
    Función: Devuelve los nombres de las 24 clases estáticas ordenados según el índice del modelo.
    """
    return [LABEL_TO_LETTER[label] for label in static_class_indices()]


def model_index_to_label() -> dict[int, int]:
    """
    Función: Mapea los índices compactos del modelo a las etiquetas originales del dataset.
    """
    return {model_index: label for model_index, label in enumerate(static_class_indices())}


def label_to_model_index() -> dict[int, int]:
    """
    Función: Mapea las etiquetas originales del dataset a los índices compactos del modelo.
    """
    return {label: model_index for model_index, label in model_index_to_label().items()}
