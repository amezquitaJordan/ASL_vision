"""
ARCHIVO: labels.py
MÓDULO: Etiquetas
DESCRIPCIÓN: Funciones de ayuda para mapear etiquetas numéricas a letras del dataset Sign MNIST.
PARTE DE LA APP QUE CONTROLA: Traducción de las salidas del modelo a letras del alfabeto.
"""

from __future__ import annotations

from string import ascii_uppercase

# Etiquetas que requieren movimiento y no se pueden predecir con una sola imagen
MOTION_LABELS = {9: "J", 25: "Z"}

# Mapea un índice numérico a una letra estática (A-Y, excluyendo J y Z)
LABEL_TO_LETTER = {
    index: letter
    for index, letter in enumerate(ascii_uppercase)
    if index not in MOTION_LABELS
}

# Mapea una letra a su etiqueta numérica original
LETTER_TO_LABEL = {letter: label for label, letter in LABEL_TO_LETTER.items()}


def letter_from_label(label: int) -> str:
    """
    Función: Devuelve la letra correspondiente al lenguaje de señas (ASL) dado un índice numérico del dataset.
    Parámetros: label (int) - El índice numérico de la clase.
    """
    try:
        return LABEL_TO_LETTER[int(label)]
    except KeyError as exc:
        raise ValueError(f"Etiqueta estática de Sign MNIST no soportada: {label}") from exc


def static_class_indices() -> list[int]:
    """
    Función: Devuelve una lista con las etiquetas numéricas que corresponden a señas estáticas.
    """
    return sorted(LABEL_TO_LETTER)


def class_names() -> list[str]:
    """
    Función: Devuelve los nombres de las clases estáticas (letras) ordenados según el índice del modelo.
    """
    return [LABEL_TO_LETTER[label] for label in static_class_indices()]


def model_index_to_label() -> dict[int, int]:
    """
    Función: Mapea los índices compactos del modelo a las etiquetas originales del dataset.
    """
    return {model_index: label for model_index, label in enumerate(static_class_indices())}


def label_to_model_index() -> dict[int, int]:
    """
    Función: Mapea las etiquetas originales del dataset a los índices compactos que utiliza el modelo.
    """
    return {label: model_index for model_index, label in model_index_to_label().items()}
