"""Label helpers for the Sign Language MNIST dataset."""

from __future__ import annotations

from string import ascii_uppercase


MOTION_LABELS = {9: "J", 25: "Z"}
LABEL_TO_LETTER = {
    index: letter
    for index, letter in enumerate(ascii_uppercase)
    if index not in MOTION_LABELS
}
LETTER_TO_LABEL = {letter: label for label, letter in LABEL_TO_LETTER.items()}


def letter_from_label(label: int) -> str:
    """Return the ASL letter for a Sign MNIST numeric label."""
    try:
        return LABEL_TO_LETTER[int(label)]
    except KeyError as exc:
        raise ValueError(f"Unsupported static Sign MNIST label: {label}") from exc


def static_class_indices() -> list[int]:
    """Return Sign MNIST labels that can be learned from static images."""
    return sorted(LABEL_TO_LETTER)


def class_names() -> list[str]:
    """Return static class names ordered by model output index."""
    return [LABEL_TO_LETTER[label] for label in static_class_indices()]


def model_index_to_label() -> dict[int, int]:
    """Map model output indexes to original Sign MNIST labels."""
    return {model_index: label for model_index, label in enumerate(static_class_indices())}


def label_to_model_index() -> dict[int, int]:
    """Map original Sign MNIST labels to compact model output indexes."""
    return {label: model_index for model_index, label in model_index_to_label().items()}
