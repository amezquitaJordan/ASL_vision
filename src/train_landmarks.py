"""
ARCHIVO: train_landmarks.py
MÓDULO: Entrenamiento con Landmarks
DESCRIPCIÓN: Entrena el clasificador principal de letras estáticas usando puntos de MediaPipe.
PARTE DE LA APP QUE CONTROLA: Generación del modelo models/asl_landmarks.joblib usado por la cámara.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

# Permite ejecutar el archivo directamente sin instalar el paquete
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.labels import class_names
from src.landmarks import LANDMARK_VECTOR_SIZE, MediaPipeLandmarkExtractor
from src.real_images import iter_labeled_image_paths


# Rutas principales del proyecto
ROOT = Path(__file__).resolve().parents[1]
REAL_IMAGE_DIR = ROOT / "dataset" / "senas_reales_entrenamiento"
MODEL_PATH = ROOT / "models" / "asl_landmarks.joblib"
CLASS_MAP_PATH = ROOT / "models" / "landmark_class_map.json"
REPORTS_DIR = ROOT / "reports"


def find_conflicting_hashes(paths: list[tuple[Path, int]]) -> tuple[dict[Path, str], set[str]]:
    """
    Función: Detecta imágenes idénticas ubicadas en carpetas de letras distintas.
    Esas muestras son ambiguas y se descartan para no entrenar etiquetas contradictorias.
    """
    path_hashes: dict[Path, str] = {}
    labels_by_hash: dict[str, set[int]] = defaultdict(set)

    for path, label in paths:
        try:
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
        except OSError:
            continue

        path_hashes[path] = digest
        labels_by_hash[digest].add(label)

    conflicts = {digest for digest, labels in labels_by_hash.items() if len(labels) > 1}
    return path_hashes, conflicts


def load_landmark_dataset(
    root: Path,
    max_per_class: int | None = None,
    min_detection_confidence: float = 0.45,
) -> tuple[np.ndarray, np.ndarray, dict[str, dict[str, int]], list[str]]:
    """
    Función: Lee imágenes reales, extrae landmarks y arma matrices X/y para entrenar.
    También reporta cuántas imágenes por letra fueron útiles o descartadas.
    """
    names = class_names()
    paths = iter_labeled_image_paths(root)
    path_hashes, conflicting_hashes = find_conflicting_hashes(paths)
    stats = {letter: {"total": 0, "usable": 0, "discarded": 0} for letter in names}
    usable_by_label: dict[int, int] = defaultdict(int)
    conflicting_files = 0

    extractor = MediaPipeLandmarkExtractor(static_image_mode=True, min_detection_confidence=min_detection_confidence)
    features: list[np.ndarray] = []
    labels: list[int] = []

    try:
        for path, label in paths:
            if max_per_class is not None and usable_by_label[label] >= max_per_class:
                continue

            letter = names[label]
            stats[letter]["total"] += 1

            if path_hashes.get(path) in conflicting_hashes:
                stats[letter]["discarded"] += 1
                conflicting_files += 1
                continue

            frame = cv2.imread(str(path))

            if frame is None:
                stats[letter]["discarded"] += 1
                continue

            extraction = extractor.extract(frame)
            if extraction is None:
                stats[letter]["discarded"] += 1
                continue

            features.append(extraction.features)
            labels.append(label)
            usable_by_label[label] += 1
            stats[letter]["usable"] += 1
    finally:
        extractor.close()

    warnings = [
        f"{letter}: menos de 20 muestras utiles ({values['usable']})"
        for letter, values in stats.items()
        if values["total"] > 0 and values["usable"] < 20
    ]
    if conflicting_files:
        warnings.append(f"duplicados con etiquetas distintas descartados: {conflicting_files}")

    if not features:
        return (
            np.empty((0, LANDMARK_VECTOR_SIZE), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
            stats,
            warnings,
        )

    return np.stack(features).astype(np.float32), np.asarray(labels, dtype=np.int64), stats, warnings


def build_classifier() -> RandomForestClassifier:
    """
    Función: Crea un clasificador robusto para vectores de landmarks.
    RandomForest entrega probabilidades y tolera bien datasets pequeños.
    """
    return RandomForestClassifier(
        n_estimators=500,
        random_state=42,
        class_weight="balanced",
        min_samples_leaf=1,
        n_jobs=-1,
    )


def safe_train_test_split(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Función: Divide train/test manteniendo clases balanceadas cuando hay suficientes muestras.
    El tamaño de test se ajusta para que cada clase pueda aparecer al menos una vez.
    """
    unique_classes = np.unique(y)
    counts = np.bincount(y, minlength=len(class_names()))
    if unique_classes.size < 2 or any(counts[label] < 2 for label in unique_classes):
        raise ValueError("Cada clase entrenada necesita al menos 2 imágenes útiles para evaluar con stratify")

    test_count = max(unique_classes.size, math.ceil(y.size * 0.20))
    test_size = min(0.40, test_count / y.size)
    return train_test_split(x, y, test_size=test_size, random_state=42, stratify=y)


def save_reports(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    stats: dict[str, dict[str, int]],
    warnings: list[str],
    max_per_class: int | None,
) -> None:
    """
    Función: Guarda métricas, reporte de clasificación y matriz de confusión del modelo de landmarks.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    names = class_names()
    labels = list(range(len(names)))

    report = classification_report(y_true, y_pred, labels=labels, target_names=names, zero_division=0)
    (REPORTS_DIR / "landmark_classification_report.txt").write_text(report, encoding="utf-8")

    metrics = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "classes": names,
        "feature_count": LANDMARK_VECTOR_SIZE,
        "max_per_class": max_per_class,
        "dataset_stats": stats,
        "warnings": warnings,
    }
    (REPORTS_DIR / "landmark_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = confusion_matrix(y_true, y_pred, labels=labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Matriz de confusión - ASL landmarks estáticos")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_xticks(range(len(names)), names, rotation=90)
    ax.set_yticks(range(len(names)), names)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "landmark_confusion_matrix.png", dpi=160)
    plt.close(fig)


def train(
    real_data_dir: Path = REAL_IMAGE_DIR,
    max_per_class: int | None = None,
    min_detection_confidence: float = 0.45,
) -> RandomForestClassifier:
    """
    Función: Ejecuta el entrenamiento completo del clasificador de landmarks y guarda sus artefactos.
    """
    x, y, stats, warnings = load_landmark_dataset(
        real_data_dir,
        max_per_class=max_per_class,
        min_detection_confidence=min_detection_confidence,
    )
    if x.size == 0 or y.size == 0:
        raise ValueError("No se encontraron imágenes útiles con landmarks para entrenar")

    x_train, x_test, y_train, y_test = safe_train_test_split(x, y)
    model = build_classifier()
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    save_reports(y_test, y_pred, stats=stats, warnings=warnings, max_per_class=max_per_class)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    CLASS_MAP_PATH.write_text(
        json.dumps(
            {
                "class_names": class_names(),
                "feature_count": LANDMARK_VECTOR_SIZE,
                "model_type": "RandomForestClassifier",
                "static_only": True,
                "excluded_letters": ["J", "Z"],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    usable_total = int(y.size)
    print(f"Modelo de landmarks guardado en {MODEL_PATH}")
    print(f"Imágenes útiles para entrenamiento: {usable_total}")
    for warning in warnings:
        print(f"ADVERTENCIA: {warning}")
    return model


def parse_args() -> argparse.Namespace:
    """Función: Parsea argumentos para personalizar el entrenamiento de landmarks."""
    parser = argparse.ArgumentParser(description="Entrena el clasificador ASL estático con landmarks de MediaPipe.")
    parser.add_argument("--real-data-dir", type=Path, default=REAL_IMAGE_DIR, help="Carpeta con subcarpetas por letra")
    parser.add_argument("--max-per-class", type=int, default=None, help="Máximo de imágenes por letra para pruebas rápidas")
    parser.add_argument("--min-detection-confidence", type=float, default=0.45, help="Confianza mínima de MediaPipe")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        real_data_dir=args.real_data_dir,
        max_per_class=args.max_per_class,
        min_detection_confidence=args.min_detection_confidence,
    )
