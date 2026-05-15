"""Train and evaluate the ASL CNN model."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.labels import class_names, label_to_model_index, model_index_to_label
from src.preprocessing import IMAGE_SIZE, normalize_pixels


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "dataset" / "sign_mnist_train.csv"
TEST_CSV = ROOT / "dataset" / "sign_mnist_test.csv"
MODEL_PATH = ROOT / "models" / "asl_cnn.keras"
CLASS_MAP_PATH = ROOT / "models" / "class_map.json"
REPORTS_DIR = ROOT / "reports"


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = pd.read_csv(csv_path)
    labels = data["label"].astype(int).to_numpy()
    pixels = data.drop(columns=["label"]).to_numpy(dtype=np.float32)
    compact_map = label_to_model_index()
    keep_mask = np.array([label in compact_map for label in labels])
    labels = labels[keep_mask]
    pixels = pixels[keep_mask]
    images = normalize_pixels(pixels).reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 1))
    y = np.array([compact_map[int(label)] for label in labels], dtype=np.int64)
    return images, y


def build_model(num_classes: int) -> keras.Model:
    model = keras.Sequential(
        [
            layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dropout(0.35),
            layers.Dense(128, activation="relu"),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_reports(y_true: np.ndarray, y_pred: np.ndarray, history: keras.callbacks.History) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    names = class_names()
    report = classification_report(y_true, y_pred, target_names=names, zero_division=0)
    (REPORTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    metrics = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classes": names,
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    import matplotlib.pyplot as plt

    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Matriz de confusion - ASL Sign MNIST")
    ax.set_xlabel("Prediccion")
    ax.set_ylabel("Etiqueta real")
    ax.set_xticks(range(len(names)), names, rotation=90)
    ax.set_yticks(range(len(names)), names)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def train(epochs: int, batch_size: int, sample_limit: int | None = None) -> keras.Model:
    x, y = load_dataset(TRAIN_CSV)
    if sample_limit:
        x = x[:sample_limit]
        y = y[:sample_limit]
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.15,
        random_state=42,
        stratify=y,
    )

    model = build_model(num_classes=len(class_names()))
    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy"),
    ]
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    x_test, y_test = load_dataset(TEST_CSV)
    y_pred = np.argmax(model.predict(x_test), axis=1)
    save_reports(y_test, y_pred, history)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    CLASS_MAP_PATH.write_text(
        json.dumps(
            {
                "model_index_to_sign_mnist_label": model_index_to_label(),
                "class_names": class_names(),
                "motion_letters": {"J": 9, "Z": 25},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ASL Sign MNIST CNN.")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sample-limit", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(epochs=args.epochs, batch_size=args.batch_size, sample_limit=args.sample_limit)
