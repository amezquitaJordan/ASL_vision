"""
ARCHIVO: train.py
MÓDULO: Entrenamiento del Modelo
DESCRIPCIÓN: Entrena y evalúa la red neuronal CNN para el reconocimiento del alfabeto ASL.
PARTE DE LA APP QUE CONTROLA: Construcción, entrenamiento y guardado del modelo principal (asl_cnn.keras).
"""

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

# Permite ejecutar el archivo directamente sin instalar el paquete
if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.labels import class_names, label_to_model_index, model_index_to_label
from src.preprocessing import IMAGE_SIZE, normalize_pixels
from src.real_images import load_real_image_dataset


# Rutas de los archivos y directorios del proyecto
ROOT = Path(__file__).resolve().parents[1]
TRAIN_CSV = ROOT / "dataset" / "sign_mnist_train.csv"
TEST_CSV = ROOT / "dataset" / "sign_mnist_test.csv"
MODEL_PATH = ROOT / "models" / "asl_cnn.keras"
CLASS_MAP_PATH = ROOT / "models" / "class_map.json"
REPORTS_DIR = ROOT / "reports"
REAL_IMAGE_DIR = ROOT / "dataset" / "senas_reales_entrenamiento"


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Función: Carga el dataset CSV del Sign MNIST y lo convierte en imágenes 28x28 listas para la CNN.
    Filtra las etiquetas no válidas (letras J y Z que requieren movimiento).
    Parámetros: csv_path (Path) - ruta al archivo CSV de entrenamiento o prueba.
    """
    data = pd.read_csv(csv_path)
    
    # Separa las etiquetas y los píxeles
    labels = data["label"].astype(int).to_numpy()
    pixels = data.drop(columns=["label"]).to_numpy(dtype=np.float32)
    
    # Obtiene el mapeo de etiquetas originales a índices compactos del modelo
    compact_map = label_to_model_index()
    
    # Filtra filas cuya etiqueta no es reconocida por el modelo (ej: J=9, Z=25)
    keep_mask = np.array([label in compact_map for label in labels])
    labels = labels[keep_mask]
    pixels = pixels[keep_mask]
    
    # Normaliza y da forma a las imágenes (N, 28, 28, 1)
    images = normalize_pixels(pixels).reshape((-1, IMAGE_SIZE, IMAGE_SIZE, 1))
    
    # Convierte las etiquetas originales a los índices compactos del modelo
    y = np.array([compact_map[int(label)] for label in labels], dtype=np.int64)
    return images, y


def build_model(num_classes: int) -> keras.Model:
    """
    Función: Construye y compila la arquitectura CNN para clasificar señas del ASL.
    La red tiene 3 bloques convolucionales seguidos de capas densas.
    Parámetros: num_classes (int) - número de clases a predecir (letras estáticas del alfabeto).
    """
    model = keras.Sequential(
        [
            # Capa de entrada: imágenes de 28x28 en escala de grises
            layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 1)),

            # Bloque 1: 32 filtros, normalización y reducción espacial
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Bloque 2: 64 filtros, normalización y reducción espacial
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.BatchNormalization(),
            layers.MaxPooling2D((2, 2)),

            # Bloque 3: 128 filtros, aplanado y regularización
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dropout(0.35),

            # Capa densa intermedia y capa de salida con softmax
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


def merge_training_data(
    sign_x: np.ndarray,
    sign_y: np.ndarray,
    real_x: np.ndarray,
    real_y: np.ndarray,
    real_weight: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Función: Combina el dataset base con las imágenes reales repetidas N veces.
    Las imágenes reales se duplican para que el modelo les dé más importancia.
    Parámetros:
      - sign_x/y: imágenes y etiquetas del dataset Sign MNIST.
      - real_x/y: imágenes y etiquetas del dataset real.
      - real_weight (int): número de veces que se repiten las imágenes reales.
    """
    # Si no hay imágenes reales o el peso es 0, devuelve el dataset original sin cambios
    if real_x.size == 0 or real_y.size == 0 or real_weight <= 0:
        return sign_x, sign_y

    # Repite las imágenes reales la cantidad de veces indicada
    repeated_x = np.repeat(real_x, repeats=real_weight, axis=0)
    repeated_y = np.repeat(real_y, repeats=real_weight, axis=0)
    
    # Concatena el dataset base con las imágenes reales repetidas
    return np.concatenate([sign_x, repeated_x], axis=0), np.concatenate([sign_y, repeated_y], axis=0)


def save_reports(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    history: keras.callbacks.History,
    real_training_images: int = 0,
    real_training_weight: int = 0,
) -> None:
    """
    Función: Guarda los reportes del entrenamiento: métricas, clasificación y matriz de confusión.
    Parámetros:
      - y_true: etiquetas reales del conjunto de prueba.
      - y_pred: predicciones del modelo sobre el conjunto de prueba.
      - history: historial de métricas del entrenamiento.
      - real_training_images: cantidad de imágenes reales usadas.
      - real_training_weight: peso aplicado a las imágenes reales.
    """
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    names = class_names()

    # Genera y guarda el reporte de clasificación (precisión, recall, F1 por clase)
    report = classification_report(y_true, y_pred, target_names=names, zero_division=0)
    (REPORTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    # Construye el diccionario de métricas generales
    metrics = {
        "accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "classes": names,
        "real_training_images": real_training_images,
        "real_training_weight": real_training_weight,
        # Convierte el historial de Keras a tipos nativos de Python
        "history": {key: [float(value) for value in values] for key, values in history.history.items()},
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    import matplotlib.pyplot as plt

    # Genera y guarda la imagen de la matriz de confusión
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(12, 10))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Matriz de confusión - ASL Sign MNIST")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")
    ax.set_xticks(range(len(names)), names, rotation=90)
    ax.set_yticks(range(len(names)), names)
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=160)
    plt.close(fig)


def train(
    epochs: int,
    batch_size: int,
    sample_limit: int | None = None,
    real_data_dir: Path | None = REAL_IMAGE_DIR,
    real_weight: int = 4,
) -> keras.Model:
    """
    Función: Orquesta el proceso completo de entrenamiento del modelo.
    Carga datos, divide en train/val, entrena la CNN y guarda el modelo final.
    Parámetros:
      - epochs (int): número máximo de épocas.
      - batch_size (int): tamaño del lote por iteración.
      - sample_limit (int | None): limita la cantidad de muestras del CSV (para pruebas rápidas).
      - real_data_dir (Path | None): carpeta con imágenes reales. None para omitirlas.
      - real_weight (int): cuántas veces se repiten las imágenes reales en entrenamiento.
    """
    # Carga el dataset principal del Sign MNIST
    x, y = load_dataset(TRAIN_CSV)
    
    # Recorta el dataset si se especificó un límite de muestras
    if sample_limit:
        x = x[:sample_limit]
        y = y[:sample_limit]

    # Separa el dataset base en entrenamiento (85%) y validación (15%)
    x_train, x_val, y_train, y_val = train_test_split(
        x,
        y,
        test_size=0.15,
        random_state=42,
        stratify=y,
    )

    real_count = 0
    if real_data_dir is not None and real_data_dir.exists():
        # Carga las imágenes reales desde la carpeta
        real_x, real_y = load_real_image_dataset(real_data_dir)
        real_count = int(real_y.size)

        # Divide las imágenes reales en train/val por separado para evitar fuga de datos
        rx_train, rx_val, ry_train, ry_val = train_test_split(
            real_x,
            real_y,
            test_size=0.15,
            random_state=42,
            stratify=real_y,
        )

        # Agrega las imágenes reales de entrenamiento con peso multiplicado
        x_train, y_train = merge_training_data(x_train, y_train, rx_train, ry_train, real_weight=real_weight)

        # Agrega las imágenes reales de validación sin duplicar (métricas honestas)
        x_val = np.concatenate([x_val, rx_val], axis=0)
        y_val = np.concatenate([y_val, ry_val], axis=0)

        print(f"Imágenes reales cargadas: {real_count} (peso x{real_weight} solo en train)")
    elif real_data_dir is not None:
        print(f"No se encontró dataset real en: {real_data_dir}")

    # Construye el modelo CNN con tantas clases como letras estáticas hay
    model = build_model(num_classes=len(class_names()))
    
    # EarlyStopping: detiene si val_accuracy no mejora en 3 épocas consecutivas
    callbacks = [
        keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True, monitor="val_accuracy"),
    ]
    
    # Inicia el entrenamiento
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
    )

    # Evalúa el modelo con el conjunto de prueba independiente
    x_test, y_test = load_dataset(TEST_CSV)
    y_pred = np.argmax(model.predict(x_test), axis=1)
    
    # Guarda reportes y métricas en la carpeta reports/
    save_reports(
        y_test,
        y_pred,
        history,
        real_training_images=real_count,
        real_training_weight=real_weight if real_count else 0,
    )

    # Crea la carpeta models/ si no existe y guarda el modelo entrenado
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_PATH)
    
    # Guarda el mapa de clases en JSON para que app.py lo use en tiempo real
    CLASS_MAP_PATH.write_text(
        json.dumps(
            {
                "model_index_to_sign_mnist_label": model_index_to_label(),
                # Las 24 letras estáticas que reconoce el modelo (sin J ni Z)
                "class_names": class_names(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return model


def parse_args() -> argparse.Namespace:
    """
    Función: Parsea los argumentos de la línea de comandos para personalizar el entrenamiento.
    """
    parser = argparse.ArgumentParser(description="Entrena la CNN ASL del Sign MNIST.")
    parser.add_argument("--epochs", type=int, default=12, help="Número máximo de épocas")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamaño del lote")
    parser.add_argument("--sample-limit", type=int, default=None, help="Límite de muestras (pruebas rápidas)")
    parser.add_argument("--real-data-dir", type=Path, default=REAL_IMAGE_DIR, help="Carpeta de imágenes reales")
    parser.add_argument("--real-weight", type=int, default=4, help="Peso de repetición de imágenes reales")
    parser.add_argument("--no-real-data", action="store_true", help="Omite las imágenes reales")
    return parser.parse_args()


if __name__ == "__main__":
    # Punto de entrada al ejecutar: python src/train.py [argumentos]
    args = parse_args()
    train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        sample_limit=args.sample_limit,
        real_data_dir=None if args.no_real_data else args.real_data_dir,
        real_weight=args.real_weight,
    )
