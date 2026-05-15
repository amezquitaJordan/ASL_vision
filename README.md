# Detector de abecedario ASL con CNN y cámara

Proyecto local para Visual Studio Code. Detecta **24 letras estáticas** del alfabeto ASL con la cámara y muestra la letra en pantalla.

> Las letras `J` y `Z` no están incluidas porque el dataset Sign MNIST no tiene imágenes estáticas para ellas (requieren movimiento). El modelo reconoce A–Y excluyendo J.

## Enfoque

- CNN entrenada con `dataset/sign_mnist_train.csv` y evaluada con `dataset/sign_mnist_test.csv`.
- Métricas guardadas en `reports/`: accuracy, macro F1, matriz de confusión y reporte por clase.
- Cámara en tiempo real con OpenCV y MediaPipe Hands.
- Solo letras estáticas: **24 clases** (A–Y sin J).

## Configuración en Windows

Abre esta carpeta en Visual Studio Code y ejecuta en la terminal:

```powershell
C:\Users\jorda\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Entrenar el modelo

```powershell
.\.venv\Scripts\python.exe src\train.py
```

Para una prueba rápida de entrenamiento:

```powershell
.\.venv\Scripts\python.exe src\train.py --epochs 1 --sample-limit 1200
```

Si existe `dataset/senas_reales_entrenamiento/`, el entrenamiento la usa
automáticamente. Esa carpeta debe tener subcarpetas por letra, por ejemplo
`A/`, `B/`, `C/`, hasta `Y/` sin `J` ni `Z`. Las imágenes reales se recortan
con MediaPipe antes de entrar al modelo y se repiten con peso `x4` para que
influyan más que el dataset base.

Para ajustar ese peso:

```powershell
.\.venv\Scripts\python.exe src\train.py --real-weight 6
```

Para entrenar solo con Sign MNIST y comparar:

```powershell
.\.venv\Scripts\python.exe src\train.py --no-real-data
```

El entrenamiento completo crea:

- `models/asl_cnn.keras`
- `models/class_map.json`
- `reports/metrics.json`
- `reports/classification_report.txt`
- `reports/confusion_matrix.png`

## Ejecutar la cámara

```powershell
.\.venv\Scripts\python.exe src\app.py
```

Controles:

- `q` o `ESC`: salir.
- Si no abre la cámara, cambia `CAMERA_INDEX` en `.env` de `0` a `1`.

Si ves una pantalla negra con un ícono de cámara tachada, OpenCV está recibiendo
la imagen de una cámara bloqueada, apagada o incorrecta. Prueba:

```powershell
.\.venv\Scripts\python.exe src\camera_probe.py
```

## Ejecutar pruebas

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

Luego usa en `.env` el índice que muestre una imagen real, por ejemplo:

```env
CAMERA_INDEX=1
```

También revisa el permiso de cámara de Windows, el obturador físico del portátil
o la tecla de privacidad de la cámara.

## Estructura

```text
src/
  app.py              # app de cámara en tiempo real (24 letras estáticas)
  train.py            # entrenamiento, validación y métricas
  real_images.py      # carga y recorte de imágenes reales por letra
  labels.py           # mapa Sign MNIST -> letras (excluye J y Z)
  preprocessing.py    # normalización para CNN
  hand_tracking.py    # detección de manos con MediaPipe o ROI fijo
  camera_probe.py     # utilidad para detectar el índice de cámara correcto
tests/
  test_*.py           # pruebas unitarias básicas
```
