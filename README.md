# Detector de abecedario ASL estático con MediaPipe

Proyecto local para Visual Studio Code. Detecta **24 letras estáticas** del alfabeto ASL con la cámara y muestra una letra solo cuando la seña es clara y estable.

> El alcance del proyecto excluye `J` y `Z` porque no son señas estáticas. La app principal reconoce `A-I` y `K-Y`.

## Enfoque principal

- MediaPipe Hands detecta 21 puntos de la mano.
- `src/landmarks.py` convierte esos puntos en un vector normalizado de 42 valores.
- `src/train_landmarks.py` entrena un `RandomForestClassifier` con las fotos reales ubicadas en `dataset/senas_reales_entrenamiento/`.
- `src/app.py` usa el modelo de landmarks en vivo y aplica rechazo conservador:
  - confianza mínima (`CONFIDENCE_THRESHOLD`)
  - margen mínimo contra la segunda clase (`MARGIN_THRESHOLD`)
  - varios fotogramas estables (`STABLE_FRAMES`)

La CNN de Sign MNIST queda como comparación técnica en `src/train.py`, pero no es el flujo principal de la cámara.

## Configuración en Windows

Abre esta carpeta en Visual Studio Code y ejecuta:

```powershell
C:\Users\jorda\AppData\Local\Programs\Python\Python312\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Entrenar el modelo principal

El dataset real debe tener subcarpetas por letra:

```text
dataset/senas_reales_entrenamiento/
  A/
  B/
  C/
  ...
  Y/
```

No incluyas carpetas `J` ni `Z` para este proyecto.

Entrenamiento completo:

```powershell
.\.venv\Scripts\python.exe src\train_landmarks.py
```

Prueba rápida de entrenamiento:

```powershell
.\.venv\Scripts\python.exe src\train_landmarks.py --max-per-class 3
```

El entrenamiento genera:

- `models/asl_landmarks.joblib`
- `models/landmark_class_map.json`
- `reports/landmark_metrics.json`
- `reports/landmark_classification_report.txt`
- `reports/landmark_confusion_matrix.png`

Revisa `reports/landmark_metrics.json`: si alguna letra tiene pocas imágenes útiles, agrega más fotos con buena luz, mano completa y variaciones leves de posición.

El entrenamiento también descarta imágenes idénticas que aparezcan en carpetas de letras distintas. Eso evita que el modelo aprenda etiquetas contradictorias, por ejemplo una misma foto repetida en `V/` y `W/`.

## Ejecutar la cámara

```powershell
.\.venv\Scripts\python.exe src\app.py
```

Controles:

- `q` o `ESC`: salir.
- Si no abre la cámara, cambia `CAMERA_INDEX` en `.env`.

Configuración recomendada:

```env
CAMERA_INDEX=0
CONFIDENCE_THRESHOLD=0.80
MARGIN_THRESHOLD=0.15
STABLE_FRAMES=4
```

Si el detector muestra `-`, significa que no hay mano, la confianza es baja, el margen es pequeño o la predicción aún no es estable.

## Probar índices de cámara

```powershell
.\.venv\Scripts\python.exe src\camera_probe.py
```

Luego usa en `.env` el índice que muestre imagen real.

## Ejecutar pruebas

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

## Comparación con CNN

El flujo anterior con CNN sigue disponible para comparar resultados contra Sign MNIST:

```powershell
.\.venv\Scripts\python.exe src\train.py
```

Ese modelo guarda `models/asl_cnn.keras` y reportes base, pero la cámara en vivo usa `models/asl_landmarks.joblib`.

## Estructura

```text
src/
  app.py               # app de cámara en tiempo real con landmarks
  train_landmarks.py   # entrenamiento principal con fotos reales
  landmarks.py         # extracción y normalización de puntos de MediaPipe
  hand_tracking.py     # detección de mano y dibujo de landmarks
  train.py             # entrenamiento CNN de comparación
  real_images.py       # carga de imágenes reales por letra
  labels.py            # mapa de 24 letras estáticas
  preprocessing.py     # preprocesamiento usado por la CNN
  camera_probe.py      # utilidad para encontrar cámara
tests/
  test_*.py            # pruebas unitarias del flujo estático
```
## Data set: 
https://drive.google.com/drive/folders/1WLcj9njN9CjMo-PWBhzANKolAB1X_1J8?usp=drive_link

## Gran parte de las imagenes fueron sacadas del dataset: 
https://www.kaggle.com/datasets/kirlelea/spanish-sign-language-alphabet-static
