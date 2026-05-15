# Detector de abecedario ASL con CNN y camara

Proyecto local para Visual Studio Code. Detecta letras del alfabeto ASL con la camara y muestra la letra en pantalla.

## Enfoque

- CNN entrenada con `dataset/sign_mnist_train.csv` y evaluada con `dataset/sign_mnist_test.csv`.
- Metricas guardadas en `reports/`: accuracy, macro F1, matriz de confusion y reporte por clase.
- Camara en tiempo real con OpenCV y MediaPipe Hands.
- Letras `J` y `Z` detectadas por movimiento del dedo indice, porque Sign MNIST no trae esas etiquetas.

## Configuracion en Windows

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

Para una prueba rapida de entrenamiento:

```powershell
.\.venv\Scripts\python.exe src\train.py --epochs 1 --sample-limit 1200
```

El entrenamiento completo crea:

- `models/asl_cnn.keras`
- `models/class_map.json`
- `reports/metrics.json`
- `reports/classification_report.txt`
- `reports/confusion_matrix.png`

## Ejecutar la camara

```powershell
.\.venv\Scripts\python.exe src\app.py
```

Controles:

- `q` o `ESC`: salir.
- Si no abre la camara, cambia `CAMERA_INDEX` en `.env` de `0` a `1`.

Si ves una pantalla negra con un icono de camara tachada, OpenCV esta recibiendo
la imagen de una camara bloqueada, apagada o incorrecta. Prueba:

```powershell
.\.venv\Scripts\python.exe src\camera_probe.py
```

## Ejecutar pruebas

```powershell
.\.venv\Scripts\python.exe -m unittest discover -v
```

Luego usa en `.env` el indice que muestre una imagen real, por ejemplo:

```env
CAMERA_INDEX=1
```

Tambien revisa el permiso de camara de Windows, el obturador fisico del portatil
o la tecla de privacidad de la camara.

## Estructura

```text
src/
  app.py              # app de camara en tiempo real
  train.py            # entrenamiento, validacion y metricas
  labels.py           # mapa Sign MNIST -> letras
  preprocessing.py    # normalizacion para CNN
  motion_letters.py   # detector temporal para J/Z
tests/
  test_*.py           # pruebas unitarias basicas
```

## Nota importante sobre J y Z

El dataset Sign MNIST omite las etiquetas 9 y 25 porque `J` y `Z` son signos con movimiento. Por eso el proyecto usa un enfoque hibrido: CNN para 24 letras estaticas y detector temporal para `J`/`Z`.
