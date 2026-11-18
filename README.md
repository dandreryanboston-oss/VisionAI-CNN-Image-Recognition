# VisionAI - CNN Image Recognition

## English

VisionAI is a Streamlit application for image recognition and CNN experimentation. It includes:

- Image classification with pretrained MobileNetV2 (ImageNet).
- A cats-versus-dogs comparison module and an optional custom CNN.
- Zero-shot image recognition with CLIP and editable candidate categories.
- CNN training, evaluation, and a model-results dashboard.

### Requirements

- Python 3.11
- The pinned packages in `requirements.txt`
- An internet connection on first use to download pretrained model weights

### Setup

From the project directory, create and activate a virtual environment, then install the dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Run the application

```powershell
streamlit run app.py
```

### Train and evaluate the custom CNN

Download and extract the Cats vs Dogs dataset, then train and evaluate the model:

```powershell
python download_dataset.py
python ai\train_cnn.py
python ai\evaluate_models.py
```

Dataset files, local image samples, model weights, and ML caches are intentionally excluded from Git. Training and evaluation create their outputs in `models/`; the custom CNN must be trained locally before its optional features can be used.

## Español

VisionAI es una aplicación de Streamlit para reconocer imágenes y experimentar con redes neuronales convolucionales. Incluye:

- Clasificación de imágenes con MobileNetV2 preentrenado en ImageNet.
- Comparación de imágenes de gatos y perros y una CNN personalizada opcional.
- Reconocimiento de imágenes sin entrenamiento adicional con CLIP y categorías editables.
- Entrenamiento, evaluación y un panel de resultados de los modelos.

### Requisitos

- Python 3.11
- Los paquetes fijados en `requirements.txt`
- Conexión a Internet durante el primer uso para descargar los pesos preentrenados

### Instalación

Desde la carpeta del proyecto, crea y activa un entorno virtual e instala las dependencias:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

### Ejecutar la aplicación

```powershell
streamlit run app.py
```

### Entrenar y evaluar la CNN personalizada

Descarga y extrae el conjunto de datos de gatos y perros; después, entrena y evalúa el modelo:

```powershell
python download_dataset.py
python ai\train_cnn.py
python ai\evaluate_models.py
```

Los conjuntos de datos, las imágenes locales, los pesos de los modelos y las cachés de aprendizaje automático se excluyen intencionalmente de Git. El entrenamiento y la evaluación guardan sus resultados en `models/`; es necesario entrenar la CNN personalizada localmente para utilizar sus funciones opcionales.
