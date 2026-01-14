# Chronic Risk Backend  
REST API desarrollada en Python/Flask para el procesamiento de datos tabulares, inferencia de modelos de predicción de riesgo, generación de datasets sintéticos y entrega de simulaciones clínicas dinámicas para una plataforma educativa.

---

## Características principales

- Predicción de riesgo para enfermedades crónicas (diabetes, hipertensión, obesidad, cardiovascular, etc.).
- Carga automática de modelos entrenados desde la carpeta `models/`.
- Lectura de datasets crudos desde `data_raw/`.
- Scripts internos para:
  - limpieza y curación de datos (`prepare_datasets.py`)
  - generación de datos sintéticos (`curate_and_synthesize.py`)
  - entrenamiento de modelos (`train_models.py`)
- API tipo REST con endpoints para:
  - `/health` → estado del servicio  
  - `/metrics/<disease>` → métricas de cada modelo  
  - `/config/<disease>` → configuración de features  
  - `/predict/<disease>` → predicción a partir de un JSON con features  

---

## Requisitos

- Python 3.9 o superior
- pip instalado
- entorno virtual de python

---

## Instalación

pip install -r requirements.txt

## Execution

python app.py