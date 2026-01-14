import json
import os
import sqlite3
import datetime
from typing import Dict, Any, List

import random
import glob

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load

app = Flask(__name__)
CORS(app)

BASE_MODELS = "models"
DB_NAME = "medical_history.db"  # <--- Nombre de la Base de Datos

# Archivos por enfermedad
FILES = {
    "diabetes": {
        "pipeline": os.path.join(BASE_MODELS, "diabetes_pipeline.pkl"),
        "features": os.path.join(BASE_MODELS, "diabetes_features.json"),
        "metrics":  os.path.join(BASE_MODELS, "diabetes_metrics.json"),
    },
    "hipertension": {
        "pipeline": os.path.join(BASE_MODELS, "hipertension_pipeline.pkl"),
        "features": os.path.join(BASE_MODELS, "hipertension_features.json"),
        "metrics":  os.path.join(BASE_MODELS, "hipertension_metrics.json"),
    },
    "obesidad": {
        "pipeline": os.path.join(BASE_MODELS, "obesidad_pipeline.pkl"),
        "features": os.path.join(BASE_MODELS, "obesidad_features.json"),
        "metrics":  os.path.join(BASE_MODELS, "obesidad_metrics.json"),
    },
    "cardiovascular": {
        "pipeline": os.path.join(BASE_MODELS, "cardiovascular_pipeline.pkl"),
        "features": os.path.join(BASE_MODELS, "cardiovascular_features.json"),
        "metrics":  os.path.join(BASE_MODELS, "cardiovascular_metrics.json"),
    },
}

MODELS: Dict[str, Any] = {}
FEATURES: Dict[str, List[str]] = {}

# ==========================================
# 1. FUNCIÓN DE BASE DE DATOS
# ==========================================
def init_db():
    """Crea la tabla si no existe. Esto cumple con el requisito de tesis."""
    conn = sqlite3.connect(DB_NAME)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            disease TEXT,
            input_data TEXT,
            prediction INTEGER,
            probability REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction_to_db(disease, input_data, prediction, probability):
    """Guarda el historial de uso."""
    try:
        conn = sqlite3.connect(DB_NAME)
        cursor = conn.cursor()
        # Guardamos el input como texto JSON para no complicarnos con columnas
        cursor.execute('''
            INSERT INTO predictions (disease, input_data, prediction, probability)
            VALUES (?, ?, ?, ?)
        ''', (disease, json.dumps(input_data), int(prediction), float(probability)))
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"⚠️ Error guardando en BD: {e}")

# ==========================================
# CARGA DE MODELOS
# ==========================================
def _load_all():
    for dis, paths in FILES.items():
        if os.path.exists(paths["pipeline"]):
            MODELS[dis] = load(paths["pipeline"])
        if os.path.exists(paths["features"]):
            with open(paths["features"], "r", encoding="utf-8") as f:
                FEATURES[dis] = json.load(f)

def _safe_get(payload: Dict[str, Any], key: str):
    if key in payload: return payload[key]
    if " " in key:
        alt = key.replace(" ", "_")
        if alt in payload: return payload[alt]
    low = {k.lower(): v for k, v in payload.items()}
    if key.lower() in low: return low[key.lower()]
    if " " in key and key.replace(" ", "_").lower() in low:
        return low[key.replace(" ", "_").lower()]
    return None

# ==========================================
# FUNCIÓN AUXILIAR PARA DATOS SINTÉTICOS
# ==========================================
def get_random_sample(disease):
    """
    Busca datos SINTÉTICOS priorizando CTGAN (que son los médicamente correctos).
    """
    disease = disease.lower()
    base_dir = os.path.join("data_curated", disease)
    
    # 1. Intentar buscar específicamente CTGAN primero (Recomendado)
    ctgan_pattern = os.path.join(base_dir, f"{disease}_synthetic_ctgan*.csv")
    found_files = glob.glob(ctgan_pattern)
    
    # 2. Si no hay CTGAN, buscar cualquier otro sintético (Fallback, por si acaso)
    if not found_files:
        print(f"⚠️ No se encontró CTGAN para {disease}, buscando otros...")
        any_pattern = os.path.join(base_dir, f"{disease}_synthetic*.csv")
        found_files = glob.glob(any_pattern)

    csv_path = None
    source_type = "real"

    if found_files:
        # Tomamos el primero (ahora seguro será CTGAN si existe)
        csv_path = found_files[0]
        source_type = "synthetic"
        # Opcional: imprimir cuál estamos usando para estar seguros
        print(f"🎲 Usando datos sintéticos: {os.path.basename(csv_path)}")
    else:
        # 3. Fallback final: Datos reales procesados
        csv_path = os.path.join("data_processed", f"{disease}_dataset.csv")
        print(f"⚠️ No se hallaron sintéticos para {disease}. Usando datos reales procesados.")

    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
        
        if "target" in df.columns:
            df = df.drop(columns=["target"])
            
        sample = df.sample(1).iloc[0].to_dict()
        
        for key, val in sample.items():
            if isinstance(val, (np.integer, np.int64)):
                sample[key] = int(val)
            elif isinstance(val, (np.floating, np.float64)):
                sample[key] = round(float(val), 2)
        
        sample['_source_type'] = source_type
        return sample
    except Exception as e:
        print(f"⚠️ Error leyendo CSV: {e}")
        return None

@app.get("/health")
def health():
    return jsonify({"status": "ok", "database": "sqlite_connected"})

@app.get("/metrics/<disease>")
def get_metrics(disease: str):
    disease = disease.lower()
    if disease not in FILES:
        return jsonify({"error": "unknown disease"}), 404
    mpath = FILES[disease]["metrics"]
    if not os.path.exists(mpath):
        return jsonify({"error": "metrics not found"}), 404
    with open(mpath, "r", encoding="utf-8") as f:
        metrics = json.load(f)
    return jsonify(metrics)

@app.get("/config/<disease>")
def get_config(disease: str):
    disease = disease.lower()
    if disease not in FEATURES:
        return jsonify({"error": "features not found"}), 404
    feats = FEATURES[disease]
    ranges = { "age": [18, 100], "bmi": [15, 50], "glucose": [60, 260], "blood_pressure": [60, 130] }
    gender_opts = sorted([f.split("gender_")[1] for f in feats if f.startswith("gender_")])
    smoke_opts  = sorted([f.split("smoking_history_")[1] for f in feats if f.startswith("smoking_history_")])
    return jsonify({
        "disease": disease,
        "features": feats,
        "ranges": ranges,
        "categoricals": { "gender": gender_opts, "smoking_history": smoke_opts }
    })

@app.post("/predict/<disease>")
def predict(disease: str):
    disease = disease.lower()
    if disease not in MODELS or disease not in FEATURES:
        return jsonify({"error": "model or features not loaded"}), 500

    try:
        payload = request.get_json(force=True) or {}
    except Exception:
        return jsonify({"error": "invalid JSON"}), 400

    feats = FEATURES[disease]
    row = []
    missing = []
    
    # Variables clave para las reglas
    clinical_glucose = 0
    clinical_hba1c = 0
    clinical_bp = 0
    clinical_bmi = 0

    for f in feats:
        val = _safe_get(payload, f)
        
        # Capturamos valores clínicos
        if f in ["glucose", "blood_glucose_level"]: clinical_glucose = float(val or 0)
        if f == "hba1c_level": clinical_hba1c = float(val or 0)
        if f == "blood_pressure": clinical_bp = float(val or 0)
        if f == "bmi": clinical_bmi = float(val or 0)

        if val is None:
            if f.startswith(("gender_", "smoking_history_", "cholesterol_", "glucose_", "bp_", "ethnicity_", "race_")) or "_" in f:
                val = 0
            else:
                val = 0
                missing.append(f)
        row.append(val)

    X = np.array([row], dtype=float)
    model = MODELS[disease]
    
    # 1. Predicción Base de la IA
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0, 1])
    else:
        pred = int(model.predict(X)[0])
        prob = float(pred)

    # =========================================================================
    # REGLAS CLÍNICAS PROGRESIVAS (Dynamic Expert System)
    # Ahora la probabilidad escala suavemente con la gravedad del síntoma.
    # =========================================================================
    
    # --- DIABETES ---
    if disease == "diabetes":
        # Glucosa > 200 es diabetes casi segura.
        if clinical_glucose >= 200:
             prob = max(prob, 0.96)
        # Rango Diabético (126 - 200): Escala de 0.85 a 0.95
        elif clinical_glucose >= 126:
            extra = (clinical_glucose - 126) * 0.001 # Sube un poco por cada mg/dL extra
            prob = max(prob, 0.85 + extra)
        # Prediabetes (100 - 125): Escala de 0.40 a 0.60 (Puente para que no se caiga a 0)
        elif clinical_glucose >= 100:
            extra = (clinical_glucose - 100) * 0.008 
            prob = max(prob, 0.40 + extra)
        
        # HbA1c también empuja hacia arriba
        if clinical_hba1c >= 6.5:
            extra_a1c = (clinical_hba1c - 6.5) * 0.05
            prob = max(prob, 0.85 + extra_a1c)

    # --- HIPERTENSIÓN ---
    elif disease == "hipertension":
        # Crisis Hipertensiva (>180): Casi 100%
        if clinical_bp >= 180:
            prob = max(prob, 0.98)
            
        # Hipertensión Grado 2 (140 - 180): Escala de 0.85 a 0.97
        elif clinical_bp >= 140:
            # Por cada punto arriba de 140, sumamos 0.003 (ej. 160 -> +0.06)
            extra = (clinical_bp - 140) * 0.003
            prob = max(prob, 0.85 + extra)
            
        # Hipertensión Grado 1 (130 - 139): Escala de 0.60 a 0.80
        elif clinical_bp >= 130:
            extra = (clinical_bp - 130) * 0.02 
            prob = max(prob, 0.60 + extra)
            
        # Elevada (120 - 129): Escala de 0.30 a 0.50 (Para que no se desplome a 0)
        elif clinical_bp >= 120:
            extra = (clinical_bp - 120) * 0.02
            prob = max(prob, 0.30 + extra)

    # --- OBESIDAD ---
    elif disease == "obesidad":
        if clinical_bmi >= 40: # Obesidad mórbida
            prob = max(prob, 0.99)
        elif clinical_bmi >= 30: # Obesidad
            # Escalar entre 0.90 y 0.98 según qué tan alto sea el BMI
            extra = (clinical_bmi - 30) * 0.005
            prob = max(prob, 0.90 + extra)
        elif clinical_bmi >= 25: # Sobrepeso (riesgo medio)
            extra = (clinical_bmi - 25) * 0.04
            prob = max(prob, 0.40 + extra)

    # Limitar siempre a máximo 1.0 (por si la suma se pasa)
    prob = min(prob, 1.0)
    # =========================================================================

    pred_class = 1 if prob >= 0.5 else 0
    
    log_prediction_to_db(disease, payload, pred_class, prob)

    return jsonify({
        "disease": disease,
        "probability": prob,
        "prediction": pred_class,
        "missing_filled_as_zero": missing
    })

@app.get("/synthetic/<disease>")
def get_synthetic(disease):
    disease = disease.lower()
    
    # Validar que la enfermedad existe en tu sistema
    if disease not in FILES:
        return jsonify({"error": "disease not supported"}), 404
        
    sample = get_random_sample(disease)
    
    if not sample:
        # Fallback: Si no hay CSV, devolvemos un error controlado
        return jsonify({"error": "could not generate synthetic data"}), 500
        
    return jsonify(sample)

if __name__ == "__main__":
    _load_all()
    init_db()  # <--- Inicializa la BD al arrancar
    print("Backend corriendo con Base de Datos SQLite activa.")
    app.run(host="0.0.0.0", port=8000, debug=True)