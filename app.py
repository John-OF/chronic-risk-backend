import json
import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load

app = Flask(__name__)
CORS(app)

BASE_MODELS = "models"

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

# Cache de modelos y features
MODELS: Dict[str, Any] = {}
FEATURES: Dict[str, List[str]] = {}

def _load_all():
    for dis, paths in FILES.items():
        if os.path.exists(paths["pipeline"]):
            MODELS[dis] = load(paths["pipeline"])
        if os.path.exists(paths["features"]):
            with open(paths["features"], "r", encoding="utf-8") as f:
                FEATURES[dis] = json.load(f)

def _safe_get(payload: Dict[str, Any], key: str):
    """Obtiene payload[key]. Si el feature tiene espacios, acepta también la versión con '_'."""
    if key in payload:
        return payload[key]
    if " " in key:
        alt = key.replace(" ", "_")
        if alt in payload:
            return payload[alt]
    # soporte extra: si el cliente manda todo en minúsculas
    low = {k.lower(): v for k, v in payload.items()}
    if key.lower() in low:
        return low[key.lower()]
    if " " in key and key.replace(" ", "_").lower() in low:
        return low[key.replace(" ", "_").lower()]
    return None

@app.get("/health")
def health():
    return jsonify({"status": "ok"})

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
    """Config mínima para sliders/inputs del frontend."""
    disease = disease.lower()
    if disease not in FEATURES:
        return jsonify({"error": "features not found"}), 404

    feats = FEATURES[disease]
    # rangos genéricos (ajústalos si tus diccionarios definen otros)
    ranges = {
        "age": [18, 100],
        "bmi": [15, 50],
        "glucose": [60, 260],
        "blood_pressure": [60, 130]
    }
    # detecta dummies de género y tabaco según features
    gender_opts = sorted([f.split("gender_")[1] for f in feats if f.startswith("gender_")])
    smoke_opts  = sorted([f.split("smoking_history_")[1] for f in feats if f.startswith("smoking_history_")])

    return jsonify({
        "disease": disease,
        "features": feats,
        "ranges": ranges,
        "categoricals": {
            "gender": gender_opts,
            "smoking_history": smoke_opts
        },
        "note": "Para claves con espacio (p.ej. 'smoking_history_not current'), también se acepta 'smoking_history_not_current'."
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

    # construimos el vector X en el orden EXACTO de FEATURES
    feats = FEATURES[disease]
    row = []
    missing = []
    for f in feats:
        val = _safe_get(payload, f)
        if val is None:
            # si es dummy (e.g., gender_Male) y no vino, asumimos 0
            if f.startswith(("gender_", "smoking_history_", "cholesterol_", "glucose_", "bp_", "ethnicity_", "race_")) or "_" in f:
                val = 0
            else:
                # variables numéricas importantes se ponen 0 si faltan
                val = 0
                missing.append(f)
        row.append(val)

    X = np.array([row], dtype=float)
    model = MODELS[disease]
    # scikit pipelines soportan predict_proba
    if hasattr(model, "predict_proba"):
        prob = float(model.predict_proba(X)[0, 1])
    else:
        # fallback (no debería entrar)
        pred = int(model.predict(X)[0])
        prob = float(pred)

    pred_class = 1 if prob >= 0.5 else 0
    return jsonify({
        "disease": disease,
        "probability": prob,
        "prediction": pred_class,
        "missing_filled_as_zero": missing
    })

if __name__ == "__main__":
    _load_all()
    # Flask dev server (suficiente para local/Windows)
    app.run(host="0.0.0.0", port=8000, debug=True)
