# train_models.py
import os, json
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from joblib import dump

PROCESSED_DIR = "data_processed"
CURATED_DIR = "data_curated"
MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)

DATASETS = ["diabetes", "hipertension", "obesidad", "cardiovascular"]

FEATURES = [
    "age","pregnancies","glucose","blood_pressure","skin_thickness","insulin",
    "bmi","diabetes_pedigree","hba1c_level","blood_glucose_level",
    "gender_Female","gender_Male",
    "smoking_history_current","smoking_history_former","smoking_history_never",
    "smoking_history_ever","smoking_history_not current",
    "hypertension","heart_disease"
]

def load_split_or_fallback(name: str):
    """Usa train/test de data_curated si existen; si no, hace split desde data_processed."""
    curated_train = os.path.join(CURATED_DIR, name, f"{name}_train.csv")
    curated_test  = os.path.join(CURATED_DIR, name, f"{name}_test.csv")
    processed_all = os.path.join(PROCESSED_DIR, f"{name}_dataset.csv")

    if os.path.exists(curated_train) and os.path.exists(curated_test):
        train_df = pd.read_csv(curated_train, low_memory=False)
        test_df  = pd.read_csv(curated_test,  low_memory=False)
    else:
        df = pd.read_csv(processed_all, low_memory=False)
        train_df, test_df = train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["target"]
        )
    return train_df, test_df

def build_pipeline(name: str) -> Pipeline:
    # Desbalance fuerte en hipertensión → class_weight="balanced"
    class_weight = "balanced" if name == "hipertension" else None
    clf = LogisticRegression(max_iter=1000, class_weight=class_weight, n_jobs=None)
    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),  # soporta columnas binarias y numéricas sin romper
        ("clf", clf),
    ])
    return pipe

def train_one(name: str):
    print(f"\n=== Entrenando {name} ===")
    train_df, test_df = load_split_or_fallback(name)

    # asegura columnas
    for c in FEATURES + ["target"]:
        if c not in train_df.columns:
            train_df[c] = 0
        if c not in test_df.columns:
            test_df[c] = 0

    X_train = train_df[FEATURES].values
    y_train = train_df["target"].astype(int).values
    X_test  = test_df[FEATURES].values
    y_test  = test_df["target"].astype(int).values

    pipe = build_pipeline(name)
    pipe.fit(X_train, y_train)

    # métricas
    y_proba = pipe.predict_proba(X_test)[:,1]
    y_pred  = (y_proba >= 0.5).astype(int)
    auc = roc_auc_score(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    # guardar modelo y metadatos
    model_path = os.path.join(MODELS_DIR, f"{name}_pipeline.pkl")
    dump(pipe, model_path)

    meta = {
        "dataset": name,
        "features": FEATURES,
        "auc": auc,
        "report": report,
    }
    with open(os.path.join(MODELS_DIR, f"{name}_metrics.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    with open(os.path.join(MODELS_DIR, f"{name}_features.json"), "w", encoding="utf-8") as f:
        json.dump(FEATURES, f)

    print(f"✅ {name}: AUC={auc:.3f} | modelo: {model_path}")

def main():
    for name in DATASETS:
        train_one(name)

if __name__ == "__main__":
    main()
