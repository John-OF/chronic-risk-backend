# prepare_datasets.py
import os
import pandas as pd
import numpy as np

# ====== CONFIG ======
RAW_DIR = "data_raw"
PROCESSED_DIR = "data_processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# Cargar todos los Dataset
CSV_FILES = [
    # diabetes
    "diabetes.csv",
    "diabetes_prediction_dataset.csv",
    "Dataset_of_Diabetes.csv",
    # hipertensión
    "Hipertension_Arterial_Mexico.csv",
    "hypertension_dataset.csv",
    # obesidad
    "ObesityDataSet_raw_and_data_sinthetic.csv",
    # cardiovascular
    "cardio_train.csv",
]

# Esquema base que se va a garantizar
COMMON_SCHEMA = [
    "age",
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "diabetes_pedigree",
    "hypertension",
    "heart_disease",
    "hba1c_level",
    "blood_glucose_level",
    "gender_Female",
    "gender_Male",
    "smoking_history_current",
    "smoking_history_former",
    "smoking_history_never",
    "smoking_history_ever",
    "smoking_history_not current",
    "target",   # lo sobreescribimos según el dataset final
]


# =========================================================
# 1. CARGA
# =========================================================
def load_csvs():
    loaded = []
    for name in CSV_FILES:
        path = os.path.join(RAW_DIR, name)
        if not os.path.exists(path):
            print(f"⚠️ No existe: {name}")
            continue

        df = None

        # 1) intento estándar (coma, utf-8 → latin-1)
        try:
            df = pd.read_csv(path)
        except UnicodeDecodeError:
            df = pd.read_csv(path, encoding="latin-1")
        except Exception:
            df = None

        # 2) si solo trae 1 columna, probamos con ; (utf-8 → latin-1)
        if df is not None and df.shape[1] == 1:
            try:
                df2 = pd.read_csv(path, sep=";")
            except UnicodeDecodeError:
                df2 = pd.read_csv(path, sep=";", encoding="latin-1")
            # si mejoró, nos quedamos con ; 
            if df2.shape[1] > df.shape[1]:
                df = df2

        if df is None:
            print(f"❌ No se pudo cargar: {name}")
            continue

        print(f"✅ Cargado: {name} -> {df.shape}")
        loaded.append((name, df))

    return loaded



# =========================================================
# 2. DETECTORES
# =========================================================

# ---- diabetes PIMA clásico
def is_pima_like(df):
    needed = {
        "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"
    }
    return needed.issubset(df.columns)

# ---- diabetes prediction (Kaggle grande)
def is_prediction_like(df):
    lower = {c.lower() for c in df.columns}
    return {"age", "bmi", "hypertension"}.issubset(lower) and ("diabetes" in lower)

# ---- diabetes de laboratorio (tu "Dataset of Diabetes .csv")
def is_lab_diabetes_like(df):
    cols = {c.lower().strip() for c in df.columns}
    return (
        "class" in cols or "clas" in cols or "status" in cols
    ) and (
        "hba1c" in cols
        or "hb1ac" in cols
        or "valor_hemoglobina_glucosilada" in cols
        or "hemoglobina_glucosilada" in cols
    )

# ---- hipertensión México (español)
def is_mexican_hypertension(df):
    cols = {c.lower().strip() for c in df.columns}
    return (
        "riesgo_hipertension" in cols
        or ("tension_arterial" in cols and "edad" in cols)
    )

# ---- hipertensión inglés
def is_english_hypertension(df):
    cols = {c.lower().strip() for c in df.columns}
    return "hypertension" in cols and ("age" in cols or "bmi" in cols)

# ---- obesidad (UCI)
def is_obesity_uci(df):
    cols = {c.lower().strip() for c in df.columns}
    # dataset típico: Gender, Age, Height, Weight, ..., NObeyesdad
    return "nobeyesdad" in cols or "obesity" in cols

# ---- cardiovascular (Kaggle cardio_train)
def is_cardio_kaggle(df):
    cols = {c.lower().strip() for c in df.columns}
    # cardio_train trae: id, age (en días), gender, height, weight, ap_hi, ap_lo, cholesterol, gluc, smoke, alco, active, cardio
    return "cardio" in cols and "ap_hi" in cols and "ap_lo" in cols


# =========================================================
# 3. NORMALIZADORES
# =========================================================

def normalize_pima(df):
    df = df.copy()
    df = df.rename(columns={
        "Pregnancies": "pregnancies",
        "Glucose": "glucose",
        "BloodPressure": "blood_pressure",
        "SkinThickness": "skin_thickness",
        "Insulin": "insulin",
        "BMI": "bmi",
        "DiabetesPedigreeFunction": "diabetes_pedigree",
        "Age": "age",
        "Outcome": "target",
    })

    # corrige ceros imposibles
    for col in ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]:
        mask = df[col] == 0
        if mask.any():
            median_val = df.loc[~mask, col].median()
            df.loc[mask, col] = median_val
            print(f"🔧 [PIMA] {col}: {mask.sum()} ceros → {median_val}")

    # columnas faltantes
    df["hypertension"] = 0
    df["heart_disease"] = 0
    df["hba1c_level"] = 0.0
    df["blood_glucose_level"] = df["glucose"]

    for col in [
        "gender_Female", "gender_Male",
        "smoking_history_current", "smoking_history_former",
        "smoking_history_never", "smoking_history_ever",
        "smoking_history_not current",
    ]:
        df[col] = 0

    return df


def normalize_prediction(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    def get_col(name):
        return lower_map.get(name.lower())

    out = pd.DataFrame()

    out["age"] = df[get_col("age")]
    out["bmi"] = df[get_col("bmi")]
    out["hypertension"] = df[get_col("hypertension")] if get_col("hypertension") else 0
    out["heart_disease"] = df[get_col("heart_disease")] if get_col("heart_disease") else 0
    out["hba1c_level"] = df[get_col("hba1c_level")] if get_col("hba1c_level") else 0
    out["blood_glucose_level"] = df[get_col("blood_glucose_level")] if get_col("blood_glucose_level") else 0

    # diabetes
    if get_col("diabetes"):
        out["target"] = df[get_col("diabetes")]
    else:
        out["target"] = 0

    # pima-like
    out["pregnancies"] = 0
    out["glucose"] = out["blood_glucose_level"]
    out["blood_pressure"] = 0
    out["skin_thickness"] = 0
    out["insulin"] = 0
    out["diabetes_pedigree"] = 0

    # gender
    out["gender_Female"] = 0
    out["gender_Male"] = 0
    if get_col("gender"):
        g = df[get_col("gender")].astype(str).str.lower().str.strip()
        out.loc[g == "female", "gender_Female"] = 1
        out.loc[g == "male", "gender_Male"] = 1

    # smoking
    for col in [
        "smoking_history_current", "smoking_history_former",
        "smoking_history_never", "smoking_history_ever",
        "smoking_history_not current",
    ]:
        out[col] = 0

    if get_col("smoking_history"):
        s = df[get_col("smoking_history")].astype(str).str.lower().str.strip()
        out.loc[s == "current", "smoking_history_current"] = 1
        out.loc[s == "former", "smoking_history_former"] = 1
        out.loc[s == "never", "smoking_history_never"] = 1
        out.loc[s == "ever", "smoking_history_ever"] = 1
        out.loc[s == "not current", "smoking_history_not current"] = 1

    return out


def normalize_lab_diabetes(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    def get_col(name):
        return lower_map.get(name.lower())

    out = pd.DataFrame()

    out["age"] = df[get_col("age")] if get_col("age") else 0

    # BMI
    if get_col("bmi"):
        out["bmi"] = df[get_col("bmi")]
    elif get_col("BMI"):
        out["bmi"] = df[get_col("BMI")]
    else:
        out["bmi"] = 0

    # HbA1c
    if get_col("hba1c"):
        out["hba1c_level"] = df[get_col("hba1c")]
    elif get_col("valor_hemoglobina_glucosilada"):
        out["hba1c_level"] = df[get_col("valor_hemoglobina_glucosilada")]
    else:
        out["hba1c_level"] = 0

    # no hay glucosa directa
    out["glucose"] = 0
    out["blood_glucose_level"] = 0
    out["blood_pressure"] = 0
    out["skin_thickness"] = 0
    out["insulin"] = 0
    out["diabetes_pedigree"] = 0
    out["pregnancies"] = 0

    # género
    out["gender_Female"] = 0
    out["gender_Male"] = 0
    if get_col("gender"):
        g = df[get_col("gender")].astype(str).str.lower().str.strip()
        out.loc[g.str.startswith("f"), "gender_Female"] = 1
        out.loc[g.str.startswith("m"), "gender_Male"] = 1

    # smoking
    for col in [
        "smoking_history_current", "smoking_history_former",
        "smoking_history_never", "smoking_history_ever",
        "smoking_history_not current",
    ]:
        out[col] = 0

    out["hypertension"] = 0
    out["heart_disease"] = 0

    # target desde CLASS
    if get_col("class"):
        c = df[get_col("class")].astype(str).str.upper().str.strip()
        out["target"] = (c != "N").astype(int)
    elif get_col("status"):
        c = df[get_col("status")].astype(str).str.lower().str.strip()
        out["target"] = (c != "normal").astype(int)
    else:
        out["target"] = 0

    return out


def normalize_mexican_hypertension(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    def get_col(name):
        return lower_map.get(name.lower())

    out = pd.DataFrame()

    out["age"] = df[get_col("edad")] if get_col("edad") else 0

    # BMI
    if get_col("masa_corporal"):
        out["bmi"] = df[get_col("masa_corporal")]
    else:
        out["bmi"] = 0

    # glucosa
    if get_col("resultado_glucosa"):
        out["glucose"] = df[get_col("resultado_glucosa")]
        out["blood_glucose_level"] = df[get_col("resultado_glucosa")]
    else:
        out["glucose"] = 0
        out["blood_glucose_level"] = 0

    # HbA1c
    if get_col("valor_hemoglobina_glucosilada"):
        out["hba1c_level"] = df[get_col("valor_hemoglobina_glucosilada")]
    else:
        out["hba1c_level"] = 0

    # pima-not
    out["blood_pressure"] = 0
    out["pregnancies"] = 0
    out["skin_thickness"] = 0
    out["insulin"] = 0
    out["diabetes_pedigree"] = 0

    # género (parece 1=hombre, 2=mujer)
    out["gender_Female"] = 0
    out["gender_Male"] = 0
    if get_col("sexo"):
        s = df[get_col("sexo")]
        out.loc[s == 2, "gender_Female"] = 1
        out.loc[s == 1, "gender_Male"] = 1

    # smoking
    for col in [
        "smoking_history_current", "smoking_history_former",
        "smoking_history_never", "smoking_history_ever",
        "smoking_history_not current",
    ]:
        out[col] = 0

    out["heart_disease"] = 0

    # target
    if get_col("riesgo_hipertension"):
        rh = df[get_col("riesgo_hipertension")]
        out["hypertension"] = rh.apply(lambda x: 1 if str(x).strip() not in ["0", "0.0", "", "nan"] else 0)
        out["target"] = out["hypertension"]
    else:
        out["hypertension"] = 0
        out["target"] = 0

    return out


def normalize_english_hypertension(df):
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    def get_col(name):
        return lower_map.get(name.lower())

    out = pd.DataFrame()

    out["age"] = df[get_col("age")] if get_col("age") else 0
    out["bmi"] = df[get_col("bmi")] if get_col("bmi") else 0

    # glucosa
    if get_col("glucose"):
        out["glucose"] = df[get_col("glucose")]
        out["blood_glucose_level"] = df[get_col("glucose")]
    else:
        out["glucose"] = 0
        out["blood_glucose_level"] = 0

    out["hba1c_level"] = 0
    out["blood_pressure"] = 0
    out["pregnancies"] = 0
    out["skin_thickness"] = 0
    out["insulin"] = 0
    out["diabetes_pedigree"] = 0

    # género
    out["gender_Female"] = 0
    out["gender_Male"] = 0
    if get_col("gender"):
        g = df[get_col("gender")].astype(str).str.lower().str.strip()
        out.loc[g == "female", "gender_Female"] = 1
        out.loc[g == "male", "gender_Male"] = 1

    # smoking
    for col in [
        "smoking_history_current", "smoking_history_former",
        "smoking_history_never", "smoking_history_ever",
        "smoking_history_not current",
    ]:
        out[col] = 0

    out["heart_disease"] = 0

    # target hipertensión
    if get_col("hypertension"):
        h = df[get_col("hypertension")].astype(str).str.lower().str.strip()
        out["hypertension"] = h.apply(lambda x: 1 if x in ["1", "yes", "true", "high", "y"] else 0)
        out["target"] = out["hypertension"]
    else:
        out["hypertension"] = 0
        out["target"] = 0

    return out


def normalize_obesity_uci(df):
    """
    ObesityDataSet_raw_and_data_sinthetic.csv
    tiene:
    Gender, Age, Height, Weight, ..., NObeyesdad

    Vamos a:
    - age -> age
    - bmi = weight / (height**2) (si height viene en metros; si viene en cm, convertimos)
    - target = 1 si NObeyesdad != 'Normal_Weight' (o 'Insufficient_Weight')
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    def get_col(name):
        return lower_map.get(name.lower())

    out = pd.DataFrame()

    # edad
    out["age"] = df[get_col("age")] if get_col("age") else 0

    # peso / altura
    if get_col("height") and get_col("weight"):
        h = df[get_col("height")].astype(float)
        w = df[get_col("weight")].astype(float)

        # algunos datasets lo traen en metros, otros en cm. Si la media es > 3 asumimos cm
        if h.mean() > 3:  # está en cm
            h = h / 100.0
        bmi = w / (h ** 2)
        out["bmi"] = bmi
    else:
        out["bmi"] = 0

    # no hay glucosa ni presión
    out["glucose"] = 0
    out["blood_glucose_level"] = 0
    out["blood_pressure"] = 0
    out["skin_thickness"] = 0
    out["insulin"] = 0
    out["diabetes_pedigree"] = 0
    out["pregnancies"] = 0
    out["hba1c_level"] = 0

    # género
    out["gender_Female"] = 0
    out["gender_Male"] = 0
    if get_col("gender"):
        g = df[get_col("gender")].astype(str).str.lower().str.strip()
        out.loc[g.str.startswith("f"), "gender_Female"] = 1
        out.loc[g.str.startswith("m"), "gender_Male"] = 1

    # smoking dummies (no hay)
    for col in [
        "smoking_history_current", "smoking_history_former",
        "smoking_history_never", "smoking_history_ever",
        "smoking_history_not current",
    ]:
        out[col] = 0

    # enfermedades
    out["hypertension"] = 0
    out["heart_disease"] = 0

    # 👉 target obesidad desde NObeyesdad
    if get_col("nobeyesdad"):
        cat = df[get_col("nobeyesdad")].astype(str).str.lower().str.strip()
        # lo binarizamos: 1 = sobrepeso / obesidad, 0 = normal / insuficiente
        bad = cat.apply(
            lambda x: 1
            if any(k in x for k in ["overweight", "obesity"])
            else 0
        )
        out["target"] = bad
    else:
        out["target"] = out["bmi"].apply(lambda x: 1 if x >= 30 else 0)

    return out


def normalize_cardio_kaggle(df):
    """
    cardio_train.csv:
    id,age,gender,height,weight,ap_hi,ap_lo,cholesterol,gluc,smoke,alco,active,cardio
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    lower_map = {c.lower(): c for c in df.columns}

    def get_col(name):
        return lower_map.get(name.lower())

    out = pd.DataFrame()

    # edad viene en días → pasamos a años
    if get_col("age"):
        age_days = df[get_col("age")].astype(float)
        out["age"] = (age_days / 365.25).round(0)
    else:
        out["age"] = 0

    # IMC
    if get_col("height") and get_col("weight"):
        h = df[get_col("height")].astype(float) / 100.0  # viene en cm
        w = df[get_col("weight")].astype(float)
        out["bmi"] = w / (h ** 2)
    else:
        out["bmi"] = 0

    # presión
    if get_col("ap_hi") and get_col("ap_lo"):
        # podríamos guardar la sistólica
        out["blood_pressure"] = df[get_col("ap_hi")]
    else:
        out["blood_pressure"] = 0

    # glucosa
    out["glucose"] = 0
    out["blood_glucose_level"] = 0
    if get_col("gluc"):
        # 1=normal, 2=above normal, 3=well above normal
        # lo podemos mapear a 90, 130, 180 para tener algo
        m = df[get_col("gluc")].map({1: 90, 2: 130, 3: 180})
        out["glucose"] = m.fillna(0)
        out["blood_glucose_level"] = m.fillna(0)

    out["skin_thickness"] = 0
    out["insulin"] = 0
    out["diabetes_pedigree"] = 0
    out["pregnancies"] = 0
    out["hba1c_level"] = 0

    # género
    out["gender_Female"] = 0
    out["gender_Male"] = 0
    if get_col("gender"):
        g = df[get_col("gender")]
        out.loc[g == 1, "gender_Female"] = 1  # en este dataset 1=female, 2=male
        out.loc[g == 2, "gender_Male"] = 1

    # smoking
    for col in [
        "smoking_history_current", "smoking_history_former",
        "smoking_history_never", "smoking_history_ever",
        "smoking_history_not current",
    ]:
        out[col] = 0

    # enfermedades
    out["hypertension"] = 0
    out["heart_disease"] = 0
    # este dataset ya trae el target cardio
    if get_col("cardio"):
        out["target"] = df[get_col("cardio")]
    else:
        out["target"] = 0

    return out


def normalize_unknown(df):
    base = pd.DataFrame(columns=COMMON_SCHEMA)
    base = pd.concat([base, pd.DataFrame(index=range(len(df)))], ignore_index=True)
    # evita FutureWarning en futuras versiones de pandas
    base = base.infer_objects(copy=False).fillna(0)
    print("⚠️ Dataset desconocido, se pasa como neutro")
    return base


def _coerce_binary_target(series):
    """Convierte cualquier variante textual/numérica a {0,1}."""
    s = series.astype(str).str.strip().str.lower()
    # mapea positivos
    pos = {"1", "yes", "true", "y", "high", "positivo", "sí", "si"}
    # mapea negativos
    neg = {"0", "no", "false", "n", "low", "negativo", "normal", "0.0", ""}

    def map_one(x):
        if x in pos:
            return 1
        if x in neg:
            return 0
        # intenta numérico
        try:
            v = float(x)
            return 1 if v >= 0.5 else 0
        except:
            return 0  # por defecto

    out = s.map(map_one).astype("int8")
    return out

def _finalize_dataset(df):
    """Asegura dtypes y target binario."""
    out = df.copy()
    # target binario
    out["target"] = _coerce_binary_target(out["target"])
    # fuerza numéricos a float32/int16 donde aplique (opcional)
    numeric_cols = [
        "age","pregnancies","glucose","blood_pressure","skin_thickness",
        "insulin","bmi","diabetes_pedigree","hypertension","heart_disease",
        "hba1c_level","blood_glucose_level",
        "gender_Female","gender_Male",
        "smoking_history_current","smoking_history_former","smoking_history_never",
        "smoking_history_ever","smoking_history_not current"
    ]
    for c in numeric_cols:
        if c in out.columns:
            # binarios → int8, el resto → float32
            if c.startswith("gender_") or c.startswith("smoking_history_") or c in ["hypertension","heart_disease","pregnancies"]:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("int8")
            else:
                out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype("float32")
    return out


# =========================================================
# 4. MAIN
# =========================================================
def main():
    loaded = load_csvs()
    normalized = []

    for name, df in loaded:
        if is_pima_like(df):
            print(f"➡️ {name} es PIMA")
            nd = normalize_pima(df)
        elif is_prediction_like(df):
            print(f"➡️ {name} es PREDICTION")
            nd = normalize_prediction(df)
        elif is_lab_diabetes_like(df):
            print(f"➡️ {name} es LAB-DIABETES")
            nd = normalize_lab_diabetes(df)
        elif is_mexican_hypertension(df):
            print(f"➡️ {name} es HIPERTENSION-MX")
            nd = normalize_mexican_hypertension(df)
        elif is_english_hypertension(df):
            print(f"➡️ {name} es HIPERTENSION-EN")
            nd = normalize_english_hypertension(df)
        elif is_obesity_uci(df):
            print(f"➡️ {name} es OBESIDAD-UCI")
            nd = normalize_obesity_uci(df)
        elif is_cardio_kaggle(df):
            print(f"➡️ {name} es CARDIO-KAGGLE")
            nd = normalize_cardio_kaggle(df)
        else:
            print(f"➡️ {name} es DESCONOCIDO")
            nd = normalize_unknown(df)

        normalized.append(nd)

    # unimos todo
    full_df = pd.concat(normalized, ignore_index=True)

    # aseguramos columnas
    for col in COMMON_SCHEMA:
        if col not in full_df.columns:
            full_df[col] = 0

    # =====================================================
    # 5. GENERAR LOS 4 DATASETS (VERSIÓN FINAL ÚNICA)
    # =====================================================

    # 1) Diabetes — deja target como venga en cada fuente (tras normalización)
    diabetes_df = _finalize_dataset(full_df.copy())
    diabetes_df.to_csv(os.path.join(PROCESSED_DIR, "diabetes_dataset.csv"), index=False)

    # 2) Hipertensión — target = hypertension (binario)
    hipert_df = full_df.copy()
    hipert_df["target"] = (pd.to_numeric(hipert_df["hypertension"], errors="coerce").fillna(0) == 1).astype(int)
    hipert_df = _finalize_dataset(hipert_df)
    hipert_df.to_csv(os.path.join(PROCESSED_DIR, "hipertension_dataset.csv"), index=False)

    # 3) Obesidad — definir SIEMPRE por BMI>=30 (independiente del target previo)
    obes_df = full_df.copy()
    obes_df["target"] = (pd.to_numeric(obes_df["bmi"], errors="coerce") >= 30).astype(int)
    obes_df = _finalize_dataset(obes_df)
    obes_df.to_csv(os.path.join(PROCESSED_DIR, "obesidad_dataset.csv"), index=False)


    # 4) Cardiovascular — preferir heart_disease==1 si existe; si no, conservar el target que ya trajo la fuente (p.ej., cardio_train)
    card_df = full_df.copy()
    if "heart_disease" in card_df.columns:
        hd = pd.to_numeric(card_df["heart_disease"], errors="coerce").fillna(0).astype(int)
        # si heart_disease==1, entonces target=1; si no, se queda el target ya existente
        card_df["target"] = np.where(hd == 1, 1, card_df.get("target", 0))
    card_df = _finalize_dataset(card_df)
    card_df.to_csv(os.path.join(PROCESSED_DIR, "cardiovascular_dataset.csv"), index=False)

    print("✅ Datasets procesados y guardados en", PROCESSED_DIR)


if __name__ == "__main__":
    main()
