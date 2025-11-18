#Script temporal, borrarlo luego de usarlo.
#Verifica que los datasets se hayan curado, se deben haber creado los 
# train (datos de entrenamiento (80%)),
# test (datos de validación (20%)), 
# data_dictionary (diccionario de variables (metadatos)), y 
# synthetic_tvae (datos sintéticos generados por SDV/TVAE)

import os, pandas as pd

BASE = "data_curated"
pairs = {
    "diabetes":       ("diabetes_train.csv",       "diabetes_test.csv",       "diabetes_synthetic_tvae_x0.5_seed42.csv"),
    "hipertension":   ("hipertension_train.csv",   "hipertension_test.csv",   "hipertension_synthetic_tvae_x0.5_seed42.csv"),
    "obesidad":       ("obesidad_train.csv",       "obesidad_test.csv",       "obesidad_synthetic_tvae_x0.5_seed42.csv"),
    "cardiovascular": ("cardiovascular_train.csv", "cardiovascular_test.csv", "cardiovascular_synthetic_tvae_x0.5_seed42.csv"),
}

def counts(df, name):
    c = df["target"].value_counts(dropna=False).to_dict()
    return f"{name:8} filas={len(df):7}  0={c.get(0,0):7}  1={c.get(1,0):7}"

for dis, (train_f, test_f, synth_f) in pairs.items():
    ddir = os.path.join(BASE, dis)
    print(f"\n=== {dis.upper()} ===")
    train_p = os.path.join(ddir, train_f)
    test_p  = os.path.join(ddir, test_f)
    synth_p = os.path.join(ddir, synth_f)

    for name, path in [("train",train_p),("test",test_p),("dictionary.csv",os.path.join(ddir,f"data_dictionary_{dis}.csv")),("synthetic",synth_p)]:
        print(f" - {name:12}: {'OK -> '+path if os.path.exists(path) else 'NO'}")

    # Cargar y validar columnas
    train = pd.read_csv(train_p)
    test  = pd.read_csv(test_p)
    print("   ", counts(train,"train"))
    print("   ", counts(test, "test"))

    tr_cols = list(train.columns)
    te_cols = list(test.columns)
    assert tr_cols == te_cols, f"[{dis}] Train/Test tienen columnas distintas"

    if os.path.exists(synth_p):
        synth = pd.read_csv(synth_p)
        print("   ", counts(synth,"synth"))
        missing = set(train.columns) - set(synth.columns)
        extra   = set(synth.columns) - set(train.columns)
        print(f"   cols diff -> faltan_en_synth={len(missing)} extra_en_synth={len(extra)}")
