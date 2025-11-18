import pandas as pd
import os

PROCESSED_DIR = "data_processed"

for name in ["diabetes_dataset", "hipertension_dataset", "obesidad_dataset", "cardiovascular_dataset"]:
    path = os.path.join(PROCESSED_DIR, f"{name}.csv")
    df = pd.read_csv(path, low_memory=False)
    # asegura target int
    df["target"] = pd.to_numeric(df["target"], errors="coerce").fillna(0).astype("int8")
    print(f"\n=== {name.upper()} ===")
    print("Filas:", len(df))
    print("Targets 0/1:")
    print(df["target"].value_counts(dropna=False))
    print(df.head(3))
