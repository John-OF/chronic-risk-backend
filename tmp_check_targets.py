import pandas as pd
d = pd.read_csv("data_processed/diabetes_dataset.csv")
o = pd.read_csv("data_processed/obesidad_dataset.csv")
print("Iguales?", (d["target"].values == o["target"].values).all())
print("Coincidencias:", (d["target"].values == o["target"].values).sum(), "de", len(d))
print("\nCrosstab entre targets:")
print(pd.crosstab(d["target"], o["target"]))
