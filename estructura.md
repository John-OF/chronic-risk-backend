backend/
│
├── app.py                    # API principal Flask
├── requirements.txt          # Dependencias
├── README.md                 # Documentación
│
├── data_raw/                 # Datasets crudos (subidos al repo)
├── data_processed/           # Datos limpios/procesados
├── data_curated/             # Datos finales listos para entrenamiento
│
├── models/                   # Modelos entrenados (pkl/joblib) usados por la API
│
├── prepare_datasets.py       # Limpieza y preparación de datos
├── curate_and_synthesize.py  # Generación de datos sintéticos (GAN/TVAE)
├── train_models.py           # Entrenamiento de modelos ML
├── check_processed.py        # Verificación rápida de datos procesados
├── tmp_check_targets.py      # Utilidades de depuración
└── tmp_verify_curated.py     # Utilidades de verificación
