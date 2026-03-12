from sklearn.model_selection import StratifiedKFold
from utils import cargar_dataset
import pandas as pd

df, X, y = cargar_dataset()

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rows = []

for fold, (_, test_idx) in enumerate(skf.split(X, y), start=1):
    y_test = y[test_idx]
    positivos = sum(y_test == 1)
    proporcion = positivos / len(y_test)
    rows.append({
        "fold": fold,
        "positivos_test": positivos,
        "tamano_fold": len(y_test),
        "proporcion_clase1": proporcion
    })

pd.DataFrame(rows).to_csv("../resultados/tarea2_folds.csv", index=False)