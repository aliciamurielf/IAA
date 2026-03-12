from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from utils import cargar_dataset
import pandas as pd

df, X, y = cargar_dataset()

# Detectar variable trampa por correlación
correlaciones = df.corr(numeric_only=True)["Clase"].sort_values(ascending=False)
print("Correlaciones con Clase:")
print(correlaciones)

variable_trampa = correlaciones.index[1] 
print("Variable trampa detectada:", variable_trampa)

# Con trampa
X_all = df.drop(columns=["Clase"]).values

# Sin trampa
X_clean = df.drop(columns=["Clase", variable_trampa]).values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
model = LogisticRegression(max_iter=200)

acc_all = cross_val_score(model, X_all, y, cv=skf, scoring="accuracy")
f1_all = cross_val_score(model, X_all, y, cv=skf, scoring=make_scorer(f1_score))

acc_clean = cross_val_score(model, X_clean, y, cv=skf, scoring="accuracy")
f1_clean = cross_val_score(model, X_clean, y, cv=skf, scoring=make_scorer(f1_score))

df_out = pd.DataFrame({
    "escenario": ["con_trampa", "sin_trampa"],
    "accuracy": [acc_all.mean(), acc_clean.mean()],
    "f1": [f1_all.mean(), f1_clean.mean()]
})

df_out.to_csv("../outputs/tarea4_comparativa.csv", index=False)
