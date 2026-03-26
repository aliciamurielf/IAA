from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, f1_score
from utils import cargar_dataset
import pandas as pd

df, X, y = cargar_dataset()

# Eliminar variable trampa
df = df.drop(columns=["ID_Hospital_Filtro"])
X = df.drop(columns=["Clase"]).values
y = df["Clase"].values

kf = KFold(n_splits=10, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def evaluar_modelo(modelo):
    acc_kf = cross_val_score(modelo, X, y, cv=kf, scoring="accuracy")
    acc_skf = cross_val_score(modelo, X, y, cv=skf, scoring="accuracy")
    f1_kf = cross_val_score(modelo, X, y, cv=kf, scoring=make_scorer(f1_score))
    f1_skf = cross_val_score(modelo, X, y, cv=skf, scoring=make_scorer(f1_score))

    return pd.DataFrame({
        "metodo": ["KFold", "StratifiedKFold"],
        "accuracy_media": [acc_kf.mean(), acc_skf.mean()],
        "accuracy_std": [acc_kf.std(), acc_skf.std()],
        "f1_media": [f1_kf.mean(), f1_skf.mean()],
        "f1_std": [f1_kf.std(), f1_skf.std()]
    })

# Evaluar ambos modelos
df_log = evaluar_modelo(LogisticRegression(max_iter=200))
df_rf  = evaluar_modelo(RandomForestClassifier(n_estimators=200, random_state=42))

df_log.to_csv("../outputs/tarea3_logistic.csv", index=False)
df_rf.to_csv("../outputs/tarea3_randomforest.csv", index=False)
