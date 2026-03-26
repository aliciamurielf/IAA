from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, f1_score
from utils import cargar_dataset
import pandas as pd

df, X, y = cargar_dataset()

model = LogisticRegression(max_iter=200)

kf = KFold(n_splits=10, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

accuracy_kf = cross_val_score(model, X, y, cv=kf, scoring="accuracy")
accuracy_skf = cross_val_score(model, X, y, cv=skf, scoring="accuracy")

f1_kf = cross_val_score(model, X, y, cv=kf, scoring=make_scorer(f1_score))
f1_skf = cross_val_score(model, X, y, cv=skf, scoring=make_scorer(f1_score))

df_out = pd.DataFrame({
    "metodo": ["KFold", "StratifiedKFold"],
    "accuracy_media": [accuracy_kf.mean(), accuracy_skf.mean()],
    "accuracy_std": [accuracy_kf.std(), accuracy_skf.std()],
    "f1_media": [f1_kf.mean(), f1_skf.mean()],
    "f1_std": [f1_kf.std(), f1_skf.std()]
})

df_out.to_csv("../outputs/tarea3_metricas.csv", index=False)
