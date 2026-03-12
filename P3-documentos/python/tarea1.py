from sklearn.model_selection import train_test_split
from utils import cargar_dataset
import pandas as pd

df, X, y = cargar_dataset()

resultados = []

for seed in range(5):
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed
    )
    positivos = sum(y_test == 1)
    resultados.append({"ejecucion": seed+1, "positivos_test": positivos})

pd.DataFrame(resultados).to_csv("../outputs/tarea1_resultados.csv", index=False)