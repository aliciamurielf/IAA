import pandas as pd

def cargar_dataset():
    df = pd.read_csv("../data/pacientes_riesgo.csv")
    X = df.drop(columns=["Clase"]).values
    y = df["Clase"].values
    return df, X, y
    
