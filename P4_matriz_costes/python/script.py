import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score
)
from sklearn.model_selection import train_test_split


# ============================================================
# Configuración general
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.30

# Costes del problema
COSTE_FP = 10
COSTE_FN = 200

# Umbrales a estudiar
THRESHOLDS = np.arange(0.0, 1.01, 0.05)

# Rutas
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR.parent / "data" / "fraude_transacciones.csv"
OUTPUT_DIR = BASE_DIR.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# Funciones auxiliares
# ============================================================

def cargar_datos(path):
    df = pd.read_csv(path)
    X = df.drop(columns=["fraude"])
    y = df["fraude"]
    return X, y


def entrenar_modelo(X_train, y_train):
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    return model


def predecir_con_umbral(y_prob, threshold):
    return (y_prob >= threshold).astype(int)


def obtener_metricas(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return tn, fp, fn, tp


def calcular_coste(fp, fn):
    return fp * COSTE_FP + fn * COSTE_FN


def construir_tabla_umbral(y_true, y_prob, thresholds):
    resultados = []
    for t in thresholds:
        y_pred = predecir_con_umbral(y_prob, t)
        tn, fp, fn, tp = obtener_metricas(y_true, y_pred)
        coste = calcular_coste(fp, fn)
        resultados.append({
            "threshold": t,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "coste_total": coste
        })
    return pd.DataFrame(resultados)

# ============================================================
# Gráficas
# ============================================================

def graficar_matriz_confusion(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(5, 5))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title("Matriz de confusión (umbral = 0.5)")
    plt.colorbar()

    clases = ["0", "1"]
    tick_marks = np.arange(len(clases))
    plt.xticks(tick_marks, clases)
    plt.yticks(tick_marks, clases)

    # Mostrar los valores dentro de las celdas
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=14
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def graficar_coste_vs_umbral(df, path):
    best_idx = df["coste_total"].idxmin()
    best_row = df.loc[best_idx]

    plt.figure(figsize=(8, 5))
    plt.plot(df["threshold"], df["coste_total"], marker="o")
    plt.scatter(best_row["threshold"], best_row["coste_total"], s=80, color="red")
    plt.xlabel("Umbral")
    plt.ylabel("Coste total (€)")
    plt.title("Coste Total vs Umbral")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def graficar_roc(y_true, y_prob, path):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Curva ROC")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def graficar_pr(y_true, y_prob, path):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap = average_precision_score(y_true, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(recall, precision, label=f"AP = {ap:.4f}")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Curva Precision-Recall")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ============================================================
# Programa principal
# ============================================================

def main():
    print("Cargando datos...")
    X, y = cargar_datos(DATA_PATH)

    print("Separando entrenamiento y prueba...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print("Entrenando modelo...")
    model = entrenar_modelo(X_train, y_train)

    print("Obteniendo probabilidades...")
    y_prob = model.predict_proba(X_test)[:, 1]

    # Matriz de confusión con umbral 0.5
    print("Generando matriz de confusión...")
    y_pred_default = predecir_con_umbral(y_prob, 0.5)
    graficar_matriz_confusion(
        y_test, y_pred_default, OUTPUT_DIR / "matriz_confusion_0_5.png"
    )

    # Tabla de umbrales
    print("Analizando umbrales...")
    df = construir_tabla_umbral(y_test, y_prob, THRESHOLDS)
    df.to_csv(OUTPUT_DIR / "tabla_umbral.csv", index=False)

    # Gráfica coste vs umbral
    graficar_coste_vs_umbral(df, OUTPUT_DIR / "coste_vs_umbral.png")

    # Curvas ROC y PR
    print("Generando curva ROC...")
    graficar_roc(y_test, y_prob, OUTPUT_DIR / "curva_roc.png")

    print("Generando curva Precision-Recall...")
    graficar_pr(y_test, y_prob, OUTPUT_DIR / "curva_pr.png")

    print("\nProceso completado. Gráficas y tabla guardadas en /outputs/")


if __name__ == "__main__":
    main()
