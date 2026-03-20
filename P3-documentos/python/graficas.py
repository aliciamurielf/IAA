import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


sns.set_style("whitegrid")

# ============================
# TAREA 1 — Gráfica positivos test
# ============================

df1 = pd.read_csv("../outputs/tarea1_resultados.csv")

plt.figure(figsize=(6,4))
plt.bar(df1["ejecucion"], df1["positivos_test"], color=sns.color_palette("Blues", 5))
plt.xlabel("Ejecución")
plt.ylabel("Positivos en test")
plt.title("Positivos en test en 5 particiones aleatorias")
plt.xticks(df1["ejecucion"])
plt.tight_layout()
plt.savefig("../outputs/tarea1_positivos_test_aleatorio.png")
plt.close()


# ============================
# TAREA 2 — Proporción por fold
# ============================

df2 = pd.read_csv("../outputs/tarea2_folds.csv")

plt.figure(figsize=(6,4))
plt.plot(df2["fold"], df2["proporcion_clase1"], marker="o", color=sns.color_palette("Blues", 8)[5])
plt.xlabel("Fold")
plt.ylabel("Proporción clase 1")
plt.title("Proporción de clase 1 por fold (StratifiedKFold)")
plt.ylim(0, 0.05)
plt.xticks(df2["fold"])
plt.tight_layout()
plt.savefig("../outputs/tarea2_proporcion_folds_estratificado.png")
plt.close()


# ============================
# TAREA 3 — Boxplots Accuracy y F1
# ============================

df3 = pd.read_csv("../outputs/tarea3_metricas.csv")

# ----------------------------
# 1. Accuracy — puntos con barras de error
# ----------------------------
plt.figure(figsize=(6,4))
plt.errorbar(
    df3["metodo"],
    df3["accuracy_media"],
    yerr=df3["accuracy_std"],
    fmt="o",
    capsize=6,
    markersize=9,
    linewidth=2,
    color=sns.color_palette("Blues", 5)[3],
    ecolor=sns.color_palette("Blues", 5)[1]
)
plt.ylabel("Accuracy")
plt.title("Accuracy: media y desviación típica")
plt.tight_layout()
plt.savefig("../outputs/tarea3_accuracy_errorpoints.png")
plt.close()

# ----------------------------
# 2. F1 — puntos con barras de error
# ----------------------------
plt.figure(figsize=(6,4))
plt.errorbar(
    df3["metodo"],
    df3["f1_media"],
    yerr=df3["f1_std"],
    fmt="o",
    capsize=6,
    markersize=9,
    linewidth=2,
    color=sns.color_palette("Blues", 5)[4],
    ecolor=sns.color_palette("Blues", 5)[2]
)
plt.ylabel("F1-score")
plt.title("F1-score: media y desviación típica")
plt.tight_layout()
plt.savefig("../outputs/tarea3_f1_errorpoints.png")
plt.close()

# ----------------------------
# 3. Gráfica de barras — desviaciones típicas
# ----------------------------
plt.figure(figsize=(6,4))
plt.bar(
    df3["metodo"],
    df3["accuracy_std"],
    color=sns.color_palette("Blues", 5)[2],
    alpha=0.8,
    label="Accuracy std"
)
plt.bar(
    df3["metodo"],
    df3["f1_std"],
    color=sns.color_palette("Blues", 5)[4],
    alpha=0.8,
    label="F1 std"
)
plt.ylabel("Desviación típica")
plt.title("Comparativa de desviación típica (Accuracy y F1)")
plt.legend()
plt.tight_layout()
plt.savefig("../outputs/tarea3_std_barras.png")
plt.close()
# ============================
# TAREA 4 — Leakage
# ============================

df4 = pd.read_csv("../outputs/tarea4_comparativa.csv")

# Accuracy con y sin trampa
plt.figure(figsize=(6,4))
plt.bar(df4["escenario"], df4["accuracy"], color=sns.color_palette("Blues", 3))
plt.ylabel("Accuracy")
plt.title("Accuracy con y sin variable trampa")
plt.tight_layout()
plt.savefig("../outputs/tarea4_accuracy_leakage.png")
plt.close()

# Correlaciones
df = pd.read_csv("../data/pacientes_riesgo.csv")
corr = df.corr(numeric_only=True)["Clase"].sort_values(ascending=False)

plt.figure(figsize=(7,4))
sns.barplot(
    x=corr.index,
    y=corr.values,
    palette="Blues"
)
plt.xticks(rotation=45)
plt.ylabel("Correlación con Clase")
plt.title("Correlaciones con la clase (detección de leakage)")
plt.tight_layout()
plt.savefig("../outputs/tarea4_correlaciones_leakage.png")
plt.close()

print("Todas las gráficas se han generado correctamente en tonos azules.")
