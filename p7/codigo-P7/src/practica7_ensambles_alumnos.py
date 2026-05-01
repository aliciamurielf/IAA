"""
PRÁCTICA 7: La Inteligencia Colectiva (Modelos de Ensambles)
Asignatura: Introducción al Aprendizaje Automático (IAA) 26/27

Código base para el alumnado.

Objetivo del script
-------------------
Este archivo proporciona una estructura guiada para que el alumnado implemente
la práctica paso a paso. No es una solución completa: contiene secciones TODO
que deben completarse y analizarse en el informe final.

Dataset utilizado
-----------------
Breast Cancer Wisconsin (incluido en scikit-learn).

Modelos a trabajar
------------------
1. Árbol de decisión simple
2. Random Forest
3. Gradient Boosting (o AdaBoost, si se quiere probar como alternativa)

Recomendación
-------------
Lee cada comentario antes de programar. No se trata solo de obtener métricas,
sino de interpretar los resultados y justificar lo que observas.
"""

from __future__ import annotations

import os
import time
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree


RANDOM_STATE = 42
TEST_SIZE = 0.25


# -----------------------------------------------------------------------------
# 1. CARGA Y PREPARACIÓN DE LOS DATOS
# -----------------------------------------------------------------------------
def load_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """
    Carga el dataset Breast Cancer Wisconsin desde scikit-learn.
    """
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    return X, y


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Divide el dataset en entrenamiento y prueba.
    """
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


# -----------------------------------------------------------------------------
# 2. TAREA 1: ÁRBOL SIMPLE (BASELINE)
# -----------------------------------------------------------------------------
def train_decision_tree(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> DecisionTreeClassifier:
    """
    Entrena un árbol de decisión sin limitar la profundidad.
    """
    model = DecisionTreeClassifier(random_state=random_state)
    model.fit(X_train, y_train)
    return model


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> Dict[str, float]:
    """
    Evalúa un modelo en entrenamiento y en prueba.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    return {
        "train_accuracy": train_accuracy,
        "test_accuracy": test_accuracy,
    }


# -----------------------------------------------------------------------------
# 3. TAREA 2: RANDOM FOREST
# -----------------------------------------------------------------------------
def random_forest_experiment(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    n_trees_list: List[int],
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Ejecuta varios experimentos cambiando el número de árboles.
    """
    rows = []

    for n_trees in n_trees_list:
        start = time.perf_counter()

        model = RandomForestClassifier(
            n_estimators=n_trees,
            random_state=random_state,
        )
        model.fit(X_train, y_train)

        elapsed = time.perf_counter() - start
        y_test_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_test_pred)

        rows.append(
            {
                "n_estimators": n_trees,
                "test_accuracy": test_accuracy,
                "train_time_seconds": elapsed,
            }
        )

    return pd.DataFrame(rows)


def plot_random_forest_results(results_df: pd.DataFrame, filename: str) -> None:
    """
    Representa la evolución del accuracy en test frente al número de árboles y la guarda.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(
        results_df["n_estimators"],
        results_df["test_accuracy"],
        marker="o",
        linestyle="--",
    )
    plt.title("Random Forest: Accuracy en test vs número de árboles")
    plt.xlabel("n_estimators")
    plt.ylabel("Accuracy en test")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_and_save_tree(model, feature_names, filename):
    """
    Dibuja y guarda el árbol de decisión. Limitamos la profundidad a 3 
    solo para la visualización.
    """
    plt.figure(figsize=(20, 10))
    plot_tree(model, 
              feature_names=feature_names, 
              class_names=["Malignant", "Benign"], 
              filled=True, 
              rounded=True, 
              max_depth=3, 
              fontsize=10)
    plt.title("Visualización del Árbol de Decisión (profundidad truncada)")
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def plot_feature_importances(model, feature_names, title, filename):
    """
    Dibuja un gráfico de barras con las variables más importantes y lo guarda.
    """
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances_top = importances.sort_values(ascending=False).head(10) 
    
    plt.figure(figsize=(8, 5))
    importances_top.plot.bar(color='teal')
    plt.title(title)
    plt.ylabel("Nivel de Importancia")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()


# -----------------------------------------------------------------------------
# 4. TAREA 3: BOOSTING
# -----------------------------------------------------------------------------
def train_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> Tuple[GradientBoostingClassifier, float]:
    """
    Entrena un modelo Gradient Boosting y mide el tiempo de entrenamiento.
    """
    start = time.perf_counter()

    model = GradientBoostingClassifier(random_state=random_state)
    model.fit(X_train, y_train)

    elapsed = time.perf_counter() - start
    return model, elapsed


# -----------------------------------------------------------------------------
# 5. IMPORTANCIA DE VARIABLES
# -----------------------------------------------------------------------------
def show_top_features(model, feature_names: List[str], top_k: int = 3) -> pd.DataFrame:
    """
    Muestra las variables más importantes según el modelo.
    """
    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    )
    importances = importances.sort_values("importance", ascending=False)
    return importances.head(top_k)


# -----------------------------------------------------------------------------
# 6. UTILIDADES DE IMPRESIÓN
# -----------------------------------------------------------------------------
def print_confusion_and_report(model, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    """
    Imprime matriz de confusión e informe de clasificación.
    """
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("\nMatriz de confusión:")
    print(cm)
    print("\nClassification report:")
    print(report)


# -----------------------------------------------------------------------------
# 7. PROGRAMA PRINCIPAL
# -----------------------------------------------------------------------------
def main() -> None:
    # --- CREAR CARPETA OUTPUT ---
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    # -------------------------------------------------------------------------
    # PASO 1. CARGAR Y DIVIDIR LOS DATOS
    # -------------------------------------------------------------------------
    X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    print("=" * 80)
    print("PRÁCTICA 7 - MODELOS DE ENSAMBLES")
    print("=" * 80)
    print(f"Número de muestras totales: {len(X)}")
    print(f"Número de variables: {X.shape[1]}")
    print(f"Muestras de entrenamiento: {len(X_train)}")
    print(f"Muestras de prueba: {len(X_test)}")

    # -------------------------------------------------------------------------
    # PASO 2. ÁRBOL SIMPLE
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TAREA 1 - ÁRBOL SIMPLE")
    print("-" * 80)

    tree_model = train_decision_tree(X_train, y_train)
    tree_results = evaluate_model(tree_model, X_train, y_train, X_test, y_test)

    print(f"Accuracy en entrenamiento: {tree_results['train_accuracy']:.4f}")
    print(f"Accuracy en test:          {tree_results['test_accuracy']:.4f}")

    print_confusion_and_report(tree_model, X_test, y_test)
    
    # Guardamos en la carpeta output
    plot_and_save_tree(tree_model, X.columns, os.path.join(output_dir, "arbol_decision.png"))

    # -------------------------------------------------------------------------
    # PASO 3. RANDOM FOREST
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TAREA 2 - RANDOM FOREST")
    print("-" * 80)

    n_trees_list = [1, 10, 50, 100]
    rf_results = random_forest_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        n_trees_list=n_trees_list,
    )

    print("\nResultados Random Forest:")
    print(rf_results)

    # Guardamos en la carpeta output
    plot_random_forest_results(rf_results, os.path.join(output_dir, "rf_evolucion_arboles.png"))

    # Entrenamos un modelo final con 100 árboles para analizarlo mejor.
    rf_model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    rf_model.fit(X_train, y_train)

    # Guardamos en la carpeta output
    plot_feature_importances(rf_model, X.columns, "Importancia Variables - Random Forest", os.path.join(output_dir, "rf_importancias.png"))
    
    print("\nEvaluación Random Forest (100 árboles):")
    print_confusion_and_report(rf_model, X_test, y_test)

    print("\nTop 3 variables más importantes en Random Forest:")
    print(show_top_features(rf_model, list(X.columns), top_k=3))

    subarbol_rf = rf_model.estimators_[0]

    # Usamos la función del profesor para dibujar este subárbol y guardarlo
    plot_and_save_tree(
        subarbol_rf, 
        X.columns, 
        os.path.join(output_dir, "subarbol_random_forest.png")
    )
    print("¡Gráfica del subárbol extraída y guardada en la carpeta output!")

    # -------------------------------------------------------------------------
    # PASO 4. BOOSTING
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TAREA 3 - BOOSTING")
    print("-" * 80)

    gb_model, gb_train_time = train_gradient_boosting(X_train, y_train)
    gb_results = evaluate_model(gb_model, X_train, y_train, X_test, y_test)

    print(f"Tiempo de entrenamiento:   {gb_train_time:.4f} s")
    print(f"Accuracy en entrenamiento: {gb_results['train_accuracy']:.4f}")
    print(f"Accuracy en test:          {gb_results['test_accuracy']:.4f}")

    print_confusion_and_report(gb_model, X_test, y_test)

    print("\nTop 3 variables más importantes en Gradient Boosting:")
    print(show_top_features(gb_model, list(X.columns), top_k=3))

    # Guardamos en la carpeta output
    plot_feature_importances(gb_model, X.columns, "Importancia Variables - Boosting", os.path.join(output_dir, "gb_importancias.png"))

    # -------------------------------------------------------------------------
    # PASO 5. TABLA COMPARATIVA FINAL
    # -------------------------------------------------------------------------
    print("\n" + "-" * 80)
    print("TABLA COMPARATIVA FINAL")
    print("-" * 80)

    comparison_df = pd.DataFrame(
        [
            {
                "Modelo": "Árbol simple",
                "Accuracy test": tree_results["test_accuracy"],
                "Ventajas": "Alta interpretabilidad, entrenamiento rápido, bajo consumo memoria",
                "Desventajas": "Alto sobreajuste, baja estabilidad, generalización pobre",
            },
            {
                "Modelo": "Random Forest",
                "Accuracy test": rf_model.score(X_test, y_test),
                "Ventajas": "Reduce varianza, estable y robusto, buen rendimiento, menor sobreajuste",
                "Desventajas": "Menos interpretable, entrenamiento más lento, mayor consumo memoria",
            },
            {
                "Modelo": "Gradient Boosting",
                "Accuracy test": gb_results["test_accuracy"],
                "Ventajas": "Mejor rendimiento predictivo, maneja relaciones complejas",
                "Desventajas": "Entrenamiento lento, riesgo de sobreajuste, baja interpretabilidad, alta complejidad",
            },
        ]
    )

    print(comparison_df)

if __name__ == "__main__":
    main()