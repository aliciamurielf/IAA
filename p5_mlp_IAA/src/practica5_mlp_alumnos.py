"""Plantilla guiada para la Práctica 5: Arquitectura de Redes Neuronales (MLP).

El objetivo de este script es servir como punto de partida. Varias partes ya
están preparadas para ahorrar trabajo mecánico, pero la práctica NO está
cerrada: debéis completar el análisis, interpretar resultados y justificar las
arquitecturas probadas.
"""
from dataclasses import dataclass

from sklearn.datasets import load_digits, make_moons
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from utils_practica5 import (
    plot_confusion_matrix,
    plot_decision_boundary,
    plot_loss_curve,
    print_table,
)

RANDOM_STATE = 42
MOONS_NOISE = 0.25
TEST_SIZE_MOONS = 0.30
TEST_SIZE_DIGITS = 0.25
TARGET_ACCURACY = 0.95


@dataclass
class ExperimentResult:
    """Resultado resumido de un experimento con MLP.

    Parameters
    ----------
    name : str
        Nombre corto del experimento.
    hidden_layers : tuple[int, ...]
        Arquitectura de capas ocultas.
    activation : str
        Función de activación.
    acc_train : float
        Accuracy en entrenamiento.
    acc_test : float
        Accuracy en prueba.
    n_iter : int
        Número de iteraciones consumidas.
    final_loss : float
        Última pérdida de entrenamiento.
    """

    name: str
    hidden_layers: tuple[int, ...]
    activation: str
    acc_train: float
    acc_test: float
    n_iter: int
    final_loss: float


def build_mlp(hidden_layer_sizes: tuple[int, ...], activation: str = "relu", max_iter: int = 2500) -> Pipeline:
    """Construye un pipeline con escalado y clasificador MLP.

    Parameters
    ----------
    hidden_layer_sizes : tuple[int, ...]
        Arquitectura del MLP. Si es `()`, no hay capas ocultas.
    activation : str, optional
        Activación de las capas ocultas.
    max_iter : int, optional
        Número máximo de iteraciones.

    Returns
    -------
    Pipeline
        Modelo listo para entrenar.
    """
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=hidden_layer_sizes,
                    activation=activation,
                    solver="adam",
                    alpha=1e-4,
                    max_iter=max_iter,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_model(
    name: str,
    model: Pipeline,
    X_train,
    X_test,
    y_train,
    y_test,
    activation: str,
    hidden_layers: tuple[int, ...],
) -> ExperimentResult:
    """Entrena y resume un experimento."""
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    mlp = model.named_steps["mlp"]
    return ExperimentResult(
        name=name,
        hidden_layers=hidden_layers,
        activation=activation,
        acc_train=accuracy_score(y_train, y_pred_train),
        acc_test=accuracy_score(y_test, y_pred_test),
        n_iter=mlp.n_iter_,
        final_loss=mlp.loss_,
    )


def load_moons_dataset():
    """Genera el dataset no lineal de lunas y devuelve su partición."""
    X, y = make_moons(n_samples=500, noise=MOONS_NOISE, random_state=RANDOM_STATE)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE_MOONS,
        stratify=y,
        random_state=RANDOM_STATE,
    )
    return X, y, X_train, X_test, y_train, y_test


def task_1_simple_perceptron() -> None:
    """Tarea 1: comprobar el límite de un modelo sin capa oculta."""
    print("\n=== TAREA 1: El problema de la no linealidad ===")
    X, y, X_train, X_test, y_train, y_test = load_moons_dataset()

    model = build_mlp(hidden_layer_sizes=(), activation="relu", max_iter=2500)
    result = evaluate_model(
        name="Perceptrón simple",
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        activation="relu",
        hidden_layers=(),
    )

    print_table(
        [
            {
                "modelo": result.name,
                "arquitectura": result.hidden_layers,
                "activacion": result.activation,
                "acc_train": f"{result.acc_train:.4f}",
                "acc_test": f"{result.acc_test:.4f}",
                "iteraciones": result.n_iter,
            }
        ]
    )

    y_pred = model.predict(X_test)
    plot_decision_boundary(model, X, y, "Tarea 1 - Perceptrón simple", "tarea1_perceptron_simple.png")
    plot_confusion_matrix(y_test, y_pred, "Tarea 1 - Matriz de confusión", "tarea1_confusion.png")

    print("\nTODO ALUMNADO:")
    print("- Explicar por qué la frontera aprendida es lineal.")
    print("- Relacionar el fallo con la geometría del dataset.")
    print("- No limitarse a decir que la accuracy es baja: justificar la causa.")


def task_2_hidden_layer() -> None:
    """Tarea 2: comparar el efecto del número de neuronas ocultas."""
    print("\n=== TAREA 2: Diseñando la capa oculta ===")
    X, y, X_train, X_test, y_train, y_test = load_moons_dataset()

    hidden_configs = [(2,), (5,), (20,)]
    rows = []
    for hidden in hidden_configs:
        model = build_mlp(hidden_layer_sizes=hidden, activation="relu", max_iter=3000)
        result = evaluate_model(
            name=f"MLP{hidden}",
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            activation="relu",
            hidden_layers=hidden,
        )
        rows.append(
            {
                "modelo": result.name,
                "arquitectura": result.hidden_layers,
                "acc_train": f"{result.acc_train:.4f}",
                "acc_test": f"{result.acc_test:.4f}",
                "iteraciones": result.n_iter,
            }
        )
        filename = f"tarea2_hidden_{'_'.join(map(str, hidden))}.png"
        plot_decision_boundary(model, X, y, f"Tarea 2 - Una capa oculta {hidden}", filename)

    print_table(rows, csv_name="tarea2_hidden_layer.csv")

    print("\nTODO ALUMNADO:")
    print("- Comparar visualmente las tres fronteras de decisión.")
    print("- Indicar dónde parece haber underfitting.")
    print("- Discutir si alguna frontera parece demasiado compleja.")
    print("- Probar, si queréis, alguna configuración extra y justificarla.")


def task_3_activation_comparison() -> None:
    """Tarea 3: comparar Sigmoide y ReLU con la MISMA arquitectura.

    Nota importante
    ---------------
    Para que la comparación sea justa, aquí NO cambiamos la arquitectura entre
    activaciones. Solo cambia la activación.
    """
    print("\n=== TAREA 3: Funciones de activación ===")
    X, y, X_train, X_test, y_train, y_test = load_moons_dataset()

    fixed_architecture = (20,)
    activations = ["logistic", "relu"]

    rows = []
    for activation in activations:
        model = build_mlp(hidden_layer_sizes=fixed_architecture, activation=activation, max_iter=3000)
        result = evaluate_model(
            name=f"MLP{fixed_architecture}-{activation}",
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            activation=activation,
            hidden_layers=fixed_architecture,
        )
        rows.append(
            {
                "modelo": result.name,
                "arquitectura": result.hidden_layers,
                "activacion": result.activation,
                "acc_test": f"{result.acc_test:.4f}",
                "iteraciones": result.n_iter,
                "loss_final": f"{result.final_loss:.4f}",
            }
        )

        plot_decision_boundary(
            model,
            X,
            y,
            f"Tarea 3 - Activación: {activation}",
            f"tarea3_activation_{activation}.png",
        )
        plot_loss_curve(
            model.named_steps["mlp"].loss_curve_,
            f"Curva de pérdida - {activation}",
            f"tarea3_loss_curve_{activation}.png",
        )

    print_table(rows, csv_name="tarea3_activation_comparison.csv")

    print("\nTODO ALUMNADO:")
    print("- Comparar convergencia y estabilidad entre Sigmoide y ReLU.")
    print("- Explicar por qué la comparación debe hacerse con la misma arquitectura.")
    print("- Relacionar lo observado con saturación, facilidad de optimización y tipo de frontera.")


def task_4_digits() -> None:
    """Tarea 4: explorar arquitecturas en el dataset de dígitos."""
    print("\n=== TAREA 4: El reto de la caja negra ===")
    digits = load_digits()
    X_train, X_test, y_train, y_test = train_test_split(
        digits.data,
        digits.target,
        test_size=TEST_SIZE_DIGITS,
        stratify=digits.target,
        random_state=RANDOM_STATE,
    )

    candidate_architectures = [
        (20,),
        (50,),
        (100,),
        (30, 15),
        (50, 20),
        (64, 32),
    ]

    rows = []
    for arch in candidate_architectures:
        model = build_mlp(hidden_layer_sizes=arch, activation="relu", max_iter=2500)
        result = evaluate_model(
            name=f"MLP{arch}",
            model=model,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            activation="relu",
            hidden_layers=arch,
        )
        rows.append(
            {
                "modelo": result.name,
                "arquitectura": result.hidden_layers,
                "acc_train": f"{result.acc_train:.4f}",
                "acc_test": f"{result.acc_test:.4f}",
                "iteraciones": result.n_iter,
                "cumple_95": "sí" if result.acc_test >= TARGET_ACCURACY else "no",
            }
        )

    table = print_table(rows, csv_name="tarea4_digits.csv")
    reached = table[table["cumple_95"] == "sí"]
    if reached.empty:
        print("\nNinguna arquitectura de la lista inicial alcanza el 95%.")
        print("Debéis ampliar la búsqueda con criterio.")
    else:
        print("\nArquitecturas de la lista inicial que alcanzan el 95%:")
        print(reached.to_string(index=False))

    print("\nTODO ALUMNADO:")
    print("- Justificar el proceso seguido para elegir arquitecturas.")
    print("- No quedarse solo con la mejor accuracy: buscar la mínima razonable que alcance 95%.")
    print("- Comparar redes anchas frente a redes más profundas.")
    print("- Redactar una decisión final de diseño.")


def main() -> None:
    """Ejecuta la plantilla completa."""
    task_1_simple_perceptron()
    task_2_hidden_layer()
    task_3_activation_comparison()
    task_4_digits()


if __name__ == "__main__":
    main()
