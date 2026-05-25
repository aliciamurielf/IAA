"""
Práctica 9 - Few-shot Learning con MNIST
Código base para alumnos

Objetivo:
    Simular un escenario few-shot en el que el modelo se entrena sin ver
    la clase 7 y posteriormente intenta reconocerla usando pocos ejemplos
    mediante prototipos en un espacio de embeddings.
"""

import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
import tensorflow as tf
from tensorflow.keras import layers, models


SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


@dataclass
class FewShotData:
    X_train_known: np.ndarray
    y_train_known: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    known_classes: np.ndarray
    novel_class: int


def load_mnist_world_without_sevens() -> FewShotData:
    """
    Load MNIST and remove digit 7 from the training set.

    Returns
    -------
    FewShotData
        Object containing the known-class training set and the full test set.
    """
    # TODO 1: cargar MNIST con tf.keras.datasets.mnist.load_data()
    # TODO 2: normalizar imágenes a [0, 1]
    # TODO 3: añadir canal final para que las imágenes tengan forma (28, 28, 1)
    # TODO 4: eliminar las imágenes del dígito 7 del entrenamiento
    # TODO 5: devolver un objeto FewShotData
    raise NotImplementedError


def build_classifier(num_classes: int) -> tf.keras.Model:
    """
    Build a simple CNN classifier.

    Parameters
    ----------
    num_classes : int
        Number of known classes used during initial training.

    Returns
    -------
    tf.keras.Model
        Compiled CNN classifier.
    """
    # TODO: crear una CNN sencilla con una capa Dense llamada "embedding"
    # Pista:
    # - Conv2D + MaxPooling2D
    # - Conv2D + MaxPooling2D
    # - Flatten
    # - Dense(64, activation="relu", name="embedding")
    # - Dense(num_classes, activation="softmax")
    raise NotImplementedError


def remap_known_labels(y: np.ndarray, known_classes: np.ndarray) -> np.ndarray:
    """
    Remap labels from their original MNIST value to 0..num_known_classes-1.

    Parameters
    ----------
    y : np.ndarray
        Original labels.
    known_classes : np.ndarray
        Known classes in sorted order.

    Returns
    -------
    np.ndarray
        Remapped labels.
    """
    # TODO: crear un diccionario {clase_original: indice} y aplicarlo a y
    raise NotImplementedError


def create_feature_extractor(classifier: tf.keras.Model) -> tf.keras.Model:
    """
    Create a feature extractor from the trained classifier.

    Parameters
    ----------
    classifier : tf.keras.Model
        Trained CNN classifier.

    Returns
    -------
    tf.keras.Model
        Model that outputs the embedding layer.
    """
    # TODO: devolver un modelo con la misma entrada que classifier y salida la capa "embedding"
    raise NotImplementedError


def sample_support_set(X: np.ndarray, y: np.ndarray, class_label: int, n_shots: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample n_shots examples from one class.

    Parameters
    ----------
    X : np.ndarray
        Image array.
    y : np.ndarray
        Label array.
    class_label : int
        Class to sample.
    n_shots : int
        Number of support examples.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Support images and support labels.
    """
    # TODO: seleccionar aleatoriamente n_shots índices de la clase indicada
    raise NotImplementedError


def compute_prototypes(feature_extractor: tf.keras.Model, X_support: np.ndarray, y_support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute one prototype per class.

    Parameters
    ----------
    feature_extractor : tf.keras.Model
        Model that maps images to embeddings.
    X_support : np.ndarray
        Support images.
    y_support : np.ndarray
        Support labels.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Prototype vectors and their class labels.
    """
    # TODO 1: obtener embeddings del support set
    # TODO 2: para cada clase, calcular la media de sus embeddings
    # TODO 3: devolver matriz de prototipos y vector de etiquetas
    raise NotImplementedError


def classify_by_nearest_prototype(feature_extractor: tf.keras.Model, X_query: np.ndarray, prototypes: np.ndarray, prototype_labels: np.ndarray) -> np.ndarray:
    """
    Classify query images by nearest prototype.

    Parameters
    ----------
    feature_extractor : tf.keras.Model
        Model that maps images to embeddings.
    X_query : np.ndarray
        Query images.
    prototypes : np.ndarray
        Prototype matrix.
    prototype_labels : np.ndarray
        Labels associated with prototypes.

    Returns
    -------
    np.ndarray
        Predicted labels.
    """
    # TODO 1: obtener embeddings de query
    # TODO 2: calcular distancias euclídeas a todos los prototipos
    # TODO 3: asignar la etiqueta del prototipo más cercano
    raise NotImplementedError


def build_fewshot_episode(data: FewShotData, n_shots: int, n_query_per_class: int = 100):
    """
    Build a few-shot episode using all MNIST classes, including digit 7.

    Parameters
    ----------
    data : FewShotData
        Dataset object.
    n_shots : int
        Number of support examples per class.
    n_query_per_class : int
        Number of query examples per class.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        X_support, y_support, X_query, y_query.
    """
    # TODO:
    # - construir un support set con n_shots ejemplos por clase
    # - para clases conocidas, puedes tomar ejemplos del entrenamiento conocido
    # - para la clase 7, toma ejemplos del test o de una reserva separada
    # - construir un query set equilibrado con n_query_per_class por clase desde test
    raise NotImplementedError


def plot_accuracy_comparison(results: dict[str, float], output_path: str = "fewshot_accuracy_comparison.png") -> None:
    """
    Plot 1-shot vs 5-shot accuracy.
    """
    plt.figure(figsize=(6, 4))
    plt.bar(list(results.keys()), list(results.values()))
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Few-shot classification: 1-shot vs 5-shot")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()


def plot_embeddings_pca(feature_extractor: tf.keras.Model, X: np.ndarray, y: np.ndarray, output_path: str = "fewshot_embeddings_pca.png") -> None:
    """
    Visualize embeddings using PCA.
    """
    # TODO: obtener embeddings, aplicar PCA a 2D y representar por clase
    raise NotImplementedError


def main() -> None:
    data = load_mnist_world_without_sevens()

    # Entrenamiento inicial del clasificador sin la clase 7
    y_train_known_remap = remap_known_labels(data.y_train_known, data.known_classes)
    classifier = build_classifier(num_classes=len(data.known_classes))

    classifier.fit(
        data.X_train_known,
        y_train_known_remap,
        validation_split=0.1,
        epochs=3,
        batch_size=128,
        verbose=1,
    )

    feature_extractor = create_feature_extractor(classifier)

    results = {}
    for n_shots in [1, 5]:
        X_support, y_support, X_query, y_query = build_fewshot_episode(data, n_shots=n_shots)
        prototypes, prototype_labels = compute_prototypes(feature_extractor, X_support, y_support)
        y_pred = classify_by_nearest_prototype(feature_extractor, X_query, prototypes, prototype_labels)
        acc = accuracy_score(y_query, y_pred)
        results[f"{n_shots}-shot"] = acc
        print(f"{n_shots}-shot accuracy: {acc:.4f}")

    plot_accuracy_comparison(results)

    # Visualización opcional
    # plot_embeddings_pca(feature_extractor, X_query, y_query)


if __name__ == "__main__":
    main()
