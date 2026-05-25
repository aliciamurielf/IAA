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
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
    # TODO 2: normalizar imágenes a [0, 1]
    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0
    # TODO 3: añadir canal final para que las imágenes tengan forma (28, 28, 1)
    X_train = np.expand_dims(X_train,axis=-1)
    X_test = np.expand_dims(X_test,axis=-1)
    # TODO 4: eliminar las imágenes del dígito 7 del entrenamiento
    mask = y_train != 7
    X_train = X_train[mask]
    y_train = y_train[mask]
    # TODO 5: devolver un objeto FewShotData
    known_classes = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9])
    novel_class = 7
    
    return FewShotData(
        X_train_known=X_train, 
        y_train_known=y_train, 
        X_test=X_test, 
        y_test=y_test, 
        known_classes=known_classes, 
        novel_class=novel_class
    )


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
    inputs = layers.Input(shape=(28, 28, 1))

    x = layers.Conv2D(32, (3, 3), activation="relu")(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    
    x = layers.Conv2D(64, (3, 3), activation="relu")(x)
    x = layers.MaxPooling2D((2, 2))(x)
    
    # Pista: Flatten
    x = layers.Flatten()(x)
    
    # Pista: Dense(64, activation="relu", name="embedding")
    embedding = layers.Dense(64, activation="relu", name="embedding")(x)
    
    # Pista: Dense(num_classes, activation="softmax")
    outputs = layers.Dense(num_classes, activation="softmax")(embedding)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

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
    mapping = {original_class: idx for idx, original_class in enumerate(known_classes)}
    return np.vectorize(mapping.get)(y)

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
    return tf.keras.Model(inputs=classifier.input, outputs=classifier.get_layer("embedding").output)


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
    indices = np.where(y == class_label)[0]
    sampled_indices = np.random.choice(indices, size=n_shots, replace=False)
    
    return X[sampled_indices], y[sampled_indices]


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
    embeddings = feature_extractor.predict(X_support, verbose=0)

    prototypes = []
    classes = np.unique(y_support)
    # TODO 2: para cada clase, calcular la media de sus embeddings
    for cls in classes:
        cls_mask = (y_support == cls)
        cls_embeddings = embeddings[cls_mask]
        cls_prototype = np.mean(cls_embeddings, axis=0)
        prototypes.append(cls_prototype)
    # TODO 3: devolver matriz de prototipos y vector de etiquetas
    return np.array(prototypes), classes


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

    q_emb = feature_extractor.predict(X_query, verbose=0)
    
    diff = q_emb[:, None, :] - prototypes[None, :, :]
    distances = np.linalg.norm(diff, axis=2)
    
    predicted_indices = np.argmin(distances, axis=1)
    
    predicted_labels = prototype_labels[predicted_indices]
    return predicted_labels.flatten()


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
    all_classes = np.append(data.known_classes, data.novel_class)
    
    X_support_list, y_support_list = [], []
    X_query_list, y_query_list = [], []
    
    for cls in all_classes:
        # Extraer Support Set
        if cls == data.novel_class:
            # Para la clase 7, extraemos del test set
            X_sup, y_sup = sample_support_set(data.X_test, data.y_test, cls, n_shots)
        else:
            # Para las demás, extraemos del train set conocido
            X_sup, y_sup = sample_support_set(data.X_train_known, data.y_train_known, cls, n_shots)
            
        X_support_list.append(X_sup)
        y_support_list.append(y_sup)
        
        # Extraer Query Set (Siempre del test set para evaluación justa)
        indices_test = np.where(data.y_test == cls)[0]
        np.random.shuffle(indices_test)
        
        # Evitar sobrelapamiento de datos si extrajimos el support del test
        query_indices = []
        for idx in indices_test:
            if len(query_indices) == n_query_per_class:
                break
            # Comprobación simple para evitar que un elemento de support acabe en query (solo aplicable a la clase nueva)
            if cls == data.novel_class:
                # Comprobación heurística para evitar solapamientos exactos (en un entorno puro se separarían los arrays antes)
                if not any(np.array_equal(data.X_test[idx], sup_img) for sup_img in X_sup):
                    query_indices.append(idx)
            else:
                query_indices.append(idx)
                
        X_query_list.append(data.X_test[query_indices])
        y_query_list.append(data.y_test[query_indices])

    return np.concatenate(X_support_list), np.concatenate(y_support_list), np.concatenate(X_query_list), np.concatenate(y_query_list)


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
    embeddings = feature_extractor.predict(X, verbose=0)
    pca = PCA(n_components=2, random_state=SEED)
    embeddings_2d = pca.fit_transform(embeddings)
    
    plt.figure(figsize=(10, 8))
    classes = np.unique(y)
    
    for cls in classes:
        mask = (y == cls)
        # Resaltamos el 7 con un marcador diferente y color fuerte
        if cls == 7:
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=f'Clase {cls} (Nueva)', alpha=0.9, s=50, marker='*')
        else:
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label=f'Clase {cls}', alpha=0.4, s=20)
            
    plt.title("Proyección 2D del espacio de características (PCA)")
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.show()


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
        print(f"\n--- Ejecutando Few-Shot con {n_shots}-shot ---")
        X_support, y_support, X_query, y_query = build_fewshot_episode(data, n_shots=n_shots)
        prototypes, prototype_labels = compute_prototypes(feature_extractor, X_support, y_support)
        y_pred = classify_by_nearest_prototype(feature_extractor, X_query, prototypes, prototype_labels)
        
        acc = accuracy_score(y_query, y_pred)
        results[f"{n_shots}-shot"] = acc
        print(f"{n_shots}-shot accuracy general: {acc:.4f}")
        
        # Calcular accuracy específica para la clase nueva (el 7)
        mask_7 = (y_query == 7)
        acc_7 = accuracy_score(y_query[mask_7], y_pred[mask_7])
        print(f"{n_shots}-shot accuracy en la clase nueva (7): {acc_7:.4f}")

    plot_accuracy_comparison(results)

    # Visualización opcional
    print("\nGenerando visualización PCA...")
    plot_embeddings_pca(feature_extractor, X_query, y_query)

if __name__ == "__main__":
    main()
