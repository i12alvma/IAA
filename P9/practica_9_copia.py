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
import os


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
    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    # TODO 4: eliminar las imágenes del dígito 7 del entrenamiento
    novel_class = 7
    known_mask = y_train != novel_class
    X_train_known = X_train[known_mask]
    y_train_known = y_train[known_mask]

    # Known classes (all except the held-out class)
    known_classes = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9])

    # TODO 5: devolver un objeto FewShotData
    return FewShotData(
        X_train_known=X_train_known,
        y_train_known=y_train_known,
        X_test=X_test,
        y_test=y_test,
        known_classes=known_classes,
        novel_class=novel_class,
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
    model = models.Sequential(
        [
            layers.Input(shape=(28, 28, 1)),
            layers.Conv2D(32, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Conv2D(64, kernel_size=3, activation="relu"),
            layers.MaxPooling2D(pool_size=2),
            layers.Flatten(),
            layers.Dense(64, activation="relu", name="embedding"),
            layers.Dense(num_classes, activation="softmax", name="classifier"),
        ]
    )
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
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
    mapping = {int(label): idx for idx, label in enumerate(known_classes)}
    return np.array([mapping[int(v)] for v in y], dtype=np.int64)


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
    return tf.keras.Model(inputs=classifier.inputs, outputs=classifier.get_layer("embedding").output, name="feature_extractor")


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


def save_classifier(classifier: tf.keras.Model, output_dir: str = "../../P9/outputs") -> str:
    """
    Guarda el clasificador entrenado en `output_dir` y devuelve la ruta donde se guardó.

    Parámetros
    ----------
    classifier : tf.keras.Model
        Modelo Keras entrenado a guardar.
    output_dir : str
        Carpeta donde se almacenará el fichero del modelo.

    Devuelve
    -------
    str
        Ruta completa al archivo guardado.
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "classifier_known_classes.h5")
    classifier.save(model_path)
    return model_path


def export_1shot_prototype(feature_extractor: tf.keras.Model, data: FewShotData, n_shots: int = 1, output_dir: str = "../../P9/outputs") -> str:
    """
    Crea y guarda un prototipo de `n_shots` para la clase reservada (por ejemplo, el dígito 7).

    Esta función selecciona `n_shots` ejemplos de `data.novel_class` desde `data.X_test`,
    calcula sus embeddings usando `feature_extractor`, promedia los embeddings para formar
    el prototipo y guarda el vector prototipo en formato `.npy` en `output_dir`.

    Parámetros
    ----------
    feature_extractor : tf.keras.Model
        Modelo que extrae embeddings a partir de imágenes.
    data : FewShotData
        Objeto que contiene los conjuntos de datos (train/test) y metadatos.
    n_shots : int
        Número de ejemplos de soporte a usar para construir el prototipo.
    output_dir : str
        Carpeta donde se guardará el prototipo.

    Devuelve
    -------
    str
        Ruta al fichero `.npy` con el prototipo guardado.
    """
    os.makedirs(output_dir, exist_ok=True)
    novel = int(data.novel_class)
    idxs = np.where(data.y_test == novel)[0]
    if len(idxs) < n_shots:
        raise ValueError("Not enough examples of the novel class in the test set to sample support shots")
    chosen = np.random.choice(idxs, size=n_shots, replace=False)
    X_support = data.X_test[chosen]
    # Calcular embeddings
    embeddings = feature_extractor.predict(X_support)
    prototype = embeddings.mean(axis=0)
    out_path = os.path.join(output_dir, f"prototype_{novel}_{n_shots}shot.npy")
    np.save(out_path, prototype)
    return out_path


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
