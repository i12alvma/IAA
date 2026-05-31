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


def sample_support_set(
    X: np.ndarray,
    y: np.ndarray,
    class_label: int,
    n_shots: int
) -> tuple[np.ndarray, np.ndarray]:

    indices = np.where(y == class_label)[0]

    selected = np.random.choice(
        indices,
        size=n_shots,
        replace=False
    )

    return X[selected], y[selected]


def compute_prototypes(
    feature_extractor: tf.keras.Model,
    X_support: np.ndarray,
    y_support: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:

    embeddings = feature_extractor.predict(X_support, verbose=0)

    prototypes = []
    prototype_labels = []

    for label in np.unique(y_support):

        class_embeddings = embeddings[y_support == label]

        prototype = np.mean(class_embeddings, axis=0)

        prototypes.append(prototype)
        prototype_labels.append(label)

    return (
        np.array(prototypes),
        np.array(prototype_labels)
    )


def classify_by_nearest_prototype(
    feature_extractor: tf.keras.Model,
    X_query: np.ndarray,
    prototypes: np.ndarray,
    prototype_labels: np.ndarray
) -> np.ndarray:

    query_embeddings = feature_extractor.predict(
        X_query,
        verbose=0
    )

    predictions = []

    for embedding in query_embeddings:

        distances = np.linalg.norm(
            prototypes - embedding,
            axis=1
        )

        nearest = np.argmin(distances)

        predictions.append(
            prototype_labels[nearest]
        )

    return np.array(predictions)


def build_fewshot_episode(
    data: FewShotData,
    n_shots: int,
    n_query_per_class: int = 100
):

    X_support_list = []
    y_support_list = []

    for cls in data.known_classes:

        X_s, y_s = sample_support_set(
            data.X_train_known,
            data.y_train_known,
            cls,
            n_shots
        )

        X_support_list.append(X_s)
        y_support_list.append(y_s)

    X7, y7 = sample_support_set(
        data.X_test,
        data.y_test,
        data.novel_class,
        n_shots
    )

    X_support_list.append(X7)
    y_support_list.append(y7)

    X_support = np.concatenate(X_support_list)
    y_support = np.concatenate(y_support_list)

    X_query_list = []
    y_query_list = []

    all_classes = list(data.known_classes)
    all_classes.append(data.novel_class)

    for cls in all_classes:

        idx = np.where(data.y_test == cls)[0]

        chosen = np.random.choice(
            idx,
            size=n_query_per_class,
            replace=False
        )

        X_query_list.append(
            data.X_test[chosen]
        )

        y_query_list.append(
            data.y_test[chosen]
        )

    X_query = np.concatenate(X_query_list)
    y_query = np.concatenate(y_query_list)

    return (
        X_support,
        y_support,
        X_query,
        y_query
    )


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


def plot_embeddings_pca(
    feature_extractor: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    output_path: str = "fewshot_embeddings_pca.png"
):

    embeddings = feature_extractor.predict(
        X,
        verbose=0
    )

    pca = PCA(n_components=2)

    reduced = pca.fit_transform(
        embeddings
    )

    plt.figure(figsize=(8, 6))

    for cls in np.unique(y):

        mask = y == cls

        plt.scatter(
            reduced[mask, 0],
            reduced[mask, 1],
            s=10,
            alpha=0.6,
            label=str(cls)
        )

    plt.legend()
    plt.title("PCA of embedding space")
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200
    )

    plt.show()


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

    outputs_dir = os.path.join(
        os.path.dirname(__file__),
        "..",
        "outputs"
    )

    outputs_dir = os.path.normpath(outputs_dir)

    results = {}

    for shots in [1, 5]:

        X_support, y_support, X_query, y_query = build_fewshot_episode(
            data,
            n_shots=shots
        )

        prototypes, prototype_labels = compute_prototypes(
            feature_extractor,
            X_support,
            y_support
        )

        predictions = classify_by_nearest_prototype(
            feature_extractor,
            X_query,
            prototypes,
            prototype_labels
        )

        acc = accuracy_score(
            y_query,
            predictions
        )

        results[f"{shots}-shot"] = acc

        print(f"{shots}-shot accuracy: {acc:.4f}")

    plot_accuracy_comparison(
        results,
        output_path=os.path.join(
            outputs_dir,
            "fewshot_accuracy_comparison.png"
        )
    )

    plot_embeddings_pca(
        feature_extractor,
        X_query,
        y_query,
        output_path=os.path.join(
            outputs_dir,
            "fewshot_embeddings_pca.png"
        )
    )
    model_path = save_classifier(classifier, output_dir=outputs_dir)
    print(f"Saved classifier to: {model_path}")

    prototype_path = export_1shot_prototype(
        feature_extractor,
        data,
        n_shots=1,
        output_dir=outputs_dir
    )

    print(f"Saved 1-shot prototype to: {prototype_path}")

    prototype_path_5 = export_1shot_prototype(
        feature_extractor,
        data,
        n_shots=5,
        output_dir=outputs_dir
    )

    print(f"Saved 5-shot prototype to: {prototype_path_5}")


if __name__ == "__main__":
    main()
