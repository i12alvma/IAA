"""
Práctica 9 - Few-shot Learning con MNIST
Solución del profesor

Esta solución implementa un escenario few-shot sencillo:
1. Se entrena una CNN sin ver el dígito 7.
2. Se reutiliza la capa intermedia como extractor de características.
3. Se construyen prototipos con 1 y 5 ejemplos por clase.
4. Se clasifica por distancia euclídea al prototipo más cercano.
"""

import random
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
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
    X_train_seven: np.ndarray
    y_train_seven: np.ndarray
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
        Dataset object with known classes and the held-out class 7.
    """
    (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

    X_train = X_train.astype("float32") / 255.0
    X_test = X_test.astype("float32") / 255.0

    X_train = np.expand_dims(X_train, axis=-1)
    X_test = np.expand_dims(X_test, axis=-1)

    novel_class = 7
    known_mask = y_train != novel_class
    seven_mask = y_train == novel_class

    X_train_known = X_train[known_mask]
    y_train_known = y_train[known_mask]
    X_train_seven = X_train[seven_mask]
    y_train_seven = y_train[seven_mask]

    known_classes = np.array([0, 1, 2, 3, 4, 5, 6, 8, 9])

    return FewShotData(
        X_train_known=X_train_known,
        y_train_known=y_train_known,
        X_train_seven=X_train_seven,
        y_train_seven=y_train_seven,
        X_test=X_test,
        y_test=y_test,
        known_classes=known_classes,
        novel_class=novel_class,
    )


def remap_known_labels(y: np.ndarray, known_classes: np.ndarray) -> np.ndarray:
    """
    Remap original MNIST labels to a compact range for CNN training.

    Parameters
    ----------
    y : np.ndarray
        Original labels.
    known_classes : np.ndarray
        Known classes.

    Returns
    -------
    np.ndarray
        Remapped labels in 0..len(known_classes)-1.
    """
    mapping = {label: idx for idx, label in enumerate(known_classes)}
    return np.array([mapping[int(label)] for label in y], dtype=np.int64)


def build_classifier(num_classes: int) -> tf.keras.Model:
    """
    Build a simple CNN classifier with an explicit embedding layer.

    Parameters
    ----------
    num_classes : int
        Number of known classes.

    Returns
    -------
    tf.keras.Model
        Compiled classifier.
    """
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
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def evaluate_known_class_classifier(classifier: tf.keras.Model, data: FewShotData) -> float:
    """
    Evaluate the initial classifier only on known classes.

    Parameters
    ----------
    classifier : tf.keras.Model
        Trained classifier.
    data : FewShotData
        Dataset object.

    Returns
    -------
    float
        Accuracy over known classes in the test set.
    """
    known_test_mask = data.y_test != data.novel_class
    X_test_known = data.X_test[known_test_mask]
    y_test_known_original = data.y_test[known_test_mask]
    y_test_known = remap_known_labels(y_test_known_original, data.known_classes)
    _, acc = classifier.evaluate(X_test_known, y_test_known, verbose=0)
    return float(acc)


def create_feature_extractor(classifier: tf.keras.Model) -> tf.keras.Model:
    """
    Create a feature extractor using the embedding layer.

    Parameters
    ----------
    classifier : tf.keras.Model
        Trained classifier.

    Returns
    -------
    tf.keras.Model
        Feature extractor model.
    """
    return tf.keras.Model(
        inputs=classifier.inputs,
        outputs=classifier.get_layer("embedding").output,
        name="feature_extractor",
    )


def sample_class_examples(X: np.ndarray, y: np.ndarray, class_label: int, n_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sample examples from one class without replacement.

    Parameters
    ----------
    X : np.ndarray
        Images.
    y : np.ndarray
        Labels.
    class_label : int
        Target class.
    n_samples : int
        Number of examples to sample.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Sampled images and labels.
    """
    idx = np.where(y == class_label)[0]
    if len(idx) < n_samples:
        raise ValueError(f"Not enough examples for class {class_label}: requested {n_samples}, found {len(idx)}")
    selected = np.random.choice(idx, size=n_samples, replace=False)
    return X[selected], y[selected]


def build_fewshot_episode(data: FewShotData, n_shots: int, n_query_per_class: int = 100) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a balanced N-way K-shot episode for digits 0..9.

    The support set uses training images. For the novel class 7, images are
    taken from the held-out training sevens, which were not used to train the CNN.
    The query set is sampled from the official MNIST test set.

    Parameters
    ----------
    data : FewShotData
        Dataset object.
    n_shots : int
        Number of support images per class.
    n_query_per_class : int
        Number of query images per class.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Support images, support labels, query images and query labels.
    """
    support_images = []
    support_labels = []
    query_images = []
    query_labels = []

    all_classes = np.arange(10)

    for class_label in all_classes:
        if class_label == data.novel_class:
            X_s, y_s = sample_class_examples(data.X_train_seven, data.y_train_seven, class_label, n_shots)
        else:
            X_s, y_s = sample_class_examples(data.X_train_known, data.y_train_known, class_label, n_shots)

        X_q, y_q = sample_class_examples(data.X_test, data.y_test, class_label, n_query_per_class)

        support_images.append(X_s)
        support_labels.append(y_s)
        query_images.append(X_q)
        query_labels.append(y_q)

    X_support = np.concatenate(support_images, axis=0)
    y_support = np.concatenate(support_labels, axis=0)
    X_query = np.concatenate(query_images, axis=0)
    y_query = np.concatenate(query_labels, axis=0)

    return X_support, y_support, X_query, y_query


def compute_prototypes(feature_extractor: tf.keras.Model, X_support: np.ndarray, y_support: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute class prototypes as mean embeddings.

    Parameters
    ----------
    feature_extractor : tf.keras.Model
        Feature extractor.
    X_support : np.ndarray
        Support images.
    y_support : np.ndarray
        Support labels.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Prototype matrix and corresponding labels.
    """
    embeddings = feature_extractor.predict(X_support, verbose=0)
    class_labels = np.unique(y_support)

    prototypes = []
    for class_label in class_labels:
        class_embeddings = embeddings[y_support == class_label]
        prototypes.append(class_embeddings.mean(axis=0))

    return np.vstack(prototypes), class_labels


def classify_by_nearest_prototype(feature_extractor: tf.keras.Model, X_query: np.ndarray, prototypes: np.ndarray, prototype_labels: np.ndarray) -> np.ndarray:
    """
    Classify query images by nearest prototype.

    Parameters
    ----------
    feature_extractor : tf.keras.Model
        Feature extractor.
    X_query : np.ndarray
        Query images.
    prototypes : np.ndarray
        Prototypes.
    prototype_labels : np.ndarray
        Prototype labels.

    Returns
    -------
    np.ndarray
        Predicted labels.
    """
    query_embeddings = feature_extractor.predict(X_query, verbose=0)
    distances = np.linalg.norm(query_embeddings[:, None, :] - prototypes[None, :, :], axis=2)
    nearest = np.argmin(distances, axis=1)
    return prototype_labels[nearest]


def plot_accuracy_comparison(results: dict[str, float], output_path: str = "fewshot_accuracy_comparison.png") -> None:
    """
    Plot accuracy comparison between few-shot settings.

    Parameters
    ----------
    results : dict[str, float]
        Mapping from setting name to accuracy.
    output_path : str
        Output figure path.
    """
    plt.figure(figsize=(6, 4))
    bars = plt.bar(list(results.keys()), list(results.values()))
    plt.ylim(0, 1)
    plt.ylabel("Accuracy")
    plt.title("Few-shot classification with MNIST prototypes")
    plt.grid(axis="y", alpha=0.3)

    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height + 0.02, f"{height:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_embeddings_pca(
    feature_extractor: tf.keras.Model,
    X: np.ndarray,
    y: np.ndarray,
    prototypes: np.ndarray | None = None,
    prototype_labels: np.ndarray | None = None,
    output_path: str = "fewshot_embeddings_pca.png",
) -> None:
    """
    Visualize query embeddings and optionally prototypes using PCA.

    Parameters
    ----------
    feature_extractor : tf.keras.Model
        Feature extractor.
    X : np.ndarray
        Query images.
    y : np.ndarray
        Query labels.
    prototypes : np.ndarray, optional
        Prototype vectors.
    prototype_labels : np.ndarray, optional
        Labels associated with prototypes.
    output_path : str
        Output figure path.
    """
    embeddings = feature_extractor.predict(X, verbose=0)

    if prototypes is not None:
        combined = np.vstack([embeddings, prototypes])
    else:
        combined = embeddings

    pca = PCA(n_components=2, random_state=SEED)
    projected = pca.fit_transform(combined)
    projected_query = projected[: len(embeddings)]

    plt.figure(figsize=(8, 6))
    for class_label in np.unique(y):
        mask = y == class_label
        plt.scatter(projected_query[mask, 0], projected_query[mask, 1], s=12, alpha=0.6, label=str(class_label))

    if prototypes is not None and prototype_labels is not None:
        projected_proto = projected[len(embeddings) :]
        plt.scatter(projected_proto[:, 0], projected_proto[:, 1], s=160, marker="X", edgecolors="black", linewidths=1.0, label="Prototypes")
        for point, label in zip(projected_proto, prototype_labels):
            plt.text(point[0], point[1], str(label), fontsize=10, weight="bold")

    plt.title("PCA visualization of MNIST embeddings")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_episode(feature_extractor: tf.keras.Model, data: FewShotData, n_shots: int) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run one few-shot episode.

    Parameters
    ----------
    feature_extractor : tf.keras.Model
        Feature extractor.
    data : FewShotData
        Dataset object.
    n_shots : int
        Number of shots.

    Returns
    -------
    tuple
        Accuracy, predictions, query labels, query images, prototypes and prototype labels.
    """
    X_support, y_support, X_query, y_query = build_fewshot_episode(data, n_shots=n_shots)
    prototypes, prototype_labels = compute_prototypes(feature_extractor, X_support, y_support)
    y_pred = classify_by_nearest_prototype(feature_extractor, X_query, prototypes, prototype_labels)
    acc = accuracy_score(y_query, y_pred)
    return acc, y_pred, y_query, X_query, prototypes, prototype_labels


def main() -> None:
    data = load_mnist_world_without_sevens()

    print("Known classes:", data.known_classes)
    print("Novel class:", data.novel_class)
    print("Training examples without sevens:", data.X_train_known.shape[0])
    print("Held-out training sevens:", data.X_train_seven.shape[0])

    y_train_known_remap = remap_known_labels(data.y_train_known, data.known_classes)

    classifier = build_classifier(num_classes=len(data.known_classes))
    classifier.summary()

    classifier.fit(
        data.X_train_known,
        y_train_known_remap,
        validation_split=0.1,
        epochs=3,
        batch_size=128,
        verbose=1,
    )

    known_acc = evaluate_known_class_classifier(classifier, data)
    print(f"Accuracy on known classes only: {known_acc:.4f}")

    feature_extractor = create_feature_extractor(classifier)

    results = {}
    saved_for_plot = None

    for n_shots in [1, 5]:
        acc, y_pred, y_query, X_query, prototypes, prototype_labels = run_episode(feature_extractor, data, n_shots=n_shots)
        results[f"{n_shots}-shot"] = acc

        print("\n" + "=" * 60)
        print(f"{n_shots}-shot results")
        print("=" * 60)
        print(f"Accuracy: {acc:.4f}")
        print("Confusion matrix:")
        print(confusion_matrix(y_query, y_pred, labels=np.arange(10)))
        print("Classification report:")
        print(classification_report(y_query, y_pred, labels=np.arange(10), zero_division=0))

        if n_shots == 5:
            saved_for_plot = (X_query, y_query, prototypes, prototype_labels)

    plot_accuracy_comparison(results, output_path="fewshot_accuracy_comparison.png")

    if saved_for_plot is not None:
        X_query, y_query, prototypes, prototype_labels = saved_for_plot
        # Usamos una submuestra para que la figura sea legible.
        sample_idx = np.random.choice(np.arange(len(X_query)), size=min(500, len(X_query)), replace=False)
        plot_embeddings_pca(
            feature_extractor,
            X_query[sample_idx],
            y_query[sample_idx],
            prototypes=prototypes,
            prototype_labels=prototype_labels,
            output_path="fewshot_embeddings_pca.png",
        )

    print("\nSummary:")
    for setting, acc in results.items():
        print(f"  {setting}: {acc:.4f}")


if __name__ == "__main__":
    main()
