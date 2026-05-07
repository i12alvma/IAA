"""
Práctica 8: Active Learning (Aprendizaje Activo)
Versión resuelta para la carpeta copilot.

El dataset ya está preparado en la carpeta ../data.
No tienes que generar los datos: se implementa el entrenamiento inicial,
la selección aleatoria, la selección por incertidumbre y la curva comparativa.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
PYTHON_DIR = CURRENT_DIR.parent / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from utils import (
    RANDOM_STATE,
    accuracy,
    add_queried_points,
    load_data,
    plot_learning_curves,
    train_model,
)


BATCH_SIZE = 5
MAX_LABELS = 50


def select_random(X_unlabeled: np.ndarray, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    """Selecciona aleatoriamente `batch_size` índices del pool no etiquetado."""
    n_select = min(batch_size, len(X_unlabeled))
    return rng.choice(len(X_unlabeled), size=n_select, replace=False)


def select_by_uncertainty(model, X_unlabeled: np.ndarray, batch_size: int) -> np.ndarray:
    """Selecciona los puntos donde el modelo tiene más incertidumbre."""
    probs = model.predict_proba(X_unlabeled)[:, 1]
    uncertainty = np.abs(probs - 0.5)
    n_select = min(batch_size, len(X_unlabeled))
    return np.argsort(uncertainty)[:n_select]


def run_query_strategy(strategy: str) -> tuple[list[int], list[float]]:
    """Ejecuta el ciclo de consulta para una estrategia."""
    rng = np.random.default_rng(RANDOM_STATE)

    X_train, y_train, X_unlabeled, y_unlabeled, X_test, y_test = load_data()

    n_labels_history: list[int] = []
    accuracy_history: list[float] = []

    while True:
        model = train_model(X_train, y_train)
        test_acc = accuracy(model, X_test, y_test)

        n_labels_history.append(len(y_train))
        accuracy_history.append(test_acc)

        if len(y_train) >= MAX_LABELS or len(X_unlabeled) == 0:
            break

        if strategy == "random":
            query_idx = select_random(X_unlabeled, BATCH_SIZE, rng)
        elif strategy == "uncertainty":
            query_idx = select_by_uncertainty(model, X_unlabeled, BATCH_SIZE)
        else:
            raise ValueError("strategy debe ser 'random' o 'uncertainty'")

        X_train, y_train, X_unlabeled, y_unlabeled = add_queried_points(
            X_train,
            y_train,
            X_unlabeled,
            y_unlabeled,
            query_idx,
        )

    return n_labels_history, accuracy_history


def main() -> None:
    X_initial, y_initial, X_unlabeled, y_unlabeled, X_test, y_test = load_data()
    initial_model = train_model(X_initial, y_initial)
    initial_acc = accuracy(initial_model, X_test, y_test)
    print(f"Accuracy inicial con 10 etiquetas: {initial_acc:.4f}")

    random_labels, random_acc = run_query_strategy("random")
    uncertainty_labels, uncertainty_acc = run_query_strategy("uncertainty")

    plot_learning_curves(random_labels, random_acc, uncertainty_labels, uncertainty_acc)

    print("\nResultados finales:")
    print(f"- Aleatorio:     {random_labels[-1]} etiquetas, accuracy = {random_acc[-1]:.4f}")
    print(f"- Incertidumbre: {uncertainty_labels[-1]} etiquetas, accuracy = {uncertainty_acc[-1]:.4f}")


if __name__ == "__main__":
    main()