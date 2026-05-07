"""
Práctica 8: Active Learning (Aprendizaje Activo)
Versión resuelta para la carpeta copilot.

El dataset ya está preparado en la carpeta ../data.
No tienes que generar los datos: se implementa el entrenamiento inicial,
la selección aleatoria, la selección por incertidumbre y la curva comparativa.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

CURRENT_DIR = Path(__file__).resolve().parent
UTILS_PATH = CURRENT_DIR.parent / "python" / "utils.py"

spec = importlib.util.spec_from_file_location("p8_utils", UTILS_PATH)
if spec is None or spec.loader is None:
    raise ImportError(f"No se pudo cargar el módulo de utilidades desde {UTILS_PATH}")

p8_utils = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p8_utils)

RANDOM_STATE = p8_utils.RANDOM_STATE
accuracy = p8_utils.accuracy
add_queried_points = p8_utils.add_queried_points
load_data = p8_utils.load_data
plot_learning_curves = p8_utils.plot_learning_curves
train_model = p8_utils.train_model


BATCH_SIZE = 5
MAX_LABELS = 50


def select_random(X_unlabeled: np.ndarray, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    """Selecciona aleatoriamente `batch_size` índices del pool no etiquetado.

    TODO 2: estrategia baseline con selección aleatoria.
    """
    n_select = min(batch_size, len(X_unlabeled))
    return rng.choice(len(X_unlabeled), size=n_select, replace=False)


def select_by_uncertainty(model, X_unlabeled: np.ndarray, batch_size: int) -> np.ndarray:
    """Selecciona los puntos donde el modelo tiene más incertidumbre.

    TODO 3: ciclo de consulta mediante incertidumbre.
    """
    probs = model.predict_proba(X_unlabeled)[:, 1]
    uncertainty = np.abs(probs - 0.5)
    n_select = min(batch_size, len(X_unlabeled))
    return np.argsort(uncertainty)[:n_select]


def run_query_strategy(strategy: str) -> tuple[list[int], list[float]]:
    """Ejecuta el ciclo de consulta para una estrategia.

    TODO 1: entrenamiento inicial con 10 etiquetas.
    TODO 2: baseline aleatorio.
    TODO 3: active learning por incertidumbre.
    TODO 4: curva comparativa y seguimiento del rendimiento.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    X_train, y_train, X_unlabeled, y_unlabeled, X_test, y_test = load_data()

    n_labels_history: list[int] = []
    accuracy_history: list[float] = []

    while True:
        # TODO 1: entrenar el clasificador con las etiquetas disponibles.
        model = train_model(X_train, y_train)

        # TODO 1: evaluar el rendimiento inicial y en cada iteración.
        test_acc = accuracy(model, X_test, y_test)

        # TODO 4: guardar el número de etiquetas usadas y el accuracy.
        n_labels_history.append(len(y_train))
        accuracy_history.append(test_acc)

        # TODO 4: detener el ciclo cuando se alcanza el presupuesto máximo.
        if len(y_train) >= MAX_LABELS or len(X_unlabeled) == 0:
            break

        # TODO 2 y TODO 3: escoger los puntos a consultar según la estrategia.
        if strategy == "random":
            query_idx = select_random(X_unlabeled, BATCH_SIZE, rng)
        elif strategy == "uncertainty":
            query_idx = select_by_uncertainty(model, X_unlabeled, BATCH_SIZE)
        else:
            raise ValueError("strategy debe ser 'random' o 'uncertainty'")

        # TODO 2 y TODO 3: consultar el oráculo y actualizar train/pool.
        X_train, y_train, X_unlabeled, y_unlabeled = add_queried_points(
            X_train,
            y_train,
            X_unlabeled,
            y_unlabeled,
            query_idx,
        )

    return n_labels_history, accuracy_history


def main() -> None:
    # TODO 1: comprobar el rendimiento inicial con solo 10 etiquetas.
    X_initial, y_initial, X_unlabeled, y_unlabeled, X_test, y_test = load_data()
    initial_model = train_model(X_initial, y_initial)
    initial_acc = accuracy(initial_model, X_test, y_test)
    print(f"Accuracy inicial con 10 etiquetas: {initial_acc:.4f}")

    # TODO 2: ejecutar la estrategia aleatoria.
    random_labels, random_acc = run_query_strategy("random")

    # TODO 3: ejecutar la estrategia por incertidumbre.
    uncertainty_labels, uncertainty_acc = run_query_strategy("uncertainty")

    # TODO 4: representar ambas curvas de aprendizaje.
    plot_learning_curves(random_labels, random_acc, uncertainty_labels, uncertainty_acc)

    # TODO 4 y TODO 5: resumir resultados y compararlos.
    print("\nResultados finales:")
    print(f"- Aleatorio:     {random_labels[-1]} etiquetas, accuracy = {random_acc[-1]:.4f}")
    print(f"- Incertidumbre: {uncertainty_labels[-1]} etiquetas, accuracy = {uncertainty_acc[-1]:.4f}")


if __name__ == "__main__":
    main()