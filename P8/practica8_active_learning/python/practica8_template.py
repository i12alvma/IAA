"""
Práctica 8: Active Learning (Aprendizaje Activo)
Versión recortada para el alumnado.

El dataset ya está preparado en la carpeta ../data.
No tienes que generar los datos: debes completar el entrenamiento inicial,
la selección aleatoria, la selección por incertidumbre y la curva comparativa.
"""

from __future__ import annotations
from sklearn.ensemble import RandomForestClassifier


import numpy as np
import pandas as pd

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
    """Selecciona aleatoriamente `batch_size` índices del pool no etiquetado.

    Debes devolver posiciones dentro de X_unlabeled, no los puntos completos.

    Pistas
    ------
    - Usa rng.choice.
    - No selecciones el mismo índice dos veces en la misma iteración.
    """
    # TODO: completa esta función.
    indices = rng.choice(len(X_unlabeled), size=batch_size, replace=False)
    return indices


def select_by_uncertainty(model, X_unlabeled: np.ndarray, batch_size: int) -> np.ndarray:
    """Selecciona los puntos donde el modelo tiene más incertidumbre.

    En clasificación binaria, la incertidumbre es mayor cuando la probabilidad
    predicha para la clase 1 está cerca de 0.5.

    Pistas
    ------
    - Usa model.predict_proba(X_unlabeled)[:, 1].
    - Calcula abs(probabilidad - 0.5).
    - Ordena de menor a mayor: los valores más pequeños son los más inciertos.
    - Devuelve los `batch_size` índices más inciertos.
    """
    # TODO: completa esta función.

    probs = model.predict_proba(X_unlabeled)[:, 1]

    uncertainty = np.abs(probs - 0.5)

    indices = np.argsort(uncertainty)[:batch_size]

    return indices

def run_query_strategy(strategy: str) -> tuple[list[int], list[float]]:
    """Ejecuta el ciclo de consulta para una estrategia.

    Parameters
    ----------
    strategy : {'random', 'uncertainty'}
        Estrategia de selección de nuevos puntos etiquetados.

    Returns
    -------
    n_labels_history : list of int
        Número de etiquetas usadas después de cada evaluación.
    accuracy_history : list of float
        Accuracy obtenido en test después de cada evaluación.
    """
    rng = np.random.default_rng(RANDOM_STATE)

    X_train, y_train, X_unlabeled, y_unlabeled, X_test, y_test = load_data()

    n_labels_history: list[int] = []
    accuracy_history: list[float] = []

    df_train = pd.read_csv("P8/practica8_active_learning/data/initial_labeled.csv")
    X_train = df_train[["x1", "x2"]].values
    y_train = df_train["y"].values

    df_test= pd.read_csv("P8/practica8_active_learning/data/test.csv")
    X_test = df_test[["x1", "x2"]].values
    y_test = df_test["y"].values
    while len(y_train) <= MAX_LABELS:
        # TODO 1: entrena el modelo con X_train, y_train.
        # TODO usando RandomForestClassifier

    
        model = RandomForestClassifier(random_state=RANDOM_STATE)
        model.fit(X_train, y_train)

        # TODO 2: evalúa el modelo en X_test, y_test y guarda el accuracy.

        acc = model.score(X_test, y_test)
        print(f"Accuracy: {acc:.4f}")
        
        # TODO 3: guarda también el número actual de etiquetas.
        accuracy_history.append(acc)
        n_labels_history.append(len(y_train))

        # TODO 4: si ya has llegado a MAX_LABELS, termina el bucle.
        
        if len(y_train) >= MAX_LABELS:
            break
        
        # TODO 5: selecciona los puntos a consultar según la estrategia.
        # - Si strategy == "random", usa select_random.
        # - Si strategy == "uncertainty", usa select_by_uncertainty.
        # - Si strategy tiene otro valor, lanza un ValueError.
        if strategy == "random":
            query_indices = select_random(
                X_unlabeled,
                BATCH_SIZE,
                rng
            )

        elif strategy == "uncertainty":
            query_indices = select_by_uncertainty(
                model,
                X_unlabeled,
                BATCH_SIZE
            )

        else:
            raise ValueError(f"Estrategia desconocida: {strategy}")
        
        # TODO 6: consulta el oráculo y actualiza train/pool usando add_queried_points.

        X_train, y_train, X_unlabeled, y_unlabeled = add_queried_points(
            X_train,
            y_train,
            X_unlabeled,
            y_unlabeled,
            query_indices
)

    return n_labels_history, accuracy_history


def main() -> None:
    # Entrenamiento inicial orientativo: puedes usar esta parte para comprobar
    # el rendimiento con solo 10 etiquetas antes de completar los bucles.
    #X_initial, y_initial, X_unlabeled, y_unlabeled, X_test, y_test = load_data()
    #initial_model = train_model(X_initial, y_initial)
    #initial_acc = accuracy(initial_model, X_test, y_test)
    #print(f"Accuracy inicial con 10 etiquetas: {initial_acc:.4f}")

    # TODO 7: ejecuta la estrategia aleatoria.
    random_labels, random_acc = run_query_strategy("random")

    # TODO 8: ejecuta la estrategia por incertidumbre.
    uncertainty_labels, uncertainty_acc = run_query_strategy("uncertainty")

    # TODO 9: representa ambas curvas de aprendizaje.
    plot_learning_curves(random_labels, random_acc, uncertainty_labels, uncertainty_acc)

    # TODO 10: imprime o comenta los resultados finales.

    print("\n=== RESULTADOS FINALES ===")
    print(f"Random sampling final accuracy: {random_acc[-1]:.4f}")
    print(f"Uncertainty sampling final accuracy: {uncertainty_acc[-1]:.4f}")


if __name__ == "__main__":
    main()
