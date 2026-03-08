#!/usr/bin/env python3
"""Práctica 2 — Regularización (Ridge vs Lasso) + Validación Cruzada (K-Fold)

Objetivo
--------
1) Entender cómo cambia el error de validación cruzada al variar K.
2) Ver cómo el parámetro de regularización (alpha/lambda) reduce coeficientes.
3) Comparar Ridge (encoge) vs Lasso (puede anular coeficientes).

Tareas (alumno)
---------------
A) Cambia `metodo` entre "lasso" y "ridge".
B) Prueba varios valores de `valor_lambda` (p.ej., 0.01, 0.1, 1, 10, 100).
C) Prueba varios valores de `k_folds` (p.ej., 2, 5, 10, 20) y comenta estabilidad vs coste.
D) (Opcional) Repite con distintas semillas en KFold para ver variabilidad.

Salida
------
- Por pantalla: RMSE medio de CV
- Figura: barras con coeficientes aprendidos
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.preprocessing import StandardScaler

DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "precios_viviendas.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"


def generar_graficas_tarea1(X_std: np.ndarray, y: np.ndarray) -> None:
    """Genera y guarda las figuras de estabilidad K-Fold para la Tarea 1."""
    modelo_lineal = LinearRegression()

    # Experimento A: K=2 con varias semillas
    seeds_k2 = [1, 7, 21, 42, 99]
    rmse_k2_runs = []
    for seed in seeds_k2:
        kf = KFold(n_splits=2, shuffle=True, random_state=seed)
        scores = cross_val_score(modelo_lineal, X_std, y, cv=kf, scoring="neg_mean_squared_error")
        rmse_k2_runs.append(float(np.sqrt(-np.mean(scores))))

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, len(rmse_k2_runs) + 1), rmse_k2_runs, marker="o")
    plt.xticks(range(1, len(rmse_k2_runs) + 1), [f"Run {i}" for i in range(1, len(rmse_k2_runs) + 1)])
    plt.ylabel("RMSE")
    plt.title("K-Fold K=2 | Variabilidad del RMSE entre ejecuciones")
    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kfold_K2_rmse_runs.png", dpi=150)
    plt.close()

    # Experimento B: K=10
    kf10 = KFold(n_splits=10, shuffle=True, random_state=42)
    scores_k10 = cross_val_score(modelo_lineal, X_std, y, cv=kf10, scoring="neg_mean_squared_error")
    rmse_folds_k10 = np.sqrt(-scores_k10)
    rmse_k10 = float(np.mean(rmse_folds_k10))

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, 11), rmse_folds_k10, marker="o")
    plt.axhline(rmse_k10, color="red", linestyle="--", label=f"RMSE medio={rmse_k10:.3f}")
    plt.xticks(range(1, 11))
    plt.xlabel("Fold")
    plt.ylabel("RMSE")
    plt.title("K-Fold K=10 | RMSE por fold")
    plt.grid(axis="y", linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kfold_K10_rmse.png", dpi=150)
    plt.close()

    # Experimento C: K=100 (LOOCV aproximado en este enunciado)
    kf100 = KFold(n_splits=100, shuffle=True, random_state=42)
    t0 = time.perf_counter()
    scores_k100 = cross_val_score(modelo_lineal, X_std, y, cv=kf100, scoring="neg_mean_squared_error")
    dt_ms = (time.perf_counter() - t0) * 1000
    rmse_folds_k100 = np.sqrt(-scores_k100)
    rmse_k100 = float(np.mean(rmse_folds_k100))

    plt.figure(figsize=(10, 4))
    plt.plot(range(1, 101), rmse_folds_k100, linewidth=1)
    plt.axhline(rmse_k100, color="red", linestyle="--", label=f"RMSE medio={rmse_k100:.3f}")
    plt.xlabel("Fold")
    plt.ylabel("RMSE")
    plt.title(f"K-Fold K=100 | RMSE por fold | tiempo={dt_ms:.1f} ms")
    plt.grid(axis="y", linestyle="--")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "kfold_K100_loocv_rmse.png", dpi=150)
    plt.close()

    print("[Tarea 1] Gráficas guardadas:")
    print(f"- {OUTPUT_DIR / 'kfold_K2_rmse_runs.png'}")
    print(f"- {OUTPUT_DIR / 'kfold_K10_rmse.png'}")
    print(f"- {OUTPUT_DIR / 'kfold_K100_loocv_rmse.png'}")
    print(f"[Tarea 1] RMSE K=10: {rmse_k10:.4f} | RMSE K=100: {rmse_k100:.4f} | tiempo K=100: {dt_ms:.1f} ms")

def main() -> None:
    # -----------------------
    # 1) Cargar datos
    # -----------------------
    df = pd.read_csv(DATA_PATH)

    X = df[[f"Var_{i}" for i in range(10)]].to_numpy()
    y = df["Precio"].to_numpy()

    # Normalización (imprescindible para regularización)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # -----------------------
    # 2) Configuración (MODIFICA ESTO)
    # -----------------------
    generar_tarea1 = True  # Si True, genera también las gráficas de la Tarea 1
    metodo = "ridge"       # Opciones: "lasso" o "ridge"
    valor_lambda = 1000     # Fuerza de la penalización (alpha en sklearn)
    k_folds = 10

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if generar_tarea1:
        generar_graficas_tarea1(X_std, y)

    # Consejo: en Lasso a veces necesitas subir max_iter si no converge
    if metodo == "lasso":
        modelo = Lasso(alpha=valor_lambda, max_iter=10000)
    else:
        modelo = Ridge(alpha=valor_lambda)

    # -----------------------
    # 3) Validación cruzada (K-Fold)
    # -----------------------
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=np.random.randint(1,1000))

    # Scoring: sklearn devuelve MSE negativo (porque maximiza scores)
    scores = cross_val_score(modelo, X_std, y, cv=kf, scoring="neg_mean_squared_error")
    rmse_cv = np.sqrt(-np.mean(scores))

    print(f"Metodo={metodo.upper()} | lambda(alpha)={valor_lambda} | K={k_folds} | RMSE_CV={rmse_cv:.2f}")

    # -----------------------
    # 4) Entrenar y visualizar coeficientes
    # -----------------------
    modelo.fit(X_std, y)
    coef = modelo.coef_

    plt.figure(figsize=(10, 4))
    plt.bar(range(10), coef)
    plt.axhline(0, color="black", linewidth=1.2)

    max_abs = float(np.max(np.abs(coef)))
    if max_abs < 1e-3:
        plt.ylim(-0.1, 0.1)
    else:
        plt.ylim(-1.1 * max_abs, 1.1 * max_abs)

    plt.xticks(range(10), [f"Var_{i}" for i in range(10)])
    plt.title(f"Coeficientes con {metodo.upper()} (λ={valor_lambda}) | RMSE CV: {rmse_cv:.2f}")
    plt.ylabel("Peso del coeficiente")
    plt.grid(axis="y", linestyle="--")
    plt.tight_layout()

    output_path = OUTPUT_DIR / f"{metodo.lower()}_lambda_{valor_lambda}_K{k_folds}_coeffs.png"
    plt.savefig(output_path, dpi=150)
    print(f"Gráfica guardada en: {output_path}")

    plt.show()

if __name__ == "__main__":
    main()
