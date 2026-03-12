#!/usr/bin/env python3
"""Práctica 3 — Metodología de Evaluación y Rigor Científico

Tareas
------
1) Lotería de la partición aleatoria: variabilidad de positivos en test.
2) Partición estratificada: StratifiedKFold preserva la proporción de clases.
3) Comparativa de varianza: KFold vs StratifiedKFold (Accuracy y F1).
4) Detección de data leakage: variable trampa ID_Hospital_Filtro.

Salida
------
- Por pantalla: tablas de resultados por tarea.
- Figuras en outputs/:
    tarea1_positivos_test_aleatorio.png
    tarea2_proporcion_folds_estratificado.png
    tarea3_boxplot_accuracy.png
    tarea3_boxplot_f1.png
    tarea4_accuracy_leakage.png
    tarea4_correlaciones_leakage.png
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    train_test_split,
    KFold,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import f1_score, make_scorer
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "pacientes_riesgo.csv"
OUTPUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------
SEPARADOR = "=" * 62


TRAMPA_COL = "ID_Hospital_Filtro"


def cargar_datos() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Carga el CSV y devuelve (df, X_limpio, X_con_trampa, y)."""
    df = pd.read_csv(DATA_PATH)
    y = df["Clase"].values.astype(int)
    cols_limpias = [c for c in df.columns if c not in ("Clase", TRAMPA_COL)]
    cols_trampa = [c for c in df.columns if c != "Clase"]
    X_limpio = df[cols_limpias].values.astype(float)
    X_trampa = df[cols_trampa].values.astype(float)
    return df, X_limpio, X_trampa, y


f1_scorer = make_scorer(f1_score, zero_division=0)

def modelo_base() -> LogisticRegression:
    """Regresión logística con equilibrio de clases para datos desbalanceados."""
    return LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)


# ===========================================================================
# TAREA 1 — Lotería de la partición aleatoria
# ===========================================================================

def tarea1(X_limpio: np.ndarray, y: np.ndarray) -> None:
    X = X_limpio  # alias
    print(f"\n{SEPARADOR}")
    print("TAREA 1 — Lotería de la partición aleatoria")
    print(SEPARADOR)

    seeds = [0, 7, 21, 42, 99]
    positivos_test = []

    for i, seed in enumerate(seeds, start=1):
        _, X_test, _, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed
        )
        n_pos = int(y_test.sum())
        positivos_test.append(n_pos)
        print(f"  Ejecución {i} (seed={seed:>2d}): positivos en test = {n_pos}")

    print(f"\n  Rango: [{min(positivos_test)}, {max(positivos_test)}]  |  "
          f"std = {np.std(positivos_test):.2f}")

    # ---- Figura ----
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([f"Run {i}" for i in range(1, 6)], positivos_test,
           color="steelblue", edgecolor="white")
    ax.axhline(np.mean(positivos_test), color="crimson", linestyle="--",
               label=f"Media = {np.mean(positivos_test):.1f}")
    ax.set_xlabel("Ejecución")
    ax.set_ylabel("Nº de positivos en test")
    ax.set_title("Tarea 1 — Positivos en test por partición aleatoria")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tarea1_positivos_test_aleatorio.png", dpi=150)
    plt.close()
    print(f"  [OK] Figura guardada en outputs/tarea1_positivos_test_aleatorio.png")


# ===========================================================================
# TAREA 2 — Partición estratificada
# ===========================================================================

def tarea2(X_limpio: np.ndarray, y: np.ndarray) -> None:
    X = X_limpio  # alias
    print(f"\n{SEPARADOR}")
    print("TAREA 2 — Partición estratificada (StratifiedKFold)")
    print(SEPARADOR)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    filas = []
    for fold, (_, test_idx) in enumerate(skf.split(X, y), start=1):
        y_test = y[test_idx]
        n_pos = int(y_test.sum())
        size = len(test_idx)
        prop = n_pos / size
        filas.append({"Fold": fold, "Positivos": n_pos,
                       "Tamaño": size, "Proporción clase 1": prop})
        print(f"  Fold {fold}: positivos={n_pos:>2d}  tamaño={size}  "
              f"proporción={prop:.4f} ({100*prop:.2f} %)")

    df_res = pd.DataFrame(filas)
    print(f"\n  Media proporción: {df_res['Proporción clase 1'].mean():.4f}  |  "
          f"std: {df_res['Proporción clase 1'].std():.6f}")

    # ---- Figura ----
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(df_res["Fold"].astype(str), df_res["Proporción clase 1"] * 100,
           color="mediumseagreen", edgecolor="white")
    ax.axhline(y[y == 1].sum() / len(y) * 100, color="crimson",
               linestyle="--", label=f"Global = {100*y.mean():.2f} %")
    ax.set_xlabel("Fold")
    ax.set_ylabel("Proporción clase 1 (%)")
    ax.set_title("Tarea 2 — Proporción de la clase positiva por fold (StratifiedKFold)")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tarea2_proporcion_folds_estratificado.png", dpi=150)
    plt.close()
    print(f"  [OK] Figura guardada en outputs/tarea2_proporcion_folds_estratificado.png")


# ===========================================================================
# TAREA 3 — Comparativa de la varianza
# ===========================================================================

def tarea3(X_limpio: np.ndarray, y: np.ndarray) -> None:
    print(f"\n{SEPARADOR}")
    print("TAREA 3 — Comparativa de varianza: KFold vs StratifiedKFold")
    print(SEPARADOR)

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_limpio)  # solo variables limpias
    clf = modelo_base()
    n_splits = 10

    resultados: dict[str, dict[str, list[float]]] = {}

    # Corremos 10 iteraciones con semillas distintas para capturar variabilidad
    seeds = list(range(10))
    for nombre, cv_cls in [("KFold", KFold), ("StratifiedKFold", StratifiedKFold)]:
        acc_list, f1_list = [], []
        for seed in seeds:
            cv = cv_cls(n_splits=n_splits, shuffle=True, random_state=seed)
            acc = cross_val_score(clf, X_std, y, cv=cv,
                                  scoring="accuracy").tolist()
            f1 = cross_val_score(clf, X_std, y, cv=cv,
                                 scoring=f1_scorer).tolist()
            acc_list.extend(acc)
            f1_list.extend(f1)
        resultados[nombre] = {"accuracy": acc_list, "f1": f1_list}

        print(f"\n  {nombre}:")
        print(f"    Accuracy — media={np.mean(acc_list):.4f}  "
              f"std={np.std(acc_list):.4f}")
        print(f"    F1       — media={np.mean(f1_list):.4f}  "
              f"std={np.std(f1_list):.4f}")

    # ---- Figura Accuracy ----
    fig, ax = plt.subplots(figsize=(7, 5))
    data_acc = [resultados[m]["accuracy"] for m in ["KFold", "StratifiedKFold"]]
    bp = ax.boxplot(data_acc, tick_labels=["KFold", "StratifiedKFold"],
                    patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], ["steelblue", "mediumseagreen"]):
        patch.set_facecolor(color)
    ax.set_ylabel("Accuracy")
    ax.set_title("Tarea 3 — Distribución de Accuracy (100 evaluaciones)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tarea3_boxplot_accuracy.png", dpi=150)
    plt.close()

    # ---- Figura F1 ----
    fig, ax = plt.subplots(figsize=(7, 5))
    data_f1 = [resultados[m]["f1"] for m in ["KFold", "StratifiedKFold"]]
    bp = ax.boxplot(data_f1, tick_labels=["KFold", "StratifiedKFold"],
                    patch_artist=True, notch=False)
    for patch, color in zip(bp["boxes"], ["steelblue", "mediumseagreen"]):
        patch.set_facecolor(color)
    ax.set_ylabel("F1-score")
    ax.set_title("Tarea 3 — Distribución de F1-score (100 evaluaciones)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tarea3_boxplot_f1.png", dpi=150)
    plt.close()
    print(f"\n  [OK] Figuras guardadas en outputs/tarea3_boxplot_*.png")


# ===========================================================================
# TAREA 4 — Detección de data leakage
# ===========================================================================

def tarea4(X_limpio: np.ndarray, X_trampa: np.ndarray,
           df: pd.DataFrame, y: np.ndarray) -> None:
    print(f"\n{SEPARADOR}")
    print("TAREA 4 — Detección de data leakage")
    print(SEPARADOR)

    clf = modelo_base()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Escenario A: sin variable trampa
    X_limpio_std = StandardScaler().fit_transform(X_limpio)

    acc_limpio = cross_val_score(clf, X_limpio_std, y, cv=skf,
                                 scoring="accuracy")
    f1_limpio = cross_val_score(clf, X_limpio_std, y, cv=skf,
                                scoring=f1_scorer)

    # Escenario B: con variable trampa
    X_trampa_std = StandardScaler().fit_transform(X_trampa)

    acc_trampa = cross_val_score(clf, X_trampa_std, y, cv=skf,
                                 scoring="accuracy")
    f1_trampa = cross_val_score(clf, X_trampa_std, y, cv=skf,
                                scoring=f1_scorer)

    print("\n  Sin variable trampa:")
    print(f"    Accuracy = {np.mean(acc_limpio):.4f}  |  F1 = {np.mean(f1_limpio):.4f}")
    print("\n  Con variable trampa (ID_Hospital_Filtro):")
    print(f"    Accuracy = {np.mean(acc_trampa):.4f}  |  F1 = {np.mean(f1_trampa):.4f}")
    print("\n  >>> La variable trampa infla artificialmente el rendimiento.")

    # ---- Figura comparativa Accuracy / F1 ----
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    escenarios = ["Sin trampa", "Con trampa"]
    colores = ["mediumseagreen", "crimson"]

    for ax, scores_limpio, scores_trampa, metrica in zip(
        axes,
        [acc_limpio, f1_limpio],
        [acc_trampa, f1_trampa],
        ["Accuracy", "F1-score"],
    ):
        medias = [np.mean(scores_limpio), np.mean(scores_trampa)]
        bars = ax.bar(escenarios, medias, color=colores, edgecolor="white")
        for bar, val in zip(bars, medias):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=11,
                    fontweight="bold")
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(metrica)
        ax.set_title(f"Tarea 4 — {metrica}")
        ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.suptitle("Impacto del Data Leakage (ID_Hospital_Filtro)", fontsize=13,
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tarea4_accuracy_leakage.png", dpi=150)
    plt.close()

    # ---- Figura correlaciones ----
    feat_cols = [c for c in df.columns if c != "Clase"]
    correlaciones = df[feat_cols].apply(
        lambda col: np.corrcoef(col.values, y)[0, 1]
    )
    correlaciones = correlaciones.sort_values(key=abs, ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    colores_barra = [
        "crimson" if col == TRAMPA_COL else "steelblue"
        for col in correlaciones.index
    ]
    ax.bar(correlaciones.index, correlaciones.values,
           color=colores_barra, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Correlación de Pearson con Clase")
    ax.set_title("Tarea 4 — Correlación de cada variable con el objetivo\n"
                 "(rojo = variable trampa)")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "tarea4_correlaciones_leakage.png", dpi=150)
    plt.close()
    print(f"  [OK] Figuras guardadas en outputs/tarea4_*.png")


# ===========================================================================
# MAIN
# ===========================================================================

def main() -> None:
    print(SEPARADOR)
    print("Práctica 3 — Metodología de Evaluación y Rigor Científico")
    print(SEPARADOR)

    df, X_limpio, X_trampa, y = cargar_datos()
    print(f"\nDataset cargado: {len(df)} muestras  |  "
          f"Clase 1: {y.sum()} ({100*y.mean():.1f} %)")

    tarea1(X_limpio, y)
    tarea2(X_limpio, y)
    tarea3(X_limpio, y)
    tarea4(X_limpio, X_trampa, df, y)

    print(f"\n{SEPARADOR}")
    print("Todas las tareas completadas. Figuras en outputs/")
    print(SEPARADOR)


if __name__ == "__main__":
    main()
