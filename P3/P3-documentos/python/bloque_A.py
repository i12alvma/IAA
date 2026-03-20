import pandas as pd
import numpy as np
from pathlib import Path

# ----------------------------
# Carga de datos
# ----------------------------
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "pacientes_riesgo.csv"
df = pd.read_csv(DATA_PATH)

y = df["Clase"].values
X_clean = df.drop(columns=["Clase", "ID_Hospital_Filtro"]).values

# =========================================================
# TAREA 1 — Lotería de la partición aleatoria
# =========================================================
from sklearn.model_selection import train_test_split

print("\n--- TAREA 1: Partición aleatoria ---")

seeds = [0, 7, 21, 42, 99]
positivos_test = []

for i, seed in enumerate(seeds, start=1):
    _, X_test, _, y_test = train_test_split(
        X_clean, y, test_size=0.2, random_state=seed
    )

    n_pos = int(y_test.sum())
    positivos_test.append(n_pos)

    print(f"Ejecución {i} (seed={seed}): {n_pos} positivos en test")

# Resumen rápido
print("\nResumen:")
print(f"Rango: [{min(positivos_test)}, {max(positivos_test)}]")
print(f"Desviación típica: {np.std(positivos_test):.2f}")

# =========================================================
# TAREA 2 — Partición estratificada
# =========================================================
from sklearn.model_selection import StratifiedKFold

print("\n--- TAREA 2: StratifiedKFold ---")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

proporciones = []

for fold, (_, test_idx) in enumerate(skf.split(X_clean, y), start=1):
    y_test = y[test_idx]

    n_pos = int(y_test.sum())
    size = len(test_idx)
    prop = n_pos / size
    proporciones.append(prop)

    print(f"\nFold {fold}:")
    print(f"Positivos: {n_pos}")
    print(f"Tamaño: {size}")
    print(f"Proporción clase 1: {prop:.4f} ({prop*100:.2f}%)")

# Resumen final
print("\nResumen:")
print(f"Proporción media: {np.mean(proporciones):.4f}")
print(f"Desviación típica: {np.std(proporciones):.6f}")
print(f"Proporción global: {y.mean():.4f}")