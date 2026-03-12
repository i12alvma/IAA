#!/usr/bin/env python3
"""Genera el dataset sintético para la Práctica 3.

Crea un CSV con 1000 pacientes, 5 variables predictoras reales,
una variable trampa (ID_Hospital_Filtro) que filtra el objetivo,
y la clase objetivo fuertemente desbalanceada (2 % positivos).

Salida
------
data/pacientes_riesgo.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path


def main() -> None:
    rng = np.random.default_rng(42)
    n_samples = 1000
    prevalencia = 0.02

    # Variables predictoras (ruido gaussiano, sin relación real con la clase)
    X = rng.standard_normal((n_samples, 5))

    # Clase positiva: ~2 % de los casos
    y = (rng.random(n_samples) < prevalencia).astype(int)

    # Variable trampa: correlacionada artificialmente con la clase
    # Para la clase 1 el valor es ~N(5, 0.1); para la clase 0 es ~N(0, 0.1)
    id_hospital = np.where(y == 1,
                           rng.normal(5.0, 0.1, n_samples),
                           rng.normal(0.0, 0.1, n_samples))

    df = pd.DataFrame(X, columns=[f"Var_{i}" for i in range(5)])
    df["ID_Hospital_Filtro"] = id_hospital
    df["Clase"] = y

    out = Path(__file__).resolve().parent.parent / "data" / "pacientes_riesgo.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"[OK] Dataset guardado en: {out}")
    print(f"     Total muestras : {n_samples}")
    print(f"     Clase positiva : {y.sum()} ({100*y.mean():.1f} %)")


if __name__ == "__main__":
    main()
