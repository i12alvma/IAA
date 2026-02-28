# Práctica 2: Regularización

## Cuaderno de Laboratorio — Práctica 2: Selección de Modelos y Regularización

### Objetivo

Dominar el protocolo de **Validación Cruzada (K-Fold)** y comprender cómo la regularización **Ridge (L2)** y **Lasso (L1)** previenen el sobreajuste y ayudan a la **selección de atributos**.

### Material de partida

Se parte del script `seleccion_modelos_viviendas.py` (o `.jl`), que incluye:

- Un dataset sintético de **200 viviendas** con **10 variables predictoras**.
- Implementación de **validación cruzada** y del cálculo del **RMSE** (raíz del error cuadrático medio).

**Nota:** En este cuaderno se insertan capturas/gráficas generadas por el script (carpeta `outputs/`). Ajusta los nombres de fichero a los que obtengas tú.

---

## Tarea 1 — Estabilidad de la Validación Cruzada (K-Fold)

### Configuración común

- **Modelo usado:** _________________ (regresión lineal sin regularización)
- **Métrica:** _________________ (RMSE)
- **Semilla(s) / aleatoriedad:** _________________

### Hipótesis inicial (antes de ejecutar)

¿Crees que el RMSE será estable al cambiar K? ¿Por qué?

_____________________________________________________________________________

_____________________________________________________________________________

### Experimento A — K = 2 (Validación simple): variabilidad entre ejecuciones

Ejecuta el modelo **varias veces** con K=2 (por ejemplo 5–10) y registra el RMSE.

![Imagen: kfold_K2_rmse_runs](outputs/kfold_K2_rmse_runs.png)

#### Registro de resultados (rellena):

| Ejecución | RMSE |
|-----------|------|
| 1 | _________________ |
| 2 | _________________ |
| 3 | _________________ |
| 4 | _________________ |
| 5 | _________________ |

#### Análisis:

- ¿Por qué el error varía tanto entre ejecuciones?
- ¿Es fiable esta métrica (con K=2) para tomar una decisión técnica?

_____________________________________________________________________________

_____________________________________________________________________________

### Experimento B — K = 10 (estándar): RMSE más estable

Ejecuta el modelo con K=10 y anota el RMSE.

![Imagen: kfold_K10_rmse](outputs/kfold_K10_rmse.png)

**RMSE observado (K=10):** _________________

#### Análisis:

Describe si el resultado parece más estable que en K=2 y por qué.

_____________________________________________________________________________

_____________________________________________________________________________

### Experimento C — K = 100 (Leave-One-Out): coste vs. precisión

Ejecuta el modelo con K=100 (LOOCV) y observa el RMSE y el tiempo.

![Imagen: kfold_K100_loocv_rmse](outputs/kfold_K100_loocv_rmse.png)

**RMSE observado (K=100):** _________________ 
**Tiempo aproximado:** _________________

#### Análisis:

Compara el coste computacional frente a K=10. ¿Merece la pena el tiempo extra para mejorar la precisión obtenida?

_____________________________________________________________________________

_____________________________________________________________________________

### Resumen comparativo (K = 2 vs 10 vs 100)

| K | RMSE | Comentario de estabilidad/coste |
|---|------|----------------------------------|
| 2 | _________________ | _________________________________ |
| 10 | _________________ | _________________________________ |
| 100 | _________________ | _________________________________ |

#### Conclusión Tarea 1 (3–6 líneas):

¿Cuál elegirías para este problema y por qué?

_____________________________________________________________________________

_____________________________________________________________________________

_____________________________________________________________________________

---

## Tarea 2 — Regularización Ridge vs. Lasso (K = 10)

### Configuración

Fijamos K = 10 y comparamos:

- **Ridge Regression** (penalización L2)
- **Lasso Regression** (penalización L1)

### Ridge (L2) — λ = 0.1, 10, 100

**Resultados:** captura el gráfico de barras de coeficientes para cada λ.

![Imagen: ridge_lambda_0.1_coeffs](outputs/ridge_lambda_0.1_coeffs.png)

**RMSE Ridge (λ=0.1):** _________________

![Imagen: ridge_lambda_10_coeffs](outputs/ridge_lambda_10_coeffs.png)

**RMSE Ridge (λ=10):** _________________

![Imagen: ridge_lambda_100_coeffs](outputs/ridge_lambda_100_coeffs.png)

**RMSE Ridge (λ=100):** _________________

#### Análisis:

¿Llegan a valer **cero** los coeficientes de las variables ruidosas (`Var_4` a `Var_10`)? Explica qué observas y por qué pasa (o no pasa).

_____________________________________________________________________________

_____________________________________________________________________________

_____________________________________________________________________________

### Lasso (L1) — λ = 0.1, 10, 100

**Resultados:** captura el gráfico de barras de coeficientes para cada λ.

![Imagen: lasso_lambda_0.1_coeffs](outputs/lasso_lambda_0.1_coeffs.png)

**RMSE Lasso (λ=0.1):** _________________

![Imagen: lasso_lambda_10_coeffs](outputs/lasso_lambda_10_coeffs.png)

**RMSE Lasso (λ=10):** _________________

![Imagen: lasso_lambda_100_coeffs](outputs/lasso_lambda_100_coeffs.png)

**RMSE Lasso (λ=100):** _________________

#### Análisis clave:

1. Identifica para qué valor de λ el modelo **Lasso** consigue **eliminar** (poner a cero) las 7 variables ruidosas, quedándose solo con las 3 importantes:

   **λ* = _________________**

2. Variables con coeficiente distinto de cero (modelo Lasso con λ*):

   _________________________________________________________________

### El dilema de la complejidad — λ demasiado alto (ej. 1000)

Prueba un valor extremo, por ejemplo λ = 1000, y registra qué ocurre.

![Imagen: lasso_lambda_1000_coeffs](outputs/lasso_lambda_1000_coeffs.png)

**RMSE (λ=1000):** _________________

#### Análisis:

Explica qué ocurre con el RMSE si aplicas un λ demasiado alto. ¿Estamos ante un caso de **sesgo alto** o **varianza alta**? Justifica.

_____________________________________________________________________________

_____________________________________________________________________________

_____________________________________________________________________________

---

## Reto — El "Cazador de Variables"

### Contexto

Imagina que cada variable incluida en tu modelo tiene un **coste de mantenimiento**. Tu objetivo es encontrar el modelo **más sencillo posible** (menos variables con coeficientes ≠ 0) que mantenga un RMSE inferior a un umbral:

**RMSE < _________________ (umbral del profesor)**

### Mejor solución encontrada (rellena):

- **λ = _________________**
- **#Variables activas = _________________**
- **RMSE = _________________**

### Variables seleccionadas (coeficiente ≠ 0):

_________________________________________________________________

![Imagen: reto_best_model_coeffs](outputs/reto_best_model_coeffs.png)

### Justificación técnica (6–10 líneas):

¿Por qué este modelo es el mejor compromiso entre simplicidad (coste) y rendimiento (RMSE)?

_____________________________________________________________________________

_____________________________________________________________________________

_____________________________________________________________________________

---

## Conclusiones finales

Resume en 8–12 líneas lo aprendido sobre:

- Estabilidad de K-Fold al variar K (K=2 vs K=10 vs LOOCV).
- Diferencias prácticas entre Ridge y Lasso (coeficientes, selección de variables).
- Relación entre λ grande y el sesgo/varianza.

_____________________________________________________________________________

_____________________________________________________________________________

_____________________________________________________________________________

_____________________________________________________________________________