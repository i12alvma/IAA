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

- **Modelo usado:** Regresión lineal estándar (`LinearRegression`, sin regularización)
- **Métrica:** RMSE (raíz del error cuadrático medio)
- **Semilla(s) / aleatoriedad:** `KFold(shuffle=True)` con semillas distintas en K=2 (`1, 7, 21, 42, 99`) y semilla fija `42` en K=10 y K=100

### Hipótesis inicial (antes de ejecutar)

¿Crees que el RMSE será estable al cambiar K? ¿Por qué?

Espero que con **K pequeño (K=2)** el RMSE sea más inestable porque cada fold usa muy pocos datos para entrenar y el resultado depende mucho de cómo se haga la partición.

Con **K mayor (K=10 o K=100)** debería ser más estable (menor varianza de la estimación), aunque el coste computacional aumentará al tener que entrenar más veces.

### Experimento A — K = 2 (Validación simple): variabilidad entre ejecuciones

Ejecuta el modelo **varias veces** con K=2 (por ejemplo 5–10) y registra el RMSE.

![Imagen: kfold_K2_rmse_runs](/P2/P2_regularizacion/outputs/kfold_K2_rmse_runs.png)

#### Registro de resultados (rellena):

| Ejecución | RMSE |
|-----------|------|
| 1 (seed=1) | 5.2015 |
| 2 (seed=7) | 5.2142 |
| 3 (seed=21) | 5.1064 |
| 4 (seed=42) | 5.2059 |
| 5 (seed=99) | 5.0154 |

#### Análisis:

- ¿Por qué el error varía tanto entre ejecuciones?
- ¿Es fiable esta métrica (con K=2) para tomar una decisión técnica?

Con K=2, cada división deja solo la mitad de los datos para entrenamiento y la otra mitad para validación. Si por azar una partición concentra más casos "fáciles" o "difíciles", el RMSE cambia bastante.

Por eso, K=2 sirve como referencia rápida, pero **no es la opción más fiable** para decisiones técnicas finas: tiene más varianza y depende mucho de la semilla.

### Experimento B — K = 10 (estándar): RMSE más estable

Ejecuta el modelo con K=10 y anota el RMSE.

![Imagen: kfold_K10_rmse](/P2/P2_regularizacion/outputs/kfold_K10_rmse.png)

**RMSE observado (K=10):** 5.0101

#### Análisis:

Describe si el resultado parece más estable que en K=2 y por qué.

Sí, se observa más estabilidad que en K=2. Al promediar el error sobre 10 particiones, el RMSE depende menos de una división concreta y representa mejor el comportamiento medio del modelo.

Es un buen compromiso entre estabilidad estadística y tiempo de ejecución, por eso se usa como valor estándar en muchos problemas.

### Experimento C — K = 100 (Leave-One-Out): coste vs. precisión

Ejecuta el modelo con K=100 (LOOCV) y observa el RMSE y el tiempo.

![Imagen: kfold_K100_loocv_rmse](/P2/P2_regularizacion/outputs/kfold_K100_loocv_rmse.png)

**RMSE observado (K=100):** 4.9929 
**Tiempo aproximado:** 111.28 ms (frente a 11.82 ms en K=10)

#### Análisis:

Compara el coste computacional frente a K=10. ¿Merece la pena el tiempo extra para mejorar la precisión obtenida?

El coste sube casi 10 veces respecto a K=10 (de ~11.82 ms a ~111.28 ms), porque hay que entrenar 100 modelos en lugar de 10.

La mejora de RMSE es muy pequeña (de 5.0101 a 4.9929), así que en este caso **normalmente no compensa** el tiempo extra salvo que necesitemos máxima precisión en la estimación.

### Resumen comparativo (K = 2 vs 10 vs 100)

| K | RMSE | Comentario de estabilidad/coste |
|---|------|----------------------------------|
| 2 | 5.01–5.21 (según semilla) | Muy variable, rápido, alta dependencia de la partición |
| 10 | 5.0101 | Estable y coste moderado; buen compromiso práctico |
| 100 | 4.9929 | Muy estable, pero coste alto para una mejora marginal |

#### Conclusión Tarea 1 (3–6 líneas):

¿Cuál elegirías para este problema y por qué?

Elegiría **K=10** para este problema.

Con K=2 la métrica cambia bastante entre ejecuciones, así que la decisión puede depender demasiado de la semilla. Con K=100 el RMSE mejora muy poco respecto a K=10, pero el tiempo aumenta de forma notable.

Por tanto, K=10 ofrece el mejor equilibrio entre **fiabilidad del RMSE** y **coste computacional**.

---

## Tarea 2 — Regularización Ridge vs. Lasso (K = 10)

### Configuración

Fijamos K = 10 y comparamos:

- **Ridge Regression** (penalización L2)
- **Lasso Regression** (penalización L1)

Parámetros del script (`01_regularizacion_cv.py`):
- `metodo`: `"ridge"` o `"lasso"`
- `valor_lambda`: intensidad de regularización
- `k_folds`: número de particiones de validación cruzada

### Ridge (L2) — λ = 0.1, 10, 100

**Resultados:** captura el gráfico de barras de coeficientes para cada λ.

![Imagen: ridge_lambda_0.1_K10_coeffs](/P2/P2_regularizacion/outputs/ridge_lambda_0.1_K10_coeffs.png)

**RMSE Ridge (λ=0.1):** 5.0099

![Imagen: ridge_lambda_10_K10_coeffs](/P2/P2_regularizacion/outputs/ridge_lambda_10_K10_coeffs.png)

**RMSE Ridge (λ=10):** 5.9613

![Imagen: ridge_lambda_100_K10_coeffs](/P2/P2_regularizacion/outputs/ridge_lambda_100_K10_coeffs.png)

**RMSE Ridge (λ=100):** 22.5189

#### Análisis:

¿Llegan a valer **cero** los coeficientes de las variables ruidosas (`Var_4` a `Var_10`)? Explica qué observas y por qué pasa (o no pasa).

No. En Ridge (L2) los coeficientes se **encogen**, pero en general no se vuelven exactamente cero.

Al aumentar λ (0.1 → 10 → 100), los pesos se reducen y el modelo se vuelve más rígido; por eso termina subiendo el RMSE cuando la penalización es demasiado fuerte.

Ridge mejora estabilidad, pero no hace selección dura de variables: mantiene todas activas con pesos pequeños.

### Lasso (L1) — λ = 0.1, 10, 100

**Resultados:** captura el gráfico de barras de coeficientes para cada λ.

![Imagen: lasso_lambda_0.1_K10_coeffs](/P2/P2_regularizacion/outputs/lasso_lambda_0.1_K10_coeffs.png)

**RMSE Lasso (λ=0.1):** 4.9990

![Imagen: lasso_lambda_10_K10_coeffs](/P2/P2_regularizacion/outputs/lasso_lambda_10_K10_coeffs.png)

**RMSE Lasso (λ=10):** 17.5753

![Imagen: lasso_lambda_100_K10_coeffs](/P2/P2_regularizacion/outputs/lasso_lambda_100_K10_coeffs.png)

**RMSE Lasso (λ=100):** 62.9825

#### Análisis clave:

1. Identifica para qué valor de λ el modelo **Lasso** consigue **eliminar** (poner a cero) las 7 variables ruidosas, quedándose solo con las 3 importantes:

   **λ* = 10**

2. Variables con coeficiente distinto de cero (modelo Lasso con λ*):

   `Var_0`, `Var_1`, `Var_2`

### El dilema de la complejidad — λ demasiado alto (ej. 1000)

Prueba un valor extremo, por ejemplo λ = 1000, y registra qué ocurre.

![Imagen: lasso_lambda_1000_K10_coeffs](/P2/P2_regularizacion/outputs/lasso_lambda_1000_K10_coeffs.png)

**RMSE (λ=1000):** 62.9825

#### Análisis:

Explica qué ocurre con el RMSE si aplicas un λ demasiado alto. ¿Estamos ante un caso de **sesgo alto** o **varianza alta**? Justifica.

Con λ=1000 (igual que con λ=100 en este caso), Lasso anula prácticamente todos los coeficientes y el modelo queda sin capacidad predictiva útil.

El RMSE se dispara porque el modelo es excesivamente simple: no capta la relación entre variables y precio.

Esto es un caso claro de **sesgo alto** (underfitting), no de varianza alta.

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

   (Versión Larga)
Ridge y Lasso son dos formas de regularizar modelos lineales, pero se comportan distinto en la práctica.
Ridge (L2) reduce el tamaño de todos los coeficientes de forma suave, sin anularlos normalmente.
Por eso, con Ridge casi todas las variables siguen activas, aunque con menor peso.
Esto ayuda cuando hay colinealidad o ruido moderado, porque estabiliza el modelo sin “borrar” información.
Lasso (L1), en cambio, sí puede llevar coeficientes exactamente a cero.
Ese efecto convierte a Lasso en un método de selección automática de variables.
En términos de interpretabilidad, Lasso suele dar modelos más simples y fáciles de explicar.
El precio es que, si la penalización es alta, puede eliminar demasiadas variables y empeorar mucho el RMSE.
Ridge suele ser más estable en rendimiento cuando muchas variables aportan algo, aunque sea poco.
En resumen: Ridge prioriza estabilidad con todos los predictores; Lasso prioriza simplicidad y selección de atributos.
   
   (Version Breve)
   En la práctica, **Ridge (L2)** reduce todos los coeficientes de forma suave, pero casi nunca los lleva exactamente a cero; por eso mejora estabilidad cuando hay ruido o colinealidad, manteniendo todas las variables en el modelo. En cambio, **Lasso (L1)** sí puede anular coeficientes, actuando como método de **selección de variables** y dejando un modelo más simple e interpretable. La contrapartida es que, si λ es demasiado alto, Lasso puede eliminar demasiada información útil y degradar fuertemente el RMSE.

- Relación entre λ grande y el sesgo/varianza.
