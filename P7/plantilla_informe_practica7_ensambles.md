<div style="text-align: justify;">

# Práctica 7: La Inteligencia Colectiva (Modelos de Ensambles)

## Introducción al Aprendizaje Automático
**3º Ingeniería Informática - Curso 2025/2026**

---

## Objetivo
Comprender por qué combinar múltiples modelos puede mejorar la capacidad de generalización frente a un clasificador individual. En particular, se estudiará el fenómeno del overfitting en árboles de decisión, el funcionamiento de estrategias de ensamble como Random Forest y Boosting, y la forma en que estos métodos logran reducir errores, estabilizar predicciones e identificar variables relevantes en un problema de clasificación real.

---

## Material de partida
Se proporciona:
- Un dataset real de clasificación, Breast Cancer Wisconsin, integrado en scikit-learn, por lo que no es necesario descargar ningún fichero externo.
- Una plantilla de código base en Python con scikit-learn y matplotlib, para facilitar la carga de datos, el entrenamiento de los modelos y la representación gráfica de resultados.
- Un problema de clasificación binaria con múltiples variables predictoras, apropiado para observar tanto el sobreajuste de un árbol individual como las ventajas de los métodos de ensamble.

---

## Introducción
En muchos problemas reales, una única decisión tomada por un único modelo puede ser frágil. Un árbol de decisión sin restricciones, por ejemplo, puede adaptarse con enorme precisión a los datos de entrenamiento, pero esa misma flexibilidad puede convertirse en una debilidad cuando el modelo se enfrenta a ejemplos nuevos. Este fenómeno se conoce como sobreajuste u overfitting, y constituye uno de los problemas centrales del aprendizaje automático.

Una estrategia clásica para reducir este problema consiste en combinar múltiples modelos en lugar de confiar en uno solo. Esta idea da lugar a los métodos de ensamble, donde varios clasificadores cooperan para producir una predicción final más robusta. En lugar de depender de un único árbol potencialmente inestable, se construye un conjunto de árboles que votan, corrigen errores o reparten la carga de aprendizaje.

En esta práctica se trabajará con tres enfoques progresivos: un árbol de decisión simple, que servirá como línea base; un Random Forest, que combina múltiples árboles entrenados sobre distintas muestras de los datos; y un modelo de Boosting, que entrena árboles secuencialmente para corregir errores previos.

El objetivo es comprender no solo cuál obtiene mejor precisión, sino también por qué se comportan de forma distinta y qué compromisos existen entre capacidad predictiva, coste de entrenamiento, interpretabilidad y robustez.

---

## Tarea 1: El árbol solitario

### Qué se hizo
- [Carga del dataset / explicación de qué se ha hecho aquí.]
- [División entre entrenamiento y prueba.]
- [Entrenamiento de un `DecisionTreeClassifier` sin limitar su profundidad.]
- [Evaluación del rendimiento en entrenamiento y prueba.]

### Cuestión
[Debes explicar por qué un árbol de decisión muy profundo puede alcanzar resultados excelentes en entrenamiento y, sin embargo, empeorar al evaluar sobre datos no vistos.

No basta con indicar que el modelo "memoriza". Debes razonar qué significa eso en términos de particiones del espacio de entrada, sensibilidad al ruido y capacidad de generalización.]

### Análisis
[Comenta si observas un comportamiento compatible con overfitting. En particular, interpreta una situación en la que el árbol obtenga una precisión muy alta, o incluso perfecta, en entrenamiento, pero claramente inferior en test.

El objetivo de esta parte es que entiendas que una alta precisión en entrenamiento no garantiza un buen modelo.]

---

## Tarea 2: Bagging y Random Forest

### Qué se hizo
- [Entrenamiento de un `RandomForestClassifier`.]
- [Experimento variando el número de árboles.]
- [Evaluación del rendimiento en el conjunto de prueba para cada configuración.]
- [Representación gráfica de la evolución del rendimiento frente al número de árboles.]

### Qué se interpreta
[Debes explicar por qué Random Forest suele generalizar mejor que un árbol individual.

Tu explicación debe relacionar, al menos de forma intuitiva, ideas como:
- el entrenamiento de múltiples árboles sobre subconjuntos distintos de ejemplos;
- la reducción de la varianza del modelo;
- y el efecto de la votación agregada sobre errores individuales.]

### Cuestión
[Analiza si llega un momento en el que añadir más árboles deja de mejorar claramente el resultado.

No se trata solo de indicar que la curva se estabiliza. Debes explicar por qué, a partir de cierto punto, el ensemble ya ha capturado casi toda la mejora que puede obtenerse con esta estrategia.]

### Variables más importantes en Random Forest
1. [variable 1]
2. [variable 2]
3. [variable 3]

[Comenta brevemente si estas variables te parecen razonables dentro del problema.]

---

## Tarea 3: El poder del Boosting

### Qué se hizo
- [Entrenamiento de un `GradientBoostingClassifier` o, alternativamente, de un `AdaBoostClassifier`.]
- [Evaluación del rendimiento sobre el conjunto de prueba.]
- [Comparación del tiempo de entrenamiento respecto a Random Forest.]
- [Extracción de la importancia de variables del modelo entrenado.]

### Interpretación
[Debes explicar en qué se diferencia conceptualmente el Boosting del Bagging.

En particular, debes comentar que en Boosting los modelos no son independientes entre sí, sino que cada nuevo árbol se construye para concentrarse en los errores o residuos que aún no han sido bien modelados.]

### Análisis
[A partir de `feature_importances_`, identifica las 3 variables que el modelo considera más relevantes y comenta si te parece razonable que esas características influyan en la clasificación.

No es necesario realizar una interpretación médica exhaustiva del dataset, pero sí debes mostrar que entiendes que el modelo no trata todas las variables por igual y que algunas aportan más capacidad discriminativa que otras.]

---

## Tarea 4: Comparación global de los modelos

### Qué se hizo
- [Comparación entre el árbol simple, el Random Forest y el modelo de Boosting.]
- [Análisis del rendimiento en test.]
- [Comparación del comportamiento frente al sobreajuste.]
- [Comparación del coste de entrenamiento aproximado.]
- [Comparación de la facilidad de interpretación.]

### Cuestión
[Debes responder razonadamente a esta idea central:
¿Por qué un conjunto de modelos puede ser mejor que un único modelo aparentemente muy potente?

Tu respuesta debe mostrar que entiendes el papel de la diversidad, la agregación de decisiones y la reducción del error de generalización.]

### Reflexión
[Comenta también qué inconvenientes presentan los métodos de ensamble. Por ejemplo, puedes discutir aspectos como:
- el mayor coste computacional;
- la pérdida de interpretabilidad frente a un árbol individual;
- o la dificultad de desplegar modelos grandes en entornos limitados.]

### Tabla comparativa final
| Modelo | Accuracy test | Sobreajuste | Coste de entrenamiento | Interpretabilidad |
|---|---:|---|---|---|
| Árbol simple | [completar] | [completar] | [completar] | [completar] |
| Random Forest | [completar] | [completar] | [completar] | [completar] |
| Gradient Boosting | [completar] | [completar] | [completar] | [completar] |

---

## Reto: Pensando en el despliegue

### Qué se hizo
[Debes elegir cuál de los tres modelos desplegarías en un contexto con recursos limitados, como un dispositivo móvil o un sistema con poca memoria y CPU.]

### Cuestiones
- ¿Qué modelo elegirías?
- ¿Qué ganarías con esa decisión?
- ¿Qué estarías sacrificando a cambio?

### Interpretación
[El objetivo de esta parte es que entiendas que elegir un modelo no depende solo de la precisión. En problemas reales también importan el coste computacional, la latencia, la memoria disponible y la facilidad de mantenimiento.]

---

## Conclusión
[Redacta aquí una conclusión breve sobre las ventajas y limitaciones de los métodos de ensamble frente a un árbol de decisión individual.]

</div>