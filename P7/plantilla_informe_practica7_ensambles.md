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
- Se cargó el dataset Breast Cancer Wisconsin incluido en scikit-learn y se separaron las variables predictoras de la etiqueta objetivo.
- Se dividieron los datos en entrenamiento y prueba con una partición estratificada para mantener la proporción de clases.
- Se entrenó un árbol de decisión sin limitar la profundidad, de forma que el modelo pudiera crecer libremente.
- Se evaluó el rendimiento del árbol en entrenamiento y en test para comprobar si aparecía sobreajuste.

### Cuestión
<strong>¿Por qué un árbol de decisión muy profundo puede ajustar muy bien los datos de entrenamiento pero rendir peor en test?</strong>

Un árbol de decisión muy profundo puede ajustar muy bien los datos de entrenamiento pero rendir peor en test debido a que al crecer sin restricciones el árbol crea particiones muy específicas del espacio de entrada y termina aprendiendo detalles del entrenamiento, incluso ruido o casos particulares. 
Esto le permite acertar mucho sobre los datos vistos, pero hace que generalice peor cuando aparecen ejemplos nuevos.

### Análisis
<strong>¿Se observa un comportamiento compatible con overfitting en este árbol?</strong>

Sí, porque el árbol obtiene una precisión muy alta en entrenamiento, incluso perfecta, pero baja claramente en test. 
Esta diferencia indica que el modelo se ha ajustado demasiado a los datos de entrenamiento y no ha aprendido patrones suficientemente generales para funcionar igual de bien con datos no vistos.

---

## Tarea 2: Bagging y Random Forest

### Qué se hizo
- Se entrenó un Random Forest sobre el conjunto de entrenamiento.
- Se repitió el experimento variando el número de árboles para comparar cómo cambia el rendimiento.
- Se evaluó la precisión en test para cada configuración.
- Se representó la evolución del accuracy frente al número de árboles para comprobar si el modelo se estabiliza.

### Qué se interpreta
<strong>¿Por qué Random Forest suele generalizar mejor que un árbol individual?</strong>

Random Forest suele generalizar mejor porque combina varios árboles entrenados sobre subconjuntos distintos de los datos. 
Cada árbol comete errores diferentes, pero al promediar o votar entre todos se reducen las decisiones demasiado inestables de un único árbol. Así disminuye la varianza del modelo y mejora su capacidad de generalizar a ejemplos nuevos.

### Cuestión
<strong>¿Llega un punto en el que añadir más árboles en Random Forest deja de mejorar claramente el rendimiento?    </strong>

Sí. En este experimento se observa que la mejora es notable al pasar de pocos árboles a una cantidad intermedia, pero a partir de cierto punto la curva de accuracy en test se estabiliza y las ganancias son cada vez menores. 
Esto ocurre porque el ensamble ya ha reducido gran parte de la varianza del modelo: al incorporar más árboles, las nuevas predicciones aportan información muy parecida a la que ya tenía el conjunto. 
Por tanto, seguir aumentando n_estimators incrementa el coste de entrenamiento y de inferencia, pero no produce una mejora proporcional en la capacidad de generalización.

### Variables más importantes en Random Forest
1. worst area
2. worst concave points
3. mean concave points

Estas variables me parecen razonables dentro del problema, porque están relacionadas con el tamaño y la forma de la lesión, que suelen ser rasgos muy informativos para diferenciar entre casos benignos y malignos.
Además, el hecho de que aparezcan medidas de “worst” (valores extremos) sugiere que el modelo está captando patrones de mayor severidad morfológica, lo cual tiene sentido en una tarea de clasificación de cáncer de mama.

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