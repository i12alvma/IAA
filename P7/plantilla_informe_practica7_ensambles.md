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
- Se entrenó un modelo de GradientBoostingClassifier sobre el conjunto de entrenamiento.
- Se evaluó el rendimiento del modelo tanto en entrenamiento como en test.
- Se midió el tiempo de entrenamiento para compararlo con Random Forest.
- Se extrajeron las variables más importantes utilizando feature_importances_.

### Interpretación
¿En qué se diferencia conceptualmente el Boosting del Bagging?

La principal diferencia es que en Bagging (como en Random Forest) los modelos se entrenan de forma independiente y en paralelo, mientras que en Boosting se entrenan uno detrás de otro.

En Boosting, cada nuevo árbol intenta corregir los errores que han cometido los anteriores, centrándose más en las muestras que se han clasificado mal.
De esta forma, el modelo va mejorando poco a poco en cada iteración.

Por tanto, mientras que Random Forest reduce sobre todo la varianza combinando muchos modelos, Boosting también intenta reducir el sesgo ajustando el modelo de forma progresiva.

### Análisis

Variables más importantes según el modelo de Boosting

- worst radius
- worst concave points
- worst perimeter

Estas variables tienen bastante sentido en el contexto del problema, ya que están relacionadas con el tamaño y la forma del tumor.
Por ejemplo, worst radius y worst perimeter indican el tamaño máximo, que es un factor importante para diferenciar entre tumores benignos y malignos.

Además, worst concave points está relacionada con lo irregular que es la forma, que también es una característica relevante.

También es interesante que varias de las variables más importantes sean del tipo “worst”, lo que indica que el modelo se fija en los valores más extremos, que suelen estar asociados a casos más graves.

En general, esto muestra que el modelo no trata todas las variables por igual, sino que da más importancia a las que realmente ayudan a distinguir mejor entre las clases.

Además, el modelo obtiene una precisión muy alta tanto en entrenamiento como en test. Aunque en entrenamiento llega a ser perfecta, en test sigue siendo muy alta, por lo que no parece que haya un sobreajuste tan fuerte como en el árbol simple.

---

## Tarea 4: Comparación global de los modelos

### Qué se hizo
- Se compararon los tres modelos entrenados: árbol de decisión, Random Forest y Gradient Boosting.
- Se analizó el rendimiento en test de cada uno.
- Se comparó el comportamiento frente al sobreajuste.
- Se tuvo en cuenta el coste de entrenamiento aproximado.
- Se valoró la facilidad de interpretación de cada modelo.

### Cuestión
¿Por qué un conjunto de modelos puede ser mejor que un único modelo aparentemente muy potente?

Un conjunto de modelos suele ser mejor porque combina varias predicciones en lugar de depender de una sola decisión.
En un modelo individual, como un árbol de decisión, pequeños cambios en los datos pueden provocar resultados muy diferentes, lo que lo hace poco estable.

En cambio, en los métodos de ensamble, como Random Forest o Boosting, cada modelo aporta su propia “opinión”.
Al combinar estas decisiones ya sea mediante votación o corrigiendo errores, se consigue un resultado más robusto.

Además, el uso de múltiples modelos introduce diversidad, lo que ayuda a reducir errores y mejora la capacidad de generalización.
En conjunto, esto permite obtener mejores resultados en datos no vistos que usando un único modelo muy complejo.

### Reflexión

Los métodos de ensamble tienen varias ventajas, pero también algunos inconvenientes.

Por un lado, suelen ofrecer un mejor rendimiento y una mayor estabilidad que un modelo individual, como se observa en los resultados obtenidos.
Sin embargo, esto tiene un coste:

- Mayor coste computacional, especialmente en modelos como Boosting, que entrenan de forma secuencial.
- Menor interpretabilidad, ya que es mucho más difícil entender cómo toman decisiones muchos árboles juntos que un único árbol.
- Mayor complejidad a la hora de desplegar el modelo, sobre todo en sistemas con recursos limitados.

Por tanto, aunque los ensambles suelen ser más precisos, no siempre son la mejor opción dependiendo del contexto.

### Tabla comparativa final
| Modelo | Accuracy test | Sobreajuste | Coste de entrenamiento | Interpretabilidad |
|---|---:|---|---|---|
| Árbol simple | [0.9231] | [Alto] | [Bajo] | [Alto] |
| Random Forest | [0.9580] | [Bajo] | [Medio] | [Medio] |
| Gradient Boosting | [0.9580] | [Bajo] | [Alto] | [Bajo] |

---

## Reto: Pensando en el despliegue

### Qué se hizo
- Se evaluaron las características de coste y precisión de los tres modelos (árbol individual, Random Forest y Gradient Boosting) para un escenario con recursos limitados.
- Se decidió desplegar una versión ligera del árbol de decisión: se reentrenó limitando la profundidad (p. ej. max_depth=4) y aplicando poda para reducir el sobreajuste, y se consideró la cuantización del modelo para ahorrar memoria y acelerar la inferencia

### Cuestiones
- ¿Qué modelo elegirías?

    Eligiría un árbol de decisión con profundidad limitada, ya que aunque su precisión es algo menor que la de los ensambles, es mucho más ligero y fácil de interpretar.
    Además, al limitar la profundidad se reduce el riesgo de sobreajuste, lo que mejora la capacidad de generalización sin necesidad de recurrir a modelos más complejos. Esto es especialmente importante en un entorno con recursos limitados, donde el coste computacional y la memoria son factores críticos.

- ¿Qué ganarías con esa decisión?

    Con esta decisión se gana en eficiencia, tanto en términos de tiempo de entrenamiento como de velocidad de inferencia, lo cuál es crucial en un entorno con recursos limitados. También se gana en interpretabilidad, ya que un árbol de decisión es mucho más fácil de entender y explicar a usuarios no técnicos que un conjunto de modelos. Esto puede ser importante para la confianza y la adopción del modelo en aplicaciones sensibles, como el diagnóstico médico.

- ¿Qué estarías sacrificando a cambio?

    A cambio, se estaría sacrificando algo de precisión, ya que el árbol de decisión con profundidad limitada no puede capturar patrones tan complejos como los métodos de ensamble. 
    Sin embargo, en muchos casos esta pérdida de precisión puede ser aceptable si se compensa con una mayor eficiencia y facilidad de uso. 
    Además, al aplicar técnicas como la poda y la cuantización, se puede mitigar parte del sobreajuste y mejorar la capacidad de generalización del modelo, lo que ayuda a mantener un rendimiento razonable a pesar de la simplicidad del modelo elegido.

### Interpretación

    Se observa que la elección de un modelo no depende únicamente de la precisión obtenida. En problemas reales también es importante tener en cuenta otros factores, como el coste computacional, la latencia, la memoria disponible y la facilidad de mantenimiento. Aunque los métodos de ensamble suelen ofrecer mejores resultados, también implican un mayor consumo de recursos y una mayor complejidad de uso. Por ello, en determinados contextos puede ser preferible optar por un modelo más simple si ofrece un rendimiento suficientemente bueno y resulta más eficiente para su despliegue.

---

## Conclusión
[Redacta aquí una conclusión breve sobre las ventajas y limitaciones de los métodos de ensamble frente a un árbol de decisión individual.]

</div>