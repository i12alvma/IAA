# Cuaderno de Laboratorio — Práctica 3: Metodología de Evaluación y Rigor Científico

## Objetivo
Aprender a realizar evaluaciones honestas en condiciones difíciles, especialmente con datos desbalanceados, y detectar errores metodológicos graves como la fuga de datos (*data leakage*).

## Material de partida
Se parte de:
- Un dataset sintético de **1000 pacientes**.
- Una clase positiva muy minoritaria (**2 %**).

---

## Tarea 1 — La lotería de la partición aleatoria

### Configuración común
- Modelo usado: LogisticRegression (class_weight="balanced")
- Métricas usadas: Accuracy, F1-score
- Semilla(s): 0, 7, 21, 42, 99

### Registro de resultados
| Ejecución | Positivos en test |
|---|---:|
| 1 | 4 |
| 2 | 3 |
| 3 | 3 |
| 4 | 9 |
| 5 | 7 |

### Análisis
- ¿Ha cambiado mucho el número de positivos en test entre ejecuciones?
  
  Sí, cambia bastante.
  Aunque en teoría deberían salir unos 4 positivos en el test, en la práctica han salido entre 3 y 9, lo cual es bastante diferencia.

  Esto pasa porque la partición es aleatoria y hay muy pocos casos positivos, así que dependiendo de cómo caigan, el resultado cambia bastante.

- ¿Sería fiable la evaluación si solo hubiera 0 o 1 positivos en test?
  
  No, no sería fiable.

  Si hay:
  
  0 positivos, no se puede evaluar bien el modelo porque no hay casos de la clase importante.

  1 positivo, todo depende de si acierta o falla ese único caso, lo cual no es representativo.

---

## Tarea 2 — Partición estratificada

### Registro por folds
| Fold | Positivos en test | Tamaño del fold | Proporción clase 1 |
|---|---:|---:|---:|
| 1 | 4 | 200 | 0.020 |
| 2 | 4 | 200 | 0.020 |
| 3 | 4 | 200 | 0.020 |
| 4 | 4 | 200 | 0.020 |
| 5 | 4 | 200 | 0.020 |

### Análisis
Explica por qué `StratifiedKFold` es más adecuado que una partición aleatoria simple en este problema.

StratifiedKFold es más adecuado en este problema porque los datos están muy desbalanceados (solo un 2 % de la clase positiva).

Con una partición aleatoria simple, puede pasar que en algunos conjuntos de test haya muy pocos positivos o incluso ninguno. Esto hace que la evaluación no sea fiable, ya que el modelo no se está probando correctamente sobre la clase importante.

---

## Tarea 3 — Comparativa de la varianza

Se realizaron 10 iteraciones de validación cruzada con KFold y 10 iteraciones con StratifiedKFold, calculando en ambos casos las métricas Accuracy y F1-score, junto con su desviación típica.

### Resumen de resultados
| Método | Accuracy media | std(Accuracy) | F1 media | std(F1) |
|---|---:|---:|---:|---:|
| KFold | 0.5780 | 0.0665 | 0.0286 | 0.0377 |
| StratifiedKFold | 0.5850 | 0.0482 | 0.0440 | 0.0256 |

### Interpretación
- ¿Qué método produce métricas más estables?

    El método que produce métricas más estables es `StratifiedKFold`, ya que presenta una desviación típica menor tanto en `Accuracy` como en `F1-score` que `KFold`. 
    Esto indica que los resultados dependen menos del azar en la partición de los datos.

- ¿Dónde se nota más la mejora: en Accuracy o en F1?

    La mejora se nota más en `F1-score`, porque además de aumentar su valor medio de 0.0286 a 0.0440, también reduce bastante su desviación típica de 0.0377 a 0.0256. En este problema, al estar los datos desbalanceados, `F1-score` es una métrica más representativa que `Accuracy` para evaluar el comportamiento del modelo sobre la clase minoritaria.
---

## Tarea 4 — Detección de data leakage

### Comparativa
| Escenario | Accuracy | F1 |
|---|---:|---:|
| Sin variable trampa | 0.6550 | 0.1039 |
| Con variable trampa | 1.0000 | 1.0000 |

### Preguntas
- ¿Qué variable sospechas que está filtrando información del objetivo?

  Se sospecha que es la variable `ID_Hospital_Filtro`, ya que su correlación con la clase es casi perfecta, lo que se confirma al obtener Accuracy y F1 de 1.0000 cuando se incluye en el entrenamiento, frente a 0.6550 y 0.1039 sin ella.

- ¿Por qué esa variable invalida la evaluación?

  Se debe a que contiene información directamente derivada del objetivo (`Clase`). 
  Con lo que el modelo no aprende ningún patrón real del problema, sino que simplemente lee la etiqueta a través de esa variable. Las métricas resultantes (100 %) no reflejan capacidad de generalización, sino explotación de una fuga de datos.

- ¿Qué diferencia conceptual hay entre un modelo "bueno" y un modelo "contaminado" por leakage?

  Por un lado, un buen modelo aprende patrones genuinos de las variables predictoras que se generalizan a datos nuevos. 
  Por otro lado, un modelo contaminado por leakage obtiene métricas perfectas en evaluación pero falla en producción, porque se apoya en información que en un caso real no estaría disponible en el momento de predecir.

---

## Reto — Auditoría metodológica
Redacta un pequeño informe (6–10 líneas) respondiendo a estas cuestiones:
1. ¿Por qué una partición aleatoria simple puede ser peligrosa en problemas desbalanceados?

    Una partición aleatoria simple puede ser peligrosa en problemas desbalanceados porque no garantiza que la clase minoritaria esté bien representada en el conjunto de test. Esto puede hacer que, por puro azar, haya muy pocos o incluso ningún caso positivo, lo que provoca evaluaciones poco fiables y muy dependientes de la partición concreta.

2. ¿Qué aporta la validación estratificada?

    La validación estratificada soluciona este problema manteniendo la misma proporción de clases en cada partición que en el dataset original. De esta forma, todos los folds contienen ejemplos de la clase minoritaria, lo que permite evaluar el modelo de forma más justa.

3. ¿Por qué el *data leakage* puede hacer inútiles las conclusiones de un experimento?

    El *data leakage* hace inútiles las conclusiones porque el modelo utiliza información que no debería conocer. Así, las métricas salen artificialmente altas y no reflejan el rendimiento real. En este caso, al incluir `ID_Hospital_Filtro`, se obtienen resultados perfectos, pero eso no significa que el modelo haya aprendido bien, sino que está aprovechando una fuga de información.

4. ¿Qué buenas prácticas metodológicas seguirías en futuros experimentos?

    En futuros experimentos revisaría bien las variables para detectar posibles fugas de información, separaría correctamente los datos de entrenamiento y prueba y usaría validación estratificada en problemas desbalanceados. Además, no me fijaría solo en `Accuracy`, sino también en métricas como `F1-score`, y desconfiaría de resultados demasiado buenos sin una explicación clara.

---

## Conclusión final
Resume qué has aprendido sobre evaluación rigurosa, estabilidad de métricas y detección de errores metodológicos.

  En esta práctica se ha comprobado que evaluar un modelo de forma rigurosa es fundamental, especialmente cuando los datos están desbalanceados. La validación estratificada ofrece resultados más estables que una validación cruzada normal, sobre todo en `F1-score`, que representa mejor el comportamiento sobre la clase minoritaria. También se ha visto que el *data leakage* puede inflar artificialmente las métricas hasta valores perfectos, dando una imagen falsa del rendimiento del modelo. Por tanto, no basta con obtener buenas métricas: también es necesario asegurarse de que la evaluación sea metodológicamente correcta.