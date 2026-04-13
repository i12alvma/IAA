# Práctica 5: Arquitectura de Redes Neuronales (MLP)

## Introducción al Aprendizaje Automático
**3º Ingeniería Informática - Curso 2025/2026**

---

## Objetivo
Comprender cómo la arquitectura de una red neuronal afecta a su capacidad de representación y generalización. En particular, se estudiará por qué un modelo lineal falla en problemas no lineales, cómo influye el número de neuronas y capas ocultas en la frontera de decisión, y qué papel juegan las funciones de activación en el aprendizaje de un Perceptrón Multicapa.

---

## Material de partida
- Dataset sintético no lineal basado en `make_moons`.
- Dataset de dígitos escritos a mano (`load_digits`).
- Plantilla de código en Python con `scikit-learn`.

> Nota: el trabajo consiste en analizar resultados experimentales, no en implementar una red neuronal desde cero.

---

## Introducción
En prácticas anteriores se ha trabajado con modelos lineales o con clasificadores cuya capacidad de decisión era relativamente limitada. Sin embargo, muchos problemas reales presentan fronteras entre clases que no pueden representarse mediante una línea recta o un hiperplano simple.

Las redes neuronales permiten superar esta limitación mediante la combinación de capas y funciones de activación no lineales. Gracias a ello, un Perceptrón Multicapa (MLP) puede aprender regiones de decisión más complejas y adaptarse a problemas donde los modelos lineales fracasan.

Pero esta mayor flexibilidad también introduce nuevas preguntas de ingeniería:
- ¿Cuántas capas ocultas hacen falta?
- ¿Cuántas neuronas conviene usar?
- ¿Qué función de activación resulta más adecuada?
- ¿Cuándo un modelo empieza a ajustarse demasiado a los datos y pierde capacidad de generalización?

En esta práctica se va a explorar estas cuestiones de forma experimental, observando cómo cambia el comportamiento del modelo al modificar su arquitectura.

---

## Tarea 1: El problema de la no linealidad

### Descripción de la tarea
[Resume aquí qué se pide en esta tarea y qué vas a analizar.]

### Pregunta
¿Por qué falla un modelo sin capas ocultas en este problema?

### Respuesta
[Escribe aquí tu respuesta.]

### Pregunta
¿Qué relación hay entre la geometría de los datos, la frontera de decisión y la capacidad expresiva del modelo?

### Respuesta
[Escribe aquí tu respuesta.]

### Resultados
[Escribe aquí el resumen de los resultados obtenidos en esta tarea.]

- Accuracy en entrenamiento: [completar]
- Accuracy en prueba: [completar]
- Iteraciones: [completar]

### Figuras
- Matriz de confusión: [insertar figura]
- Frontera de decisión: [insertar figura]

---

## Tarea 2: Diseñando la capa oculta

### Descripción de la tarea
[Resume aquí qué se pide en esta tarea y qué vas a comparar.]

### Pregunta
¿Qué ocurre al aumentar el número de neuronas en una sola capa oculta?

### Respuesta
[Escribe aquí tu respuesta.]

### Pregunta
¿Cómo cambia la frontera de decisión al aumentar el número de neuronas ocultas?

### Respuesta
[Escribe aquí tu respuesta.]

### Pregunta
¿Hay underfitting u overfitting en estas configuraciones?

### Respuesta
[Escribe aquí tu respuesta.]

### Resultados
[Escribe aquí el resumen de los resultados obtenidos en esta tarea.]

| Modelo | Arquitectura | acc_train | acc_test | Iteraciones |
| --- | --- | --- | --- | --- |
| MLP(2,) | [completar] | [completar] | [completar] | [completar] |
| MLP(5,) | [completar] | [completar] | [completar] | [completar] |
| MLP(20,) | [completar] | [completar] | [completar] | [completar] |

### Figuras
- Frontera de decisión con 2 neuronas: [insertar figura]
- Frontera de decisión con 5 neuronas: [insertar figura]
- Frontera de decisión con 20 neuronas: [insertar figura]

---

## Tarea 3: Funciones de activación

### Descripción de la tarea
En esta parte se estudiará cómo influye la función de activación de las capas ocultas. Se elegirá una arquitectura fija y se entrenará dos veces: una con Sigmoide y otra con ReLU. Luego se compararán ambos modelos en términos de evolución del entrenamiento, iteraciones necesarias, rendimiento final y aspecto de la frontera de decisión.

### Pregunta
¿Cuál de las dos funciones converge más rápido y qué diferencias se observan?

### Respuesta
[Escribe aquí tu respuesta. Considera: iteraciones necesarias, estabilidad del entrenamiento, accuracy final, forma de la frontera.]

### Pregunta
¿Por qué puede ocurrir lo observado?

### Respuesta
[Escribe aquí tu respuesta. Puedes considerar: saturación de la sigmoide, facilidad de optimización, capacidad de introducir no linealidad, efecto práctico sobre el aprendizaje.]

### Resultados
[Escribe aquí el resumen de los resultados obtenidos en esta tarea.]

| Activación | acc_test | Iteraciones | loss_final |
| --- | --- | --- | --- |
| logistic | [completar] | [completar] | [completar] |
| relu | [completar] | [completar] | [completar] |

### Figuras
- Frontera con logistic: [insertar figura]
- Pérdida con logistic: [insertar figura]
- Frontera con ReLU: [insertar figura]
- Pérdida con ReLU: [insertar figura]

---

## Tarea 4: El reto de la caja negra

### Descripción de la tarea
Se trabajará con un problema más realista: clasificación de dígitos escritos a mano. Se cargará el dataset de dígitos y se diseñarán distintas arquitecturas MLP variando el número de capas ocultas y neuronas por capa, buscando una configuración que alcance al menos un 95% de acierto.

No se trata de probar combinaciones al azar: se justificará el proceso seguido, explicar qué arquitecturas se probaron, cuáles funcionaron mejor o peor, y cuál se considera la configuración mínima razonable.

### Pregunta
¿Qué arquitectura alcanza al menos un 95% de acierto? Justifica el proceso experimental seguido.

### Respuesta
[Escribe aquí tu respuesta. Incluye: qué arquitecturas probaste, cuáles funcionaron mejor/peor, cuál es la mínima razonable.]

### Pregunta
¿Es preferible una sola capa con muchas neuronas o dos capas con menos neuronas cada una?

### Respuesta
[Escribe aquí tu respuesta basándote en los resultados obtenidos.]

### Pregunta
¿Qué compromiso existe entre simplicidad, capacidad de representación, dificultad de entrenamiento, interpretabilidad y rendimiento?

### Respuesta
[Escribe aquí tu respuesta. Apóyate en los resultados y reflexiona sobre cada aspecto del compromiso.]

### Arquitectura final seleccionada
[Especifica aquí la arquitectura que recomiendas y justifica por qué.]

### Resultados
[Escribe aquí el resumen de los resultados obtenidos en esta tarea.]

| Modelo | Arquitectura | acc_train | acc_test | Iteraciones | Cumple 95% |
| --- | --- | --- | --- | --- | --- |
| MLP(20,) | [completar] | [completar] | [completar] | [completar] | [sí/no] |
| MLP(50,) | [completar] | [completar] | [completar] | [completar] | [sí/no] |
| MLP(100,) | [completar] | [completar] | [completar] | [completar] | [sí/no] |
| MLP(30, 15) | [completar] | [completar] | [completar] | [completar] | [sí/no] |

---

## Reto: Decisión final de diseño

### Descripción de la tarea
Imagina que formas parte de un equipo técnico que necesita desplegar un clasificador basado en redes neuronales para un problema real. Debes redactar una breve propuesta respondiendo a las preguntas. Se valorará especialmente la capacidad de conectar los resultados experimentales con una decisión de diseño razonada.

### Pregunta
¿Cuándo tiene sentido usar una red sin capas ocultas y cuándo no?

### Respuesta
[Escribe aquí tu respuesta.]

### Pregunta
¿Qué arquitectura recomendarías para un problema no lineal sencillo?

### Respuesta
[Escribe aquí tu respuesta.]

### Pregunta
¿Qué ventajas e inconvenientes observas al aumentar el número de neuronas?

### Respuesta
[Escribe aquí tu respuesta.]

### Pregunta
¿Qué función de activación elegirías en un problema general de clasificación y por qué?

### Respuesta
[Escribe aquí tu respuesta.]

### Pregunta
¿Qué criterio usarías para decidir si una arquitectura es suficientemente buena sin hacerla innecesariamente compleja?

### Respuesta
[Escribe aquí tu respuesta.]

---

## Conclusión
[Redacta aquí una conclusión breve sobre el papel de la arquitectura, la función de activación y el compromiso entre simplicidad y rendimiento. Reflexiona sobre por qué una red neuronal no es una caja mágica, sino una familia de modelos cuyo comportamiento depende de decisiones concretas de diseño: arquitectura, activación y equilibrio entre ajuste y generalización.]
