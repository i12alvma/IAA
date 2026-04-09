# Práctica 5: Arquitectura de Redes Neuronales (MLP)

## Introducción al Aprendizaje Automático
**3º Ingeniería Informática - Curso 2025/2026**

---

## Objetivo
[Describe aquí el objetivo general de la práctica.]

---

## Material de partida
- Dataset sintético no lineal basado en `make_moons`.
- Dataset de dígitos escritos a mano (`load_digits`).
- Plantilla de código en Python con `scikit-learn`.

> Nota: el trabajo consiste en analizar resultados experimentales, no en implementar una red neuronal desde cero.

---

## Introducción
[Escribe aquí una introducción breve sobre por qué las redes neuronales son útiles en problemas no lineales y qué vas a estudiar en la práctica.]

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
[Resume aquí qué se pide en esta tarea y qué funciones vas a comparar.]

### Pregunta
¿Qué cambia al usar Sigmoide o ReLU con la misma arquitectura?

### Respuesta
[Escribe aquí tu respuesta.]

### Pregunta
¿Cuál de las dos funciones converge más rápido y qué diferencias se observan?

### Respuesta
[Escribe aquí tu respuesta.]

### Pregunta
¿Por qué puede ocurrir lo observado?

### Respuesta
[Escribe aquí tu respuesta.]

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
[Resume aquí qué se pide en esta tarea y cómo vas a buscar la arquitectura adecuada.]

### Pregunta
¿Qué arquitectura alcanza al menos un 95% de acierto?

### Respuesta
[Escribe aquí tu respuesta.]

### Pregunta
¿Es preferible una sola capa con muchas neuronas o dos capas con menos neuronas cada una?

### Respuesta
[Escribe aquí tu respuesta.]

### Pregunta
¿Qué compromiso hay entre simplicidad, capacidad de representación, dificultad de entrenamiento, interpretabilidad y rendimiento?

### Respuesta
[Escribe aquí tu respuesta.]

### Pregunta
¿Qué arquitectura final seleccionas para la entrega?

### Respuesta
[Escribe aquí tu respuesta.]

### Resultados
[Escribe aquí el resumen de los resultados obtenidos en esta tarea.]

| Modelo | Arquitectura | acc_train | acc_test | Iteraciones | Cumple 95% |
| --- | --- | --- | --- | --- | --- |
| MLP(20,) | [completar] | [completar] | [completar] | [completar] | [sí/no] |
| MLP(50,) | [completar] | [completar] | [completar] | [completar] | [sí/no] |
| MLP(100,) | [completar] | [completar] | [completar] | [completar] | [sí/no] |
| MLP(30, 15) | [completar] | [completar] | [completar] | [completar] | [sí/no] |

---

## Decisión final de diseño

### Descripción de la tarea
[Resume aquí la decisión de diseño que vas a proponer a partir de los resultados.]

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
[Redacta aquí una conclusión breve sobre el papel de la arquitectura, la activación y el compromiso entre simplicidad y rendimiento.]
