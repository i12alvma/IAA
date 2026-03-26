# Práctica 5: Arquitectura de Redes Neuronales (MLP)

## Introducción al Aprendizaje Automático
**3º Ingeniería Informática - Curso 2025/2026**

---

## Objetivo
Comprender cómo la arquitectura de una red neuronal afecta a su capacidad de representación y generalización. Se estudiará por qué un modelo lineal falla en problemas no lineales, cómo influye el número de neuronas y capas ocultas en la frontera de decisión, y qué papel juegan las funciones de activación en el aprendizaje de un Perceptrón Multicapa.

---

## Material de partida
- Dataset sintético no lineal (ej: `make_moons`).
- Dataset de dígitos escritos a mano (versión simplificada de MNIST).
- Plantilla de código base en Python con scikit-learn.

> **Nota:** El objetivo es analizar cómo cambian los resultados al modificar la arquitectura del modelo, no programar una red desde cero.

---

## Introducción
En prácticas anteriores has trabajado con modelos lineales o con clasificadores cuya capacidad de decisión era relativamente limitada. Sin embargo, muchos problemas reales presentan fronteras entre clases que no pueden representarse mediante una línea recta o un hiperplano simple.

Las redes neuronales permiten superar esta limitación mediante la combinación de capas y funciones de activación no lineales. Gracias a ello, un Perceptrón Multicapa (MLP) puede aprender regiones de decisión más complejas y adaptarse a problemas donde los modelos lineales fracasan.

Pero esta mayor flexibilidad también introduce nuevas preguntas de ingeniería:
- ¿Cuántas capas ocultas hacen falta?
- ¿Cuántas neuronas conviene usar?
- ¿Qué función de activación resulta más adecuada?
- ¿Cuándo un modelo empieza a ajustarse demasiado a los datos y pierde capacidad de generalización?

En esta práctica vas a explorar estas cuestiones de forma experimental, observando cómo cambia el comportamiento del modelo al modificar su arquitectura.

---

## Tarea 1: El problema de la no linealidad

### Qué debes hacer
- Genera o carga un dataset no lineal de clasificación binaria.
- Entrena un Perceptrón simple (sin capas ocultas).
- Evalúa su rendimiento sobre los datos.
- Representa gráficamente:
  - Los puntos del dataset.
  - La frontera de decisión aprendida por el modelo.

### Cuestión
Explica por qué este modelo falla en este problema. Relaciónalo con la naturaleza del modelo: ¿por qué una combinación lineal de las variables de entrada no es suficiente para separar correctamente las clases?

### Reflexión
Comenta la relación entre:
- La forma geométrica de los datos.
- La forma de la frontera de decisión.
- La capacidad expresiva del modelo.

---

## Tarea 2: Diseñando la capa oculta

### Qué debes hacer
- Entrena varios MLP con una sola capa oculta.
- Prueba al menos estas configuraciones:
  - 2 neuronas
  - 5 neuronas
  - 20 neuronas
- Para cada configuración:
  - Entrena el modelo
  - Evalúa su rendimiento
  - Representa la frontera de decisión resultante

### Qué debes observar
- ¿Cómo cambia la frontera de decisión al aumentar el número de neuronas ocultas?
- ¿Con pocas neuronas la frontera sigue siendo demasiado simple?
- ¿Con muchas neuronas el modelo capta mejor la estructura no lineal?
- ¿Aparece alguna frontera excesivamente irregular o sobreajustada?

### Análisis
- Relaciona la complejidad del modelo con su comportamiento:
  - ¿Underfitting con modelos simples?
  - ¿Overfitting con modelos muy flexibles?
- Apóyate en las gráficas y resultados observados.

---

## Tarea 3: Funciones de activación

### Qué debes hacer
- Elige una arquitectura fija de MLP.
- Entrénala dos veces:
  - Una usando Sigmoide
  - Otra usando ReLU
- Compara ambos modelos en términos de:
  - Evolución del entrenamiento
  - Número de iteraciones necesarias para converger
  - Rendimiento final
  - Aspecto de la frontera de decisión

### Cuestiones
- ¿Cuál de las dos funciones parece converger más rápido?
- ¿Se observan diferencias en la estabilidad del entrenamiento?
- ¿La frontera de decisión resultante parece más suave o abrupta?

### Interpretación
- Explica por qué puede ocurrir lo observado:
  - Saturación de la sigmoide
  - Facilidad de optimización
  - Capacidad de introducir no linealidad
  - Efecto práctico sobre el aprendizaje

---

## Tarea 4: El reto de la caja negra

### Qué debes hacer
- Carga el dataset de dígitos.
- Diseña distintas arquitecturas MLP variando:
  - Número de capas ocultas
  - Número de neuronas por capa
- Encuentra una configuración que alcance al menos un 95% de acierto.

### Restricción experimental
- No pruebes combinaciones al azar sin criterio.
- Justifica el proceso seguido:
  - ¿Qué arquitecturas has probado?
  - ¿Cuáles han funcionado peor o mejor?
  - ¿Cuál es la configuración mínima razonable para alcanzar el objetivo?

### Reflexión
- ¿Es preferible una sola capa con muchas neuronas o dos capas con menos neuronas cada una?
- Apóyate en los resultados y discute el compromiso entre:
  - Simplicidad
  - Capacidad de representación
  - Dificultad de entrenamiento
  - Interpretabilidad
  - Rendimiento

---

## Reto: Decisión de diseño

Imagina que formas parte de un equipo técnico que necesita desplegar un clasificador basado en redes neuronales para un problema real. Redacta una breve propuesta respondiendo a:
- ¿Cuándo tiene sentido usar una red sin capas ocultas y cuándo no?
- ¿Qué arquitectura recomendarías para un problema no lineal sencillo?
- ¿Qué ventajas e inconvenientes observas al aumentar el número de neuronas?
- ¿Qué función de activación elegirías en un problema general de clasificación y por qué?
- ¿Qué criterio usarías para decidir si una arquitectura es suficientemente buena sin hacerla innecesariamente compleja?

---

## Formato de entrega
- Explicación breve del problema de la no linealidad
- Resultados con el perceptrón simple
- Al menos tres experimentos con una capa oculta variando el número de neuronas
- Gráficas de las fronteras de decisión
- Comparación entre Sigmoide y ReLU
- Experimentos sobre el dataset de dígitos
- Arquitectura final seleccionada para ≥95%
- Discusión: redes más anchas vs. más profundas
- Conclusión final sobre el papel de la arquitectura en el MLP

---

## Recomendaciones
- No te limites a pegar código o tablas sin explicación.
- Interpreta los resultados en cada fase:
  - Por qué falla un modelo lineal en datos no lineales
  - Cómo las capas ocultas modifican la frontera de decisión
  - Por qué más neuronas no siempre implican mejor solución
  - Qué diferencias introduce la función de activación
  - Cómo justificar una arquitectura desde la generalización, no solo el acierto
- Prioriza resultados, visualizaciones y su interpretación.

---

## Objetivo final
Entender que una red neuronal no es una caja mágica, sino una familia de modelos cuyo comportamiento depende de decisiones de diseño concretas: arquitectura, función de activación y equilibrio entre ajuste y generalización.
