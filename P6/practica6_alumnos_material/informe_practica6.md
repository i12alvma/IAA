# Práctica 6: Clasificación de Texto y Filtros Anti-Spam (Naive Bayes)

## Introducción al Aprendizaje Automático
**3º Ingeniería Informática - Curso 2025/2026**

---

## Objetivo
Comprender cómo un modelo probabilístico puede clasificar texto a partir de la frecuencia de las palabras. En particular, se estudiará cómo representar mensajes mediante una bolsa de palabras, cómo aplica
el clasificador Naïve Bayes el Teorema de Bayes para distinguir entre mensajes legítimos y spam, y por qué el suavizado de Laplace resulta esencial cuando aparecen términos no vistos durante el entrenamiento.

---

## Material de partida
Se proporciona:
- Un dataset real de mensajes SMS etiquetados como ham o spam, descargado de Kaggle.
- Una plantilla de código base en Python con pandas y scikit-learn, para facilitar la carga de datos, la vectorización del texto y el entrenamiento inicial del modelo.
- Un problema de clasificación binaria en el que el objetivo es decidir si un mensaje es legítimo o no deseado a partir de su contenido textual.

---

## Introducción
En problemas de clasificación de texto, los datos de entrada no son números ni vectores geométricos sencillos, sino mensajes escritos en lenguaje natural. Antes de poder aplicar un algoritmo de aprendizaje automático, es necesario transformar ese texto en una representación numérica que el modelo pueda manejar.

Una estrategia clásica consiste en usar una bolsa de palabras (Bag of Words), donde cada mensaje se representa mediante el recuento de las palabras que contiene. A partir de esa representación, un clasificador como Naive Bayes puede estimar qué palabras aparecen con mayor frecuencia en los mensajes spam y cuáles son más habituales en mensajes legítimos.

Este enfoque es especialmente interesante porque, aunque es sencillo, permite introducir varias ideas fundamentales del aprendizaje automático: la necesidad de preprocesar los datos antes de entrenar un modelo; la conversión de información textual en variables numéricas; el uso de probabilidades para clasificar ejemplos; y la importancia de evitar problemas numéricos mediante técnicas como el suavizado de Laplace.

En esta práctica se va a trabajar con estas ideas de forma experimental, observando cómo un modelo puede aprender a detectar patrones de lenguaje asociados al spam.

---

## Tarea 1: Del texto a los números

### Qué se hizo
- Se cargó el dataset de mensajes SMS desde el archivo spam.csv y se verificaron las columnas Category y Message.
- Se transformaron las etiquetas a formato numérico para clasificación binaria: ham -> 0 y spam -> 1.
- Se aplicó un preprocesamiento básico al texto: conversión a minúsculas, eliminación de signos/símbolos no alfanuméricos y normalización de espacios.
- Se dividió el conjunto de datos en entrenamiento y prueba con partición estratificada para conservar la proporción de clases (test_size = 0.25, random_state = 42).
- Se vectorizó el texto con Bag of Words usando CountVectorizer, obteniendo una matriz documento-término para entrenar y evaluar el modelo.

### Cuestión
<strong>¿Por qué es necesario transformar los mensajes de texto a una representación numérica tipo Bag of Words antes de entrenar Naive Bayes, y qué información se conserva o se pierde con esta transformación? </strong>

Esta transformación es necesaria debido a que Naive Bayes opera con variables numéricas y necesita contar evidencias para estimar probabilidades por clase. Con Bag of Words, cada mensaje se convierte en un vector de frecuencias de palabras, lo que conserva información léxica clave: qué términos aparecen y cuántas veces, algo muy útil para distinguir patrones típicos de spam y ham.

Sin embargo, se pierde el orden exacto de las palabras y parte del contexto gramatical o semántico. Por eso, mensajes con las mismas palabras pero en distinto orden pueden quedar representados de forma muy parecida. En resumen, Bag of Words simplifica el texto para hacerlo tratable por el modelo, manteniendo señal útil para clasificar, aunque sacrificando información de estructura lingüística.

### Reflexión sobre la matriz documento-término
 <strong>¿Qué representa cada elemento de la matriz documento-término (filas, columnas y valores) y cómo condiciona esta representación lo que el modelo puede aprender? </strong>

En la matriz documento-término, cada fila corresponde a un mensaje concreto del dataset, cada columna corresponde a una palabra del vocabulario total y cada valor indica cuántas veces aparece esa palabra en ese mensaje.

Esta representación convierte texto en números y permite que el clasificador compare patrones de frecuencia entre mensajes ham y spam. Por tanto, el modelo aprende asociaciones entre palabras y clases, por ejemplo qué términos son más habituales en spam.

Sin embargo, también impone límites: al trabajar con recuentos, se pierde el orden de las palabras y parte del contexto lingüístico. Eso significa que el modelo capta bien señales léxicas, pero no entiende la estructura completa de la frase. En resumen, la forma de representar los datos determina directamente qué información aprovecha el modelo y qué información queda fuera del aprendizaje.

---

## Tarea 2: Entrenando el clasificador Naive Bayes

### Qué se hizo
- [Entrenamiento del modelo Multinomial Naïve Bayes]
- [Ajuste del parámetro alpha]
- [Ajustar el modelo con los datos de entrenamiento]
- [Usar el modelo para predecir las etiquetas del conjunto de prueba]

### Interpretación probabilística
- Probabilidad a priori:
	- P(ham) = [completar]
	- P(spam) = [completar]

[Explica aquí qué significa la probabilidad a priori y cómo interpretar una probabilidad condicional como P("gratis" | spam)

Interpretar de forma razonada una idea como P("gratis" | spam) y explicar qué información aporta al modelo ]

### Por qué se llama Naive
[Explica aquí por qué el modelo asume independencia entre palabras y por qué esa hipótesis es una simplificación útil.

En particular, debes comentar que el clasificador asume una independencia simplificada entre palabras, aunque en lenguaje real esa independencia no se cumpla estrictamente. El interés aquí no es demostrar formalmente esta hipótesis, sino entender que se trata de una simplificación útil que permite construir un clasificador eficaz y fácil de entrenar]

---

## Tarea 3: Papel del suavizado de Laplace

### Problema sin suavizado
[Explica aquí qué ocurre cuando una palabra tiene frecuencia cero en una clase y por qué eso puede anular toda la probabilidad.]

### Solución
[Explica aquí por qué se usa alpha = 1.0 y cómo el suavizado de Laplace evita probabilidades nulas.]

### Caso pedido: palabra no vista
[Responde aquí qué ocurriría si se intentara clasificar una palabra como "oferta" cuando no ha aparecido en el entrenamiento.

Se debe dejar claro por qué, sin suavizado, una probabilidad nula podría arruinar el cálculo completo del modelo]

---

## Tarea 4: Evaluación del modelo

### Resultados principales
- Accuracy global: [completar]
- Matriz de confusión:
	- TN = [completar]
	- FP = [completar]
	- FN = [completar]
	- TP = [completar]

- Métricas por clase:
	- ham: precision [completar], recall [completar], f1 [completar]
	- spam: precision [completar], recall [completar], f1 [completar]

### Interpretación
[Interpreta aquí qué indican estas métricas sobre el comportamiento del modelo: si detecta bien la mayoría del spam, si penaliza demasiado mensajes legítimos o si parece conservador a la hora de marcar mensajes sospechosos.]

### Qué error es más problemático
[Explica aquí cuál de los dos errores te parece más problemático en un filtro anti-spam real y por qué.]

### Figura generada
[Inserta aquí la matriz de confusión y comenta brevemente su significado.]

---

## Tarea 5: Inspeccionando qué aprendió el modelo

### Top 5 palabras más asociadas a spam
1. [palabra 1]
2. [palabra 2]
3. [palabra 3]
4. [palabra 4]
5. [palabra 5]

### Reflexión de interpretabilidad
[Comenta aquí si las palabras encontradas son razonables o esperables y hasta qué punto esta inspección permite interpretar el comportamiento del clasificador.

Puedes apoyarte en cuestiones como si el modelo parece aprender patrones comprensibles para una persona, si algunas palabras tienen sentido claramente comercial o promocional, y si el vocabulario aprendido refleja realmente diferencias entre spam y mensajes normales.]

---

## Reto: Diseñar mensajes de prueba

### Mensajes propuestos y resultados
1. Mensaje claramente ham:
	 - [Escribe aquí tu mensaje]
	 - Predicción: [completar]
	 - P(ham) = [completar], P(spam) = [completar]

2. Mensaje claramente spam:
	 - [Escribe aquí tu mensaje]
	 - Predicción: [completar]
	 - P(ham) = [completar], P(spam) = [completar]

3. Mensaje ambiguo:
	 - [Escribe aquí tu mensaje]
	 - Predicción: [completar]
	 - P(ham) = [completar], P(spam) = [completar]

### Cuestiones
- ¿Coincide con lo esperado?: [completar]
- ¿Dónde está más seguro?: [completar]
- ¿Por qué duda en el ambiguo?: [completar]

---

## Conclusión
[Redacta aquí una conclusión breve sobre las ventajas y limitaciones de Naive Bayes para clasificación de texto.]
