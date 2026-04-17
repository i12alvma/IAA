# Práctica 6: Clasificación de Texto y Filtros Anti-Spam (Naive Bayes)

## Introducción al Aprendizaje Automático
**3º Ingeniería Informática - Curso 2025/2026**

---

## Objetivo
Comprender cómo un modelo probabilístico puede clasificar texto a partir de la frecuencia de
las palabras. En particular, se estudiará cómo representar mensajes mediante una bolsa de palabras, cómo aplica
el clasificador Naïve Bayes el Teorema de Bayes para distinguir entre mensajes legítimos y spam, y por qué el
suavizado de Laplace resulta esencial cuando aparecen términos no vistos durante el entrenamiento.

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

En esta práctica vas a trabajar con estas ideas de forma experimental, observando cómo un modelo puede aprender a detectar patrones de lenguaje asociados al spam.

---

## Tarea 1: Del texto a los números

### Qué se hizo
- [Carga del dataset / explicación de qué se ha hecho aquí.]
- [Conversión de etiquetas a formato numérico.]
- [Preprocesamiento básico del texto.]
- [División entre entrenamiento y prueba.]
- [Vectorización con Bag of Words.]

### Cuestión
[Debes explicar por qué esta transformación es necesaria.
No basta con indicar que el algoritmo necesita números. Debes razonar qué información conserva una representación basada en frecuencias de palabras y qué tipo de información se pierde al ignorar el orden exacto de las palabras en la frase.]

### Reflexión sobre la matriz documento-término
[Comenta qué significa que en la matriz resultante que cada fila represente un mensaje, que cada columna represente una palabra del vocabulario, y que cada valor indique cuántas veces aparece esa palabra en ese mensaje.

El objetivo de esta parte es que entiendas que la representación de los datos condiciona completamente lo que el modelo puede aprender.]

[Escribe aquí una reflexión breve sobre qué aprende el modelo a partir de esta matriz.]

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
