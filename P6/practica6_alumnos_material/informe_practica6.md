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
[Redacta aquí una breve introducción sobre por qué el texto debe transformarse a números, qué es Bag of Words y qué papel juega Naive Bayes en esta práctica.]

---

## Tarea 1: Del texto a los números

### Qué se hizo
- [Carga del dataset / explicación de qué se ha hecho aquí.]
- [Conversión de etiquetas a formato numérico.]
- [Preprocesamiento básico del texto.]
- [División entre entrenamiento y prueba.]
- [Vectorización con Bag of Words.]

### Cuestión
[Explica aquí por qué esta transformación es necesaria y qué información conserva o pierde la representación Bag of Words.]

### Reflexión sobre la matriz documento-término
En la matriz resultante:
- Cada fila representa [completar].
- Cada columna representa [completar].
- Cada valor representa [completar].

[Escribe aquí una reflexión breve sobre qué aprende el modelo a partir de esta matriz.]

---

## Tarea 2: Entrenando el clasificador Naive Bayes

### Qué se hizo
- [Entrenamiento del modelo MultinomialNB.]
- [Ajuste del parámetro alpha.]
- [Predicción sobre el conjunto de prueba.]

### Interpretación probabilística
- Probabilidad a priori:
	- P(ham) = [completar]
	- P(spam) = [completar]

[Explica aquí qué significa la probabilidad a priori y cómo interpretar una probabilidad condicional como P("gratis" | spam).]

### Por qué se llama Naive
[Explica aquí por qué el modelo asume independencia entre palabras y por qué esa hipótesis es una simplificación útil.]

---

## Tarea 3: Papel del suavizado de Laplace

### Problema sin suavizado
[Explica aquí qué ocurre cuando una palabra tiene frecuencia cero en una clase y por qué eso puede anular toda la probabilidad.]

### Solución
[Explica aquí por qué se usa alpha = 1.0 y cómo el suavizado de Laplace evita probabilidades nulas.]

### Caso pedido: palabra no vista
[Responde aquí qué ocurriría si se intentara clasificar una palabra como "oferta" cuando no ha aparecido en el entrenamiento.]

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
[Interpreta aquí qué indican estas métricas sobre el comportamiento del modelo.]

### Qué error es más problemático
[Explica aquí cuál de los errores (spam como ham o ham como spam) te parece más problemático y por qué.]

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
[Comenta aquí si las palabras encontradas son razonables y qué dicen sobre lo que ha aprendido el modelo.]

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
