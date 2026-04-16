# Práctica 6: Clasificación de Texto y Filtros Anti-Spam (Naive Bayes)

## Introducción al Aprendizaje Automático
**3º Ingeniería Informática - Curso 2025/2026**

---

## Objetivo
Comprender cómo un modelo probabilístico puede clasificar texto a partir de la frecuencia de las palabras. En concreto, se estudia cómo representar mensajes con una bolsa de palabras, cómo aplica Naive Bayes el Teorema de Bayes para distinguir entre mensajes legítimos y spam, y por qué el suavizado de Laplace es esencial cuando aparecen términos no vistos en entrenamiento.

---

## Material de partida
- Dataset real de SMS etiquetados como ham o spam.
- Script base en Python para carga de datos, vectorización y entrenamiento.
- Problema de clasificación binaria: detectar spam a partir del contenido textual.

---

## Introducción
El texto no se puede usar directamente por la mayoría de algoritmos de aprendizaje automático, por lo que primero se transforma a formato numérico. En esta práctica se usa Bag of Words: cada mensaje se convierte en un vector de frecuencias de palabras.

A partir de esa representación, el clasificador Multinomial Naive Bayes estima probabilidades por clase usando:
- Probabilidades a priori de cada clase.
- Probabilidades condicionales de las palabras dadas las clases.

Este enfoque, aunque simple, permite construir un clasificador eficaz y además interpretable en términos probabilísticos.

---

## Tarea 1: Del texto a los números

### Qué se hizo
- Carga del dataset spam.csv.
- Conversión de etiquetas:
  - ham -> 0
  - spam -> 1
- Preprocesamiento básico:
  - paso a minúsculas;
  - limpieza de símbolos no alfanuméricos;
  - normalización de espacios.
- División entrenamiento/prueba con partición estratificada.
- Vectorización Bag of Words con CountVectorizer.

### Cuestión
La transformación es necesaria porque el modelo requiere variables numéricas para calcular probabilidades y comparar clases. Bag of Words conserva la información léxica (qué palabras aparecen y con qué frecuencia), que es precisamente la señal útil para detectar spam.

Lo que se pierde es el orden exacto de las palabras y parte del contexto sintáctico. Por ejemplo, dos mensajes con las mismas palabras pero distinto orden se representan igual.

### Reflexión sobre la matriz documento-término
- Cada fila representa un mensaje.
- Cada columna representa una palabra del vocabulario.
- Cada valor representa cuántas veces aparece esa palabra en ese mensaje.

Esta representación condiciona lo que aprende el modelo: aprende asociaciones palabra-clase, no estructura gramatical completa.

---

## Tarea 2: Entrenando el clasificador Naive Bayes

### Qué se hizo
- Entrenamiento de MultinomialNB con alpha = 1.0.
- Ajuste del modelo en entrenamiento y predicción en prueba.

### Interpretación probabilística
- Probabilidad a priori (estimada en train):
  - P(ham) = 0.8660
  - P(spam) = 0.1340

Esto indica que la clase ham es mayoritaria en el dataset.

- Probabilidad condicional de palabras:
  - Ejemplo conceptual: P("gratis" | spam) representa la frecuencia relativa de esa palabra dentro de la clase spam. Si es alta respecto a P("gratis" | ham), empuja la predicción hacia spam.

### Por qué se llama Naive
Se asume independencia condicional entre palabras dado el tipo de mensaje. En lenguaje real esa independencia no es estricta, pero simplifica mucho el cálculo y funciona bien en la práctica.

---

## Tarea 3: Papel del suavizado de Laplace

### Problema sin suavizado
Si una palabra tiene frecuencia cero en una clase, su probabilidad condicional sería 0. Como Naive Bayes combina probabilidades multiplicando términos, una sola probabilidad cero puede anular toda la probabilidad de esa clase.

### Solución
Con alpha = 1.0 (suavizado de Laplace), se evita que aparezcan probabilidades exactamente nulas. En la práctica:
- mejora la robustez ante vocabulario no observado;
- evita colapsos numéricos por ceros;
- mantiene estimaciones razonables con datos incompletos.

### Caso pedido: palabra no vista ("oferta")
En la ejecución de prueba con mensajes inventados en español:
- mensaje: "oferta"
- salida del modelo: P(ham)=0.8660, P(spam)=0.1340

Como la palabra no pertenece al vocabulario aprendido (entrenado mayoritariamente en inglés), el vector resultante no aporta evidencia léxica y la predicción cae en la probabilidad a priori de las clases, en lugar de romperse por probabilidad nula.

---

## Tarea 4: Evaluación del modelo

### Resultados principales
- Accuracy global: 0.9842
- Matriz de confusión:
  - TN = 1204
  - FP = 2
  - FN = 20
  - TP = 167

- Métricas por clase:
  - ham: precision 0.9837, recall 0.9983, f1 0.9909
  - spam: precision 0.9882, recall 0.8930, f1 0.9382

### Interpretación
El modelo detecta muy bien ham y tiene muy pocos falsos positivos (solo 2 ham marcados como spam). Sin embargo, deja escapar parte del spam (20 falsos negativos), por lo que su recall en spam es la métrica más mejorable.

### Qué error es más problemático
En un filtro anti-spam real suele ser más problemático clasificar un ham como spam (falso positivo), porque puede ocultar mensajes legítimos importantes. En este resultado, ese error es bajo.

Figura generada:
- matriz_confusion_spam_base.png

---

## Tarea 5: Inspeccionando qué aprendió el modelo

### Top 5 palabras más asociadas a spam
1. claim
2. prize
3. 150p
4. tone
5. 18

Estas palabras son razonables: tienen un tono promocional/comercial y son típicas en campañas de spam (premios, reclamaciones y mensajes de tarificación).

### Reflexión de interpretabilidad
Este análisis permite entender parcialmente por qué el modelo decide spam: si aparecen términos con alta asociación diferencial a esa clase, la predicción se inclina con fuerza hacia spam. No es una explicación causal completa, pero sí una evidencia interpretable del patrón aprendido.

---

## Reto: Diseñar mensajes de prueba

### Mensajes propuestos y resultados
1. Mensaje claramente ham:
   - "Hi, are we still meeting tomorrow at the library?"
   - Predicción: ham
   - P(ham)=1.0000, P(spam)=0.0000

2. Mensaje claramente spam:
   - "Congratulations! You have won a free vacation. Claim your prize now!"
   - Predicción: spam
   - P(ham)=0.0000, P(spam)=1.0000

3. Mensaje ambiguo:
   - "Hello, we have a special offer for you if you reply today."
   - Predicción: ham
   - P(ham)=0.9743, P(spam)=0.0257

### Cuestiones
- ¿Coincide con lo esperado?: Sí en los dos casos extremos; en el ambiguo el modelo no detecta suficiente señal de spam.
- ¿Dónde está más seguro?: En los casos claramente ham y claramente spam (probabilidades extremas).
- ¿Por qué duda en el ambiguo?: El texto no contiene muchas palabras fuertemente asociadas al spam en el vocabulario aprendido y usa términos más generales.

---

## Conclusión
Naive Bayes, pese a su simplicidad, resuelve bien un problema real de clasificación de texto cuando la representación de entrada es adecuada. La vectorización Bag of Words ofrece una señal suficiente para separar ham y spam con alta precisión. El suavizado de Laplace es crítico para evitar probabilidades nulas y mantener robustez ante términos no vistos.

Al mismo tiempo, el modelo depende de supuestos concretos (independencia entre palabras) y de decisiones de preprocesado/vectorización. Por ello, no es una caja mágica: su rendimiento y su tipo de error responden directamente al diseño del pipeline y al vocabulario observado en entrenamiento.
