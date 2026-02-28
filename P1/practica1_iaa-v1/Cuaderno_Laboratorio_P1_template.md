<div style="text-align: justify;">

# Cuaderno de Laboratorio — Práctica 1: La Física del Aprendizaje Automático

**Asignatura:** Introducción al Aprendizaje Automático (3º Ing. Informática)  
**Curso:** 2025/2026  
**Alumno/a:** Juan Luis Prieto Panadero y Alberto Álvarez Mejías <br>
**Fecha:** 27/02/2026

## Objetivo
Comprender la dinámica de la optimización mediante gradiente descendente y el impacto de los hiperparámetros en la convergencia.

---

## 1. Tarea 1 — La importancia de la tasa de aprendizaje (α)

### 1.1 Escenario A (αslow): convergencia lenta
- **Valor elegido:** α = 1e-4
- **Figura:** `outputs/alpha_A_alpha_slow_...png`

![αslow](/P1/outputs/alpha_A_alpha_slow_a0.0001.png)

**Análisis (obligatorio):**
- ¿Por qué la curva baja tan lentamente?

Con α = 1e-4, cada actualización de pesos es muy pequeña, por eso el descenso del coste es casi lineal y muy suave. El algoritmo avanza en la dirección correcta, pero con pasos demasiado cortos.

- ¿Cuántas iteraciones estimas que harían falta para llegar cerca del mínimo? Justifica.

Tomando la pendiente media observada en la curva, el número de iteraciones para acercarse al mínimo sería **muy alto (del orden de \(10^4\)–\(10^5\) iteraciones)**, claramente poco eficiente en tiempo de entrenamiento.

### 1.2 Escenario B (αopt): “curva de codo”
- **Valor elegido:** α = 0.01
- **Figura:** `outputs/alpha_B_alpha_opt_...png`

![αopt](/P1/outputs/alpha_B_alpha_opt_a0.01.png)

**Análisis (obligatorio):**
- Describe la “curva de codo” y qué te dice sobre estabilidad y rapidez.

La curva muestra una bajada rápida al principio (reducción fuerte del error) y después una meseta estable. Ese “codo” indica un buen equilibrio entre **rapidez** y **estabilidad**: el modelo llega pronto a una zona cercana al mínimo y luego hace ajustes finos sin oscilar de forma excesiva.

### 1.3 Escenario C (αosc): oscilación amortiguada
- **Valor elegido:** α = 0.13
- **Figura:** `outputs/alpha_C_alpha_osc_...png`

![αosc](/P1/outputs/alpha_C_alpha_osc_a0.13.png)

**Análisis (obligatorio):**
- ¿Qué está ocurriendo “físicamente” con los pesos θ en el espacio de búsqueda?
- Explica por qué puede oscilar y aun así estabilizarse.

Aquí los pesos θ “rebotan” alrededor del valle de la función de coste: en cada paso se sobrepasa el mínimo local y se cruza al otro lado, generando subidas y bajadas del error.
Físicamente, en el espacio de búsqueda hay un movimiento en zig-zag por usar un paso grande. Aun así, la oscilación se amortigua porque la tendencia global sigue apuntando hacia una región de menor coste.

### 1.4 Escenario D (αfail): divergencia
- **Valor elegido:** α = 1.1
- **Figura:** `outputs/alpha_D_alpha_fail_...png`

![αfail](/P1/outputs/alpha_D_alpha_fail_a1.1.png)

**Análisis (obligatorio):**
- ¿Por qué el error crece? Relaciónalo con “pasarse” del mínimo.

Con α = 1.1, el tamaño de paso es demasiado grande: el algoritmo se “pasa” sistemáticamente del mínimo y cada actualización puede alejar más los pesos de la zona estable. Por eso la oscilación no se amortigua y el coste termina creciendo (divergencia), en lugar de converger.


---

## 2. Tarea 2 — El compromiso del Mini-batch

Usa tu **αopt** (del escenario B) y compara:

### 2.1 Batch completo
![Batch completo](/P1/outputs/batch_Batch_completo_b500.png)

### 2.2 Mini-batch (32 o 16)
![Mini-batch](/P1/outputs/batch_Mini_batch_32_b32.png)

### 2.3 Estocástico puro (batch=1)
![Estocástico](/P1/outputs/batch_Estocastico_1_b1.png)

**Preguntas de reflexión (obligatorio):**
1. ¿Cuál de las tres curvas es más “ruidosa” y por qué?
   
   La más ruidosa es la del método **estocástico (batch = 1)**.  
   Esto pasa porque en cada paso solo se usa un dato para calcular el gradiente. Entonces, cada actualización cambia bastante y la función de coste sube y baja continuamente.

   El **batch completo** es el más suave, porque usa todos los datos y el gradiente es mucho más estable.

   El **mini-batch** está en medio: no es tan estable como el batch completo, pero tampoco tan inestable como el estocástico.
---

2. A nivel de tiempo de ejecución (CPU), ¿cuál ha sido más eficiente?  
   Justifica basándote en computación vectorial y número de actualizaciones.

   El más eficiente ha sido el **mini-batch**.

   El batch completo aprovecha bien las operaciones con matrices grandes, pero cada iteración tarda más.
   
   El estocástico hace muchísimas actualizaciones pequeñas y no aprovecha tanto la vectorización, por lo que puede ser menos eficiente.

   El mini-batch combina lo mejor de los dos: operaciones suficientemente grandes para ser eficientes y menos coste por iteración que el batch completo.
---

## 3. El reto — “Ajuste de Precisión” (criterio de parada)

Modifica el script para que el entrenamiento se detenga automáticamente cuando:

$|J_t - J_{t-1}| < 10^{-5}$

- **Mejor combinación encontrada:** α = 0.01 ; batch = 32
- **Épocas hasta parar:** 77
- **Figura (opcional):** <br>
![Early stop](/P1/outputs/reto_early_stop.png)

**Conclusión:**
- ¿Qué combinación para antes con error aceptable?
  
  La combinación que ha mostrado mejor comportamiento ha sido **α = 0.01 y batch = 32**, ya que consigue detener el entrenamiento en menos épocas y el error final es prácticamente el mismo que el del batch completo.

- ¿Qué sacrificas (si algo) para conseguirlo?
  
   Al usar mini-batch se introduce un poco más de variabilidad en el cálculo del gradiente porque no se usan todos los datos en cada actualización. Aun así, esta variabilidad no afecta mucho al resultado final y permite que el modelo converja más rápido.
---

## 4. Conclusiones finales (obligatorio)
Resume en 8–12 líneas lo que has aprendido sobre:
- relación entre α y estabilidad:

Durante los experimentos, se ha observado que la tasa de aprendizaje α controla el tamaño de cada paso en el espacio de parámetros:
- Si α es muy pequeña, el entrenamiento es estable pero extremadamente lento.
En ese caso, la curva de coste desciende casi en línea recta y requiere muchas iteraciones.

- Si α toma un valor intermedio, aparece la “curva de codo”: caída rápida y estabilización.
Ese régimen es el más eficiente porque combina velocidad de convergencia y buen control del error.

- Si α toma un valor muy alto, el coste empieza a oscilar por sobrepasar el mínimo en cada actualización. &nbsp; <br>
Si esta oscilación se amortigua, el modelo aún puede converger, aunque con menor precisión temporal. <br>
Cuando α supera un umbral crítico, la oscilación no se amortigua y el entrenamiento diverge.

Por tanto, α define el equilibrio clave entre rapidez, estabilidad y fiabilidad del aprendizaje.

- Efecto del batch en ruido/velocidad:

   El tamaño del batch influye directamente en el ruido del entrenamiento. 
   - El batch completo produce una curva muy suave porque el gradiente se calcula usando toda la información del dataset. 
   - El mini-batch introduce un pequeño ruido que puede ayudar a escapar de mínimos locales, manteniendo un buen equilibrio entre estabilidad y velocidad. 
   - El método estocástico es el más ruidoso porque cada actualización se basa en un solo ejemplo.

- Utilidad de un criterio de parada:

   El criterio de parada es muy útil porque evita seguir entrenando el modelo cuando la mejora entre una iteración y la siguiente es muy pequeña. Esto permite ahorrar tiempo de cálculo y recursos computacionales.

   En esta práctica se ha utilizado el criterio $|J_t - J_{t-1}| < 10^{-5}$, que detiene el entrenamiento cuando el coste prácticamente deja de cambiar.

   De esta forma se evita realizar iteraciones innecesarias una vez que el modelo ha alcanzado una zona de convergencia, manteniendo un error aceptable y reduciendo el tiempo de entrenamiento.


