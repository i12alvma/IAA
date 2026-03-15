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
- Modelo usado: ______________________
- Métricas usadas: ___________________
- Semilla(s): ________________________

### Registro de resultados
| Ejecución | Positivos en test |
|---|---:|
| 1 | |
| 2 | |
| 3 | |
| 4 | |
| 5 | |

### Análisis
- ¿Ha cambiado mucho el número de positivos en test entre ejecuciones?
- ¿Sería fiable la evaluación si solo hubiera 0 o 1 positivos en test?

---

## Tarea 2 — Partición estratificada

### Registro por folds
| Fold | Positivos en test | Tamaño del fold | Proporción clase 1 |
|---|---:|---:|---:|
| 1 | | | |
| 2 | | | |
| 3 | | | |
| 4 | | | |
| 5 | | | |

### Análisis
Explica por qué `StratifiedKFold` es más adecuado que una partición aleatoria simple en este problema.

---

## Tarea 3 — Comparativa de la varianza

### Resumen de resultados
| Método | Accuracy media | std(Accuracy) | F1 media | std(F1) |
|---|---:|---:|---:|---:|
| KFold | 0.5771 | 0.0517 | 0.0383 | 0.0380 |
| StratifiedKFold | 0.5783 | 0.0464 | 0.0430 | 0.0293 |

### Interpretación
- ¿Qué método produce métricas más estables?

- ¿Dónde se nota más la mejora: en Accuracy o en F1?

---

## Tarea 4 — Detección de data leakage

### Comparativa
| Escenario | Accuracy | F1 |
|---|---:|---:|
| Sin variable trampa | 0.5750 | 0.0404 |
| Con variable trampa | 1.0000 | 1.0000 |

### Preguntas
- ¿Qué variable sospechas que está filtrando información del objetivo?

- ¿Por qué esa variable invalida la evaluación?

- ¿Qué diferencia conceptual hay entre un modelo “bueno” y un modelo “contaminado” por leakage?

---

## Reto — Auditoría metodológica
Redacta un pequeño informe (6–10 líneas) respondiendo a estas cuestiones:
1. ¿Por qué una partición aleatoria simple puede ser peligrosa en problemas desbalanceados?

2. ¿Qué aporta la validación estratificada?

3. ¿Por qué el *data leakage* puede hacer inútiles las conclusiones de un experimento?

4. ¿Qué buenas prácticas metodológicas seguirías en futuros experimentos?

---

## Conclusión final
Resume qué has aprendido sobre evaluación rigurosa, estabilidad de métricas y detección de errores metodológicos.

Esta práctica muestra que, en clasificación desbalanceada, el rigor metodológico es tan importante como el modelo.
La validación estratificada mejora la estabilidad de la evaluación, especialmente en `F1`, al mantener una
representación consistente de la clase minoritaria en cada partición. También se ha comprobado que el *data leakage*
puede inflar artificialmente el rendimiento hasta valores perfectos sin que exista verdadera capacidad predictiva.
Por tanto, para obtener conclusiones fiables es imprescindible controlar la partición de datos, elegir métricas
adecuadas al desbalanceo y auditar posibles fugas de información antes de aceptar resultados.
