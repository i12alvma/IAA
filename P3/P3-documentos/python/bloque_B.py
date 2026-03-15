import pandas as pd
from pathlib import Path

# ----------------------------
# Carga de datos
# ----------------------------
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "pacientes_riesgo.csv"
df = pd.read_csv(DATA_PATH)

y = df["Clase"].values

# Sin trampa
X_clean = df.drop(columns=["Clase", "ID_Hospital_Filtro"]).values

# Con trampa
X_leak = df.drop(columns=["Clase"]).values

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

# Evaluación sin leakage #
print("--- Evaluación sin Data Leakage ---")

# 1) Partición simple 
X_train, X_test, y_train, y_test = train_test_split(
    X_clean, y, test_size=0.2, random_state=42
)

# 2) Entrenamiento del modelo
modelo = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
modelo.fit(X_train, y_train)

# 3) Predecir
y_pred = modelo.predict(X_test)

# 4) Matriz de confusión
tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()

#.ravel() es un método de NumPy que aplana un array multidimensional en un array unidimensional.

print("Matriz de confusión:")
print(f"TN: {tn}, FP: {fp}, FN: {fn}, TP: {tp}")

# 5) Calcular métricas
accuracy = (tp + tn) / (tp + tn + fp + fn)
f1_score = 2 * tp / (2 * tp + fp + fn)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score: {f1_score:.4f}")


# Ahora con leakage #
print("\n--- Evaluación con Data Leakage ---")

# 1) Partición 
X_train_l, X_test_l, y_train_l, y_test_l = train_test_split(
    X_leak, y, test_size=0.2, random_state=42
)

# 2) Entrenamiento del modelo
modelo_l = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
modelo_l.fit(X_train_l, y_train_l)

# 3) Predecir
y_pred_l = modelo_l.predict(X_test_l)

# 4) Matriz de confusión
tn_l, fp_l, fn_l, tp_l = confusion_matrix(y_test_l, y_pred_l, labels=[0,1]).ravel()

print("Matriz de confusión:")
print(f"TN: {tn_l}, FP: {fp_l}, FN: {fn_l}, TP: {tp_l}")

# 5) Calcular métricas
acc_l = (tp_l + tn_l) / (tp_l + tn_l + fp_l + fn_l)
f1_l = 2 * tp_l / (2 * tp_l + fp_l + fn_l)

print(f"Accuracy: {acc_l:.4f}")
print(f"F1 Score: {f1_l:.4f}")

