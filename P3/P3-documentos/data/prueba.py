import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("P3/P3-documentos/data/pacientes_riesgo.csv")


X = df.drop("Clase", axis=1)
y = df["Clase"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      # 20% para test
    random_state= np.random.randint(1,100)
)

print("X_train:", X_train)
print("X_test:", X_test)
print("y_train:", y_train)
print("y_test:", y_test)

plt.scatter(X[y == 0]["Var_1"], y[y == 0], label="Clase 0", alpha=0.5, color="blue")
plt.scatter(X[y == 1]["Var_1"], y[y == 1], label="Clase 1", alpha=0.5, color="red")
plt.xlabel("Var_1")
plt.ylabel("Clase")
plt.title("Distribución de Var_1 por clase")
plt.legend()
plt.savefig("P3/P3-documentos/outputs/distribucion_trampa_train_test.png", dpi=150)
plt.show()