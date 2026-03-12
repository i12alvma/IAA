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



df_0 = df[df["Clase"]==0]
df_1 = df[df["Clase"]==1]

plt.figure(figsize=(8,5))

# Scatter para clase 0
plt.scatter(
    df_0["ID_Hospital_Filtro"], 
    df_0["Clase"], 
    color="blue", 
    label="Clase = 0", 
    alpha=0.6
)

# Scatter para clase 1
plt.scatter(
    df_1["ID_Hospital_Filtro"], 
    df_1["Clase"], 
    color="red", 
    label="Clase = 1", 
    alpha=0.6
)

plt.xlabel("ID_Hospital_Filtro")
plt.ylabel("Clase")
plt.title("Data Leakage: ID_Hospital_Filtro revela la Clase")
plt.legend()  # agrega la leyenda
plt.savefig("P3/P3-documentos/outputs/idvsclase", dpi=150)
plt.show()