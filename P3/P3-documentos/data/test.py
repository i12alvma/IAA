import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

df = pd.read_csv("pacientes_riesgo.csv")


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