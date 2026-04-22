"""
Práctica 6: Clasificación de Texto y Filtros Anti-Spam (versión base para alumnos)

Objetivo:
    Completar un clasificador de texto basado en Naïve Bayes para distinguir
    entre mensajes legítimos (ham) y spam.

Instrucciones:
    1. Lee el código completo antes de empezar.
    2. Completa las zonas marcadas con TODO.
    3. Ejecuta el script y revisa los resultados.
    4. Responde en tu informe a las preguntas planteadas en los comentarios.

Dependencias:
    pip install pandas scikit-learn matplotlib

Ejecución:
    python practica_6_naive_bayes_spam_base_alumno.py
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


# ============================================================================
# PARTE 1. CARGA Y LIMPIEZA DEL DATASET
# ============================================================================

def clean_text(text: str) -> str:
    """
    Limpia un texto de entrada.

    TODO:
        Completa esta función para que haga, al menos, lo siguiente:
        - pasar el texto a minúsculas;
        - eliminar signos de puntuación y símbolos extraños;
        - eliminar espacios múltiples.

    Pista:
        Puedes usar expresiones regulares con re.sub(...).
    """
    text = str(text)

    # Convertir a minúsculas
    text = text.lower()

    # Sustituir caracteres no alfanuméricos por espacios
    ''' 
    Existen varias formas de hacerlo, algunas son:
     - Expresiones regulares -> para ello se usa  # import re (librería de expresiones regulares) y el método .sub() que devuelve la subcadena resultante 
      · re.sub(r'[^\w\s]', ' ', text) # ^\w\s: cualquier cosa q no sea una palabra (letra o num) ni un espacio
      · re.sub(r'[^a-zA-Z0-9\s]', ' ', text) # ^[a-zA-Z0-9\s]: cualquier cosa q no esté comprendida entre a-z, A-Z, 0-9 o espacios
     - LOOP:
        result = []
        for char in text:
            if char.isalnum() or char.isspace(): # isalnum: alphanumeric
                result.append(char)
            else:
                result.append(' ')
        text = ''.join(result)
    '''
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)

    # Eliminar espacios repetidos y recortar extremos
    text = re.sub(r'\s+', ' ', text).strip()
    # \s+ captura uno o más espacios
    # .strip() elimina espacios al inicio y final

    return text


def load_dataset(csv_path: str | Path) -> pd.DataFrame:
    """
    Carga el dataset y prepara una etiqueta numérica.

    Se espera un CSV con las columnas:
        - Category
        - Message

    ham  -> 0
    spam -> 1
    """
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró el fichero: {csv_path.resolve()}\n"
            "Asegúrate de que 'spam.csv' está en la misma carpeta que este script."
        )

    df = pd.read_csv(csv_path)

    print("\nColumnas detectadas en el CSV:")
    print(list(df.columns))

    # Nos quedamos solo con las columnas necesarias.
    df = df[["Category", "Message"]].copy()
    df["Category"] = df["Category"].astype(str).str.strip().str.lower()
    df["Message"] = df["Message"].astype(str)

    # Crear una columna label_num que convierta:
    # ham -> 0
    # spam -> 1
    # Ejemplo orientativo:
    # df["label_num"] = ...
    df["label_num"] = df["Category"].map({"ham": 0, "spam": 1})
    if df["label_num"].isna().any():
        unknown = sorted(df.loc[df["label_num"].isna(), "Category"].unique())
        raise ValueError(f"Etiquetas no reconocidas en Category: {unknown}")
    ''' 
    if df["label_num"].isna().any():
     ·comprueba si, al convertir etiquetas texto a números, quedó alguna sin convertir
     ·.isna() devuelve un booleano indicando si hay valores NaN (no numéricos) en la columna label_num
     ·.any() devuelve True si hay al menos un valor NaN
    
    - df["label_num"].isna()
        Detecta las filas donde label_num quedó vacío (NaN), o sea, categorías que no se pudieron mapear a 0/1.

    - df.loc[ ..., "Category" ]
        De esas filas problemáticas, se queda solo con el valor original de Category.

    - .unique()
        Elimina repetidos, para no listar la misma etiqueta varias veces.

    - sorted(...)
        Ordena alfabéticamente esas etiquetas para mostrar un mensaje limpio y consistente.
    
    Resultado:
     unknown queda como una lista con las categorías no reconocidas 
      (por ejemplo, algo como ["hamm", "spm"]), 
      y luego se lanza una execpticón ValueError para decir exactamente
       qué etiquetas están mal en el CSV.
    '''

    df["label_num"] = df["label_num"].astype(int)
    # Con .astype(int) nos aseguramos que las etiquetas queden como int, 
     # ya que después de .map() los valores pueden quedar como float

    return df


# ============================================================================
# PARTE 2. EXPLORACIÓN INICIAL
# ============================================================================

def show_basic_info(df: pd.DataFrame) -> None:
    """Muestra información básica del dataset."""
    print("\n" + "=" * 80)
    print("INFORMACIÓN BÁSICA DEL DATASET")
    print("=" * 80)
    print(f"Número total de mensajes: {len(df)}")
    print("\nDistribución de clases:")
    print(df["Category"].value_counts())
    print("\nPrimeros 5 mensajes:")
    print(df.head())

    # Pregunta para el informe:
    # ¿El dataset está balanceado o hay bastantes más mensajes ham que spam?


# ============================================================================
# PARTE 3. PARTICIÓN TRAIN / TEST Y BOLSA DE PALABRAS
# ============================================================================

def prepare_data(df: pd.DataFrame):
    """
    Prepara los datos para el entrenamiento.

    TODO:
        - Separar variables de entrada X y etiquetas y.
        - Dividir en entrenamiento y test.
        - Crear un CountVectorizer usando clean_text como preprocessor.
        - Ajustar el vectorizador con train y transformar train/test.
    """
    # Seleccionar mensajes y etiquetas
    X = df["Message"]
    y = df["label_num"]
    # La forma más fácil de hacerlo es usar las categorías

    # dividir en train y test
    # Pista: usa test_size=0.25, random_state=42 y stratify=y
    X_train, X_test, y_train, y_test = train_test_split( # import train_test_split from sklearn.model_selection
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )
    ''' 
    Este bloque hace la partición del dataset en entrenamiento y prueba:
     ·X, y: son las entradas (mensajes) y sus etiquetas (ham/spam en 0/1).
     ·test_size=0.25: reserva el 25% para test y deja 75% para train.
     ·random_state=42: fija la semilla aleatoria para que siempre salga
       la misma partición (reproducible).
     ·stratify=y: mantiene la proporción de clases en train y test
    
    Salida:
     ·X_train, y_train: para entrenar el modelo.
     ·X_test, y_test: para evaluar en datos no vistos
    '''

    # Crear el vectorizador Bag of Words
    vectorizer = CountVectorizer(preprocessor=clean_text)
    '''
    Crea el objeto que convierte texto en números
    En el argumento indica que antes de vectorizar cada mensaje,
     llame a clean_text() para limpiar cada mensaje

    Esta línea únicamente crea y configura el vectorizador
    '''

    # Generar la matriz documento-término de train y test
    X_train_dtm = vectorizer.fit_transform(X_train)
    X_test_dtm = vectorizer.transform(X_test)

    '''
    X_train_dtm = vectorizer.fit_transform(X_train):
    ·Transforma mensajes de texto en números que Naïve Bayes puede procesar
    - CountVectorizer:
     · Herramienta que convierte texto en matriz de frecuencias de palabras
     · Lee el texto y crea un vocabulario formado por todas las palabras únicas
     · Convierte cada mensaje en un vector mostrando cuántas veces aparece cada palabra
    - fit_transform(X_train), consiste en 2 operaciones combinadas:
     · fit = Aprende el vocabulario escaneando X_train
     · transform = Convierte X_train a números usando ese vocabulario
     · Resultado: matriz documento-término (sparse matrix)
    - X_train_dtm, donde se guarda la matriz resultante:
     · Cada fila representa un mensaje
     · Cada columna representa una palabra del vocabulario aprendido
     · El valor en cada celda es la frecuencia de esa palabra en ese mensaje
    Se trata de una operación crítica para Naïve Bayes, ya que:
     · Naïve Bayes no entiende texto, solo números
     · Aprende: "¿Qué palabras indican spam vs. ham?"
    Fit sólo se aplica en test, ya que:
     · fit solo en X_train: Aprende el vocabulario
     · transform en X_test: Usa ese vocabulario aprendido
    Esto asegura que el modelo trata los datos de test como realmente nuevos.
    '''

    return X_train, X_test, y_train, y_test, vectorizer, X_train_dtm, X_test_dtm


# ============================================================================
# PARTE 4. ENTRENAMIENTO DEL MODELO
# ============================================================================

def train_model(X_train_dtm, y_train):
    """
    Entrena un modelo Multinomial Naïve Bayes.

    Crear el modelo con alpha=1.0 y ajustarlo con fit(...).

    Pregunta para el informe:
        ¿Por qué es importante usar alpha=1.0 en vez de dejar que una palabra
        no observada provoque probabilidad cero?
    """
    model = MultinomialNB(alpha=1.0)  # import MultinomialNB from sklearn.naive_bayes
    ''' 
    Crea una instancia del clasificador Multinomial Naïve Bayes con
     Laplace smoothing activado. 

    MultinomialNB: Es el clasificador Naïve Bayes diseñado
     específicamente para datos discretos con conteos (como bolsa de
      palabras).

    alpha=1.0 (Laplace smoothing):
     Sin smoothing (alpha=0):
      ·Si una palabra nunca apareció en spam durante entrenamiento →
        P(palabra|spam) = 0
      ·Entonces cualquier documento con esa palabra sería clasificado
        automáticamente como ham (probabilidad spam = 0)
      ·Problema: palabras nuevas tienen peso infinito
     Con smoothing (alpha=1.0):
      ·Se agrega 1 a la frecuencia de cada palabra, evitando probabilidades cero
      ·Mejora la generalización a palabras nuevas
    '''
    
    model.fit(X_train_dtm, y_train) # entrena el modelo Naive Bayes con los datos de entrenamiento. 
    # .fit(): método de scikit-learn que realiza el aprendizaje supervisado

    return model


# ============================================================================
# PARTE 5. EVALUACIÓN
# ============================================================================

def evaluate_model(model, X_test_dtm, y_test, output_dir: str | Path | None = None):
    
    y_pred = model.predict(X_test_dtm)

    print("\n" + "=" * 80)
    print("EVALUACIÓN DEL MODELO")
    print("=" * 80)

    # Métricas de clasificación
    print(classification_report(y_test, y_pred, target_names=["ham", "spam"]))

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)

    print("\nMatriz de confusión:")
    print(cm)

    # Representación gráfica
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["ham", "spam"])
    disp.plot()
    plt.title("Matriz de confusión - Clasificador Spam")
    plt.tight_layout()

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        fig_path = output_dir / "matriz_confusion_spam_base.png"

        plt.savefig(fig_path)
        print(f"\nFigura guardada en: {fig_path}")

    # Mostrar o cerrar figura
    backend = plt.get_backend().lower()
    if "agg" not in backend:
        plt.show()
    else:
        plt.close()

    return y_pred


# ============================================================================
# PARTE 6. PALABRAS MÁS ASOCIADAS AL SPAM
# ============================================================================

def get_top_spam_words(model, vectorizer, top_n: int = 5) -> pd.DataFrame:
    """
    Devuelve las palabras más asociadas a la clase spam.

    Idea:
        Puedes comparar las probabilidades logarítmicas aprendidas para spam y ham.
        Cuanto mayor sea la diferencia logP(word|spam) - logP(word|ham),
        más indicativa será esa palabra del spam.
    """
    # TODO: obtener el vocabulario
    feature_names = None

    # TODO: extraer las log-probabilidades para ham y spam
    log_prob_ham = None
    log_prob_spam = None

    # TODO: calcular una puntuación diferencial y ordenar
    score = None
    top_idx = None

    result = pd.DataFrame(
        {
            "word": None,
            "logP(word|spam)": None,
            "logP(word|ham)": None,
            "spam_minus_ham": None,
        }
    )
    return result


# ============================================================================
# PARTE 7. MENSAJES DE PRUEBA INVENTADOS POR EL ALUMNO
# ============================================================================

def classify_custom_messages(messages: list[str], model, vectorizer) -> pd.DataFrame:
    """
    Clasifica mensajes inventados manualmente.

    TODO:
        - Transformar los mensajes con el vectorizador.
        - Predecir la clase.
        - Obtener predict_proba.
        - Devolver una tabla clara con resultados.
    """
    X_new = None
    predicted_class = None
    predicted_proba = None

    results = pd.DataFrame(
        {
            "message": messages,
            "predicted_label": None,
            "P(ham)": None,
            "P(spam)": None,
        }
    )
    return results


# ============================================================================
# PROGRAMA PRINCIPAL
# ============================================================================

def main() -> None:
    """Ejecuta toda la práctica."""
    csv_path = Path(__file__).with_name("spam.csv")

    # 1. Cargar dataset
    df = load_dataset(csv_path)
    show_basic_info(df)

    # 2. Preparar datos
    (
        X_train,
        X_test,
        y_train,
        y_test,
        vectorizer,
        X_train_dtm,
        X_test_dtm,
    ) = prepare_data(df)

    # 3. Tamaño del vocabulario
    print("\n" + "=" * 80)
    print("INFORMACIÓN DEL VOCABULARIO")
    print("=" * 80)

    # mostrar el número de palabras distintas del vocabulario
    print(f"Tamaño del vocabulario: {len(vectorizer.get_feature_names_out())}")
    '''
    Imprime cuántas palabras distintas hay el dataset. 

    vectorizer:
    ·Instancia de CountVectorizer que transformó los mensajes de texto
      en números
    ·Contiene el "diccionario" de todas las palabras únicas que vio
      durante el entrenamiento

    get_feature_names_out():
    ·Método que devuelve un array con los nombres de todas las
      "features" (palabras)
    '''

    # 4. Entrenar modelo
    model = train_model(X_train_dtm, y_train)

    # 5. Mostrar probabilidades a priori
    print("\n" + "=" * 80)
    print("PROBABILIDADES A PRIORI")
    print("=" * 80)

    # Mostrar las probabilidades a priori
    # Extraer model.class_log_prior_, pasarlo a probabilidad normal con np.exp(...)
    # e imprimir P(ham) y P(spam).
    class_prior = np.exp(model.class_log_prior_)
    ''' 
    model.class_log_prior_:
    ·Atributo del modelo entrenado que contiene los logaritmos de las
      probabilidades previas

    np.exp(...):
    ·Función exponencial que invierte el logaritmo
    ·Convierte las probabilidades logarítmicas a probabilidades normales [0-1]
    '''

    print(f"P(ham) = {class_prior[0]:.4f}") # Muestra la probabilidad previa del 1º elem del array con 4 decimales
    print(f"P(spam) = {class_prior[1]:.4f}")

    # 6. Evaluación
    evaluate_model(model, X_test_dtm, y_test, output_dir=Path(__file__).parent)

    # 7. Palabras más asociadas al spam
    print("\n" + "=" * 80)
    print("PALABRAS MÁS CARACTERÍSTICAS DEL SPAM")
    print("=" * 80)
    top_words = get_top_spam_words(model, vectorizer, top_n=5)
    print(top_words)

    # 8. Prueba con mensajes inventados
    custom_messages = [
        "Hi, are we still meeting tomorrow at the library?",
        "Congratulations! You have won a free vacation. Claim your prize now!",
        "Hello, we have a special offer for you if you reply today.",
    ]

    print("\n" + "=" * 80)
    print("CLASIFICACIÓN DE MENSAJES DE PRUEBA")
    print("=" * 80)
    custom_results = classify_custom_messages(custom_messages, model, vectorizer)
    print(custom_results)

    # 9. Pregunta final para el informe
    print("\n" + "=" * 80)
    print("PREGUNTA FINAL")
    print("=" * 80)
    print(
        "Explica qué ocurre si intentas clasificar una palabra como 'oferta' "
        "cuando no ha aparecido en el entrenamiento y cómo ayuda el "
        "suavizado de Laplace en ese caso."
    )


if __name__ == "__main__":
    main()
