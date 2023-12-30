# -*- coding: utf-8 -*-
"""
Created on Sun Dec 24 09:33:23 2023

@author: ivoto
"""

# Redes Neuronales Artificiales

# Intalamos keras abiendo anaconda prompt e ejecutando conda install -c conda-forge keras

# Parte 1 - Pre Procesado de datos

# importar las librerías
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import tensorflow as tf

print(tf.__version__)


# Importar el data set
dataset = pd.read_csv("Churn_Modelling.csv")

X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

# Codificamos datos categoricos
# Codificamos
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
labelencoder_X_1.fit_transform(X[:, 1])
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
labelencoder_X_2.fit_transform(X[:, 2])
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
# Transformacion a V Dummy o ficticias
from sklearn.compose import ColumnTransformer

ct = ColumnTransformer(
    [("one_hot_encoder", OneHotEncoder(categories="auto"), [1])],
    remainder="passthrough",
)
X = np.array(ct.fit_transform(X), dtype=float)
# Elimino una columna para evitar multicolinealidad
X = X[:, 1:]


# Dividir el data set en conjunto de entrenamiento y conjunto de testing
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Escalado de variables
from sklearn.preprocessing import StandardScaler

sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# Parte 2 - Contruir RNA

# Importar Keras y librerias adicionales
import keras
from keras.models import Sequential
from keras.layers import Dense

# Inicializar la RNA
classifier = Sequential()

# Agregamos la capa de entrada y primera capa oculta
classifier.add(
    Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=11)
)

# Agregamos la segunda capa oculta
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))

# Agregamos la capa de salida
classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))

# Compilar la RNA
classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Ajustamos la RNA al Conjunto de Entrenamiento
classifier.fit(X_train, y_train, batch_size=10, epochs=100)

# Predicción de los resultados con el Conjunto de Testing
y_pred = classifier.predict(X_test)
y_pred = y_pred > 0.5  # Convierte los valores de probabilidad a verdadero o falso

# Elaborar una matriz de confusión
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
