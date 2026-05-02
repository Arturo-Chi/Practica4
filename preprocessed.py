from sklearn.datasets import load_iris

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.preprocessing import StandardScaler

dataset = load_iris()

dataframe = pd.DataFrame(data= dataset.data, columns=dataset.feature_names)

dataframe['target'] = dataset.target

#Eliminación de los duplicados del dataframe
dataframe = dataframe.drop_duplicates()

x = dataframe.drop('target', axis = 1)
y = dataframe['target']

print("X ", x)
print("Y ", y)

"""

La separación de los datos sirve para determinar cuáles son las entradas
y salidas esperadas

x = (inputs) Lo que el modelo usa para aprender
y = (outputs) Lo que se quiere predecir

"""

scaler = StandardScaler()
print("ESCALER", scaler)

escalado_x = scaler.fit_transform(X=x)

print("ESCALADO X", escalado_x)

scaled_dataframe = pd.DataFrame(escalado_x, columns= x.columns)
print(" ")
print(" ")
print(dataframe.drop_duplicates().head())
print(" ")
print(" ")
print(scaled_dataframe.head())