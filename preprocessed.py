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

#print("X ", x)
#print("Y ", y)

"""

La separación de los datos sirve para determinar cuáles son las entradas
y salidas esperadas

x = (inputs) Lo que el modelo usa para aprender
y = (outputs) Lo que se quiere predecir

"""

scaler = StandardScaler()
#print("ESCALER", scaler)

escalado_x = scaler.fit_transform(X=x)

#print("ESCALADO X", escalado_x)

scaled_dataframe = pd.DataFrame(escalado_x, columns= x.columns)
#print(" ")
#print(" ")
#print(dataframe.drop_duplicates().head())
#print(" ")
#print(" ")
#print(scaled_dataframe.head())



"""
df1 = dataframe[['sepal length (cm)']]
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name="sheet1", index=False)
"""

class DataFrameBuilder:
    def __init__ (self, dataset):
        self.dataframe = pd.DataFrame(dataset.data, columns= dataset.feature_names)
        self.dataframe['target'] = dataset.target

    def getDataFrame(self):
        return self.dataframe


class Preprocessor:
    def __init__(self, dataFrame: pd.DataFrame):
        self.dataFrame = dataFrame.drop_duplicates()
        self.x = None
        self.y = None
        self.x_scaled = None
        self.scaler = StandardScaler()

    def values_separation(self):

        # x se refiera a las inputs que se deben de recibir
        # y a lo que se quiere predecir


        self.x = self.dataFrame.drop('target', axis = 1)
        self.y = self.dataFrame['target']
        return self.x, self.y

    def scale(self):
        if self.x is None:
            self.values_separation()
        
        self.x_scaled = self.scaler.fit_transform(X=x)
        return self.x_scaled

 
    def process (self):
        self.values_separation()
        self.scale()
        return self.x_scaled, self.y