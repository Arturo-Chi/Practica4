
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler


#AQUI SE CARGA EL DATASET
print(" ")
print("CARGA DEL DATASET")
dataset = load_iris()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

df['target'] = dataset.target

print(df.head(150))

# CARGA DE LOS DATOS BÁSICOS
print(" ")
print("LOS DATOS BÁSICOS")
print(df.info())

print(" ")
print("VALORES VACÍOS")
print(df.isnull().sum())

print(" ")
print("DUPLICADOS")
print(df.duplicated())
print("TOTAL Valores Duplicados: ", df.duplicated().sum())


# Histogramas
print(" ")
print("LOS HISTOGRAMAS")
df.drop('target', axis=1).hist(figsize=(10, 8))
plt.show()

# MATRIZ DE CORRELACIÓN (HEATMAP)
print(" ")
print("HEATMAP")
sns.heatmap(df.corr(), annot=True)
plt.show()

# PAIRPLOT
print(" ")
print("PAIRPLOT")
sns.pairplot(df, hue='target')
plt.show()

#df.drop_duplicates()



#PREPROCESADO DE DATOS
# PREPROCESAMIENTO

print(" ")
print("PREPROCESAMIENTO")

# Eliminar duplicados
print("ELIMINANDO DUPLICADOS")
df = df.drop_duplicates()

# Separar variables
print("SEPARACIÓN DE DATOS")
X = df.drop('target', axis=1)
y = df['target']

# Escalado
from sklearn.preprocessing import StandardScaler

print("ESCALADO DE DATOS")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("PRIMERAS FILAS ESCALADAS")
print(X_scaled[:5])