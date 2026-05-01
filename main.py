
from sklearn.datasets import load_iris
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



#AQUI SE CARGA EL DATASET
print("CARGA DEL DATASET")
dataset = load_iris()
df = pd.DataFrame(dataset.data, columns=dataset.feature_names)

df['target'] = dataset.target

print(df.head(5))

# CARGA DE LOS DATOS BÁSICOS
print("LOS DATOS BÁSICOS")
print(df.info())

print("VALORES VACÍOS")
print(df.isnull().sum())

print("DUPLICADOS")
print(df.duplicated().sum())


# Histogramas
print("LOS HISTOGRAMAS")
df.hist(figsize=(10, 8))
plt.show()

# MATRIZ DE CORRELACIÓN (HEATMAP)
print("HEATMAP")
sns.heatmap(df.corr(), annot=True)
plt.show()

# PAIRPLOT
print("PAIRPLOT")
sns.pairplot(df, hue='target')
plt.show()