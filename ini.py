from sklearn.datasets import load_iris
from preprocessed import DataFrameBuilder
from preprocessed import Preprocessor


dataset = load_iris()
builder = DataFrameBuilder(dataset=dataset)
df = builder.getDataFrame()

preprocess = Preprocessor(df)
xs, y = preprocess.process()
print(xs, y)


