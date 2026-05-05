from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd

from sklearn.datasets import load_iris
from preprocessed import DataFrameBuilder, Preprocessor

class Testing:

    def __init__(self):
        self.dataset =  load_iris()
        self.builder = DataFrameBuilder(dataset=self.dataset)
        self.dataframe = self.builder.getDataFrame()
        self.preprocessor = Preprocessor(self.dataframe)
        self.xp, self.y = self.preprocessor.process()
        self.xtrain = None
        self.ytrain = None
        self.xtest = None
        self.ytest = None
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.y_pred = None



    def train_test_values(self):
        self.xtrain, self.xtest, self.ytrain, self.ytest = train_test_split(
            self.xp,
            self.y,
            test_size=0.2,
            random_state=42
        )
    
    def test_predict(self):
        if self.xtrain is None or self.xtest is None or self.ytrain is None or self.ytest is None:
            self.train_test_values()
        
        self.model.fit(self.xtrain, self.ytrain)
        self.y_pred = self.model.predict(self.xtest)

    def matrix(self):
        if self.y_pred is None:
            self.test_predict()
        return confusion_matrix(self.ytest, self.y_pred)

    def clasificationReport(self):
        if self.y_pred is None:
            self.test_predict()
        return classification_report(self.ytest, self.y_pred)

    def accuracyScore(self):
        if self.y_pred is None:
            self.test_predict()
        return accuracy_score(self.ytest, self.y_pred)
    

    def evaluate(self):
        return {
            "matrix": self.matrix(),
            "classification_report": self.clasificationReport(),
            "accuracy_score": self.accuracyScore()
        }
    
    def predict_flower(self, sl, sw, pl, pw):
        if self.y_pred is None:
            self.test_predict()

        flower_df = pd.DataFrame(
            [[sl, sw, pl, pw]],
            columns= self.preprocessor.x.columns
        )
        flower_data_scaled = self.preprocessor.scaler.transform(flower_df)
        prediction = self.model.predict(flower_data_scaled)

        return {
            "class_number": int(prediction[0]),
            "class_name": self.dataset.target_names[prediction[0]]
        }

    


    






"""

dataset = load_iris()
builder = DataFrameBuilder(dataset=dataset)
dataframe = builder.getDataFrame()

preprocess = Preprocessor(dataframe)
xp, y = preprocess.process()


xtrain, xtest, ytrain, ytest = train_test_split(
    xp,
    y,
    test_size=0.2,
    random_state=42

)

model = KNeighborsClassifier(n_neighbors=3)
model.fit(xtrain, ytrain)

ypred = model.predict(xtest)

#print(confusion_matrix(ytest, ypred))
#print(classification_report(ytest, ypred))
#print(accuracy_score(ytest, ypred))

prueba = pd.DataFrame(
    [[0.2, 3,2, 4.1, 1.0]],
    columns= preprocess.x.columns
)

prueba_scaled = preprocess.scaler.transform(prueba)


prediccion = model.predict(prueba_scaled)
#print(prediccion[0])
print(dataset.target_names[prediccion[0]])
"""