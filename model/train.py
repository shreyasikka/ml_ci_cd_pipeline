import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

#loading the dataset
data=pd.read_csv("data\iris.csv")

#preprocessing the dataset
X=data.drop('species',axis=1)
y=data['species']

#splitting the dataset into training and testing sets
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

model=LogisticRegression(max_iter=200)

#model training
model.fit(X_train,y_train)

#saving the model
with open("iris_logistic_regression.pkl","wb") as model_file:
    pickle.dump(model,model_file)
