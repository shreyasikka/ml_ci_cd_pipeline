import pandas as pd
from sklearn.model_selection import train_test_split
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

#loading the saved model
loaded_model=pickle.load(model_file)

#making predictions on testing data
y_pred=model.predict(X_test)

#evaluation of model
accuracy=accuracy_score(y_test,y_pred)
print(f"Model Accuracy:{accuracy}")
conf_mat=confusion_matrix(y_test,y_pred)
print(f"Confusion Matrix:\n{conf_mat}")
class_report=classification_report(y_test,y_pred)
print(f"Classification Report:\n{class_report}")
