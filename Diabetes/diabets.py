import numpy as np
import pandas as pd
from sklearn.preprocessing  import StandardScaler 
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
data=pd.read_csv('diabetes.csv')
# getting the statistical of data
print("  display first 5 Rows :  ")
print(data.head())
print("the Description of data : ")
print(data.describe())
print(" the data information : ")
print(data.info())
print(data.isna())
print(data["Outcome"].value_counts())
print(data.groupby("Outcome").mean())
X=data.drop("Outcome",axis=1)
y=data["Outcome"]
print(X.head())
print(y.head())
scalar=StandardScaler().fit(X)
X=scalar.transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=42)
print(X_train)
print(X_test)
print(y_train)
print(y_test)
model=SVC(kernel='linear')
classifier=model.fit(X_train,y_train)
print(classifier.predict(X_test))
print("train accuracy = ",accuracy_score(y_train,classifier.predict(X_train)))
print("test accuracy = ",accuracy_score(y_test,classifier.predict(X_test)))
print(classification_report(y_test,classifier.predict(X_test)))
print(confusion_matrix(y_test,classifier.predict(X_test)))
