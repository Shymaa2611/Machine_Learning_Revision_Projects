import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import pickle as pk
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score
data=pd.read_csv('Heart/heart.csv')
print(data.head())
print(data.describe())
print(data["target"].value_counts())
print(data.groupby("target").mean())
X=data.drop("target",axis=1)
y=data["target"]
print(y)
Scalar=StandardScaler().fit(X)
X=Scalar.transform(X)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=SVC(kernel='poly')
classifier=model.fit(X_train,y_train)
print(classification_report(y_test,classifier.predict(X_test)))
print("accuracy = ", accuracy_score(y_test,classifier.predict(X_test)))
print(confusion_matrix(y_test,classifier.predict(X_test)))
#pk.dump(model,open('model.pkl','wb'))
#model=pk.load(open('model.pkl','rb'))
