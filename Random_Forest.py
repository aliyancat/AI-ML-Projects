from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np


url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)


df = df [["Pclass" , "Sex", "Age" , "Fare" , "Survived"]]
df.dropna(inplace=True)

df["Sex"] = df["Sex"].map({"male" :1 , "female":0})
 

X = df[["Pclass","Sex","Age","Fare"]]
Y = df["Survived"]

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=42)

model = DecisionTreeClassifier()
model.fit(X_train,Y_train)

y_pred = model.predict(X_test)

print(f"Accuracy " , accuracy_score(Y_test,y_pred))
print("Classification Report:\n", classification_report(Y_test, y_pred))

test_passenger = np.array([[2,0,28,25.0]])
prediction = model.predict(test_passenger)
print("Prediction (1=Survived, 0 = Didn't survive)", prediction[0])
