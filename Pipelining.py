import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

iris = load_iris()

df = pd.DataFrame(iris.data, columns=iris.feature_names)

#scaler = StandardScaler()
#features_to_scale = ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
#scaled_data = scaler.fit_transform(df[features_to_scale])

x = iris.data
y = iris.target

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

pipeline = Pipeline([('scaler' ,StandardScaler()),('model', LogisticRegression())])

pipeline.fit(x_train,y_train)

accuracy = pipeline.score(x_test,y_test)
print(f"Accuracy {accuracy}")
predictions = pipeline.predict(x_test)

for i in range(5):
    actual_name = iris.target_names[y_test[i]]
    predicted_name = iris.target_names[predictions[i]]
    print(f"Flower {i+1}. Actual {actual_name}. Predicted = {predicted_name}.")