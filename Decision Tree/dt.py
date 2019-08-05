from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np


data = np.asarray(pd.read_csv('data.csv', header=None))
# Assign the features to the variable X, and the labels to the variable y. 
X = data[:,0:2]
y = data[:,2]

model = DecisionTreeClassifier()
model.fit(X,y)

y_pred = model.predict(X)
print(y_pred)
acc = accuracy_score(y,y_pred)

print(acc)