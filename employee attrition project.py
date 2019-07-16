import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
"""
f = open("employ.csv", "r")

X = []
Y = []
"""
import pandas as pd
binary = pd.read_csv("employ.csv")
array = binary.values


X = array[:,0:8]
Y = array[:,8]
uniqueY = list(set(Y))
print("X : ", X)
print("Y : ", Y)
print("Unique Y ", uniqueY)

Y = [uniqueY.index(val) for val in Y]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, random_state=0, test_size=0.3)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
#print("Test set predictions:\n", y_pred)

# Code to compare prediction values with expected values
print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
print("Test set score: {:.2f}".format(knn.score(X_test, y_test)))


