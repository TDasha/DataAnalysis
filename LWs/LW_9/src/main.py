from sklearn import cross_validation
from lib.data_analysis.algorithms.neighbors.KNeighborsClassifier import KNeighborsClassifier
from lib.data_analysis.algorithms.neighbors.KNeighborsRegressor import KNeighborsRegressor
import pandas as pd
from os import getcwd
import numpy as np

X_train = [
    [1, 1],
    [1, 2],
    [2, 4],
    [3, 5],
    [1, 0],
    [0, 0],
    [1, -2],
    [-1, 0],
    [-1, -2],
    [-2, -2]
]

y_train = [1, 1, 1, 1, 1, 2, 2, 2, 2, 2]

X_test = [
    [5, 5],
    [0, -1],
    [-5, -5]
]

y_test = [1, 2, 2]

knnClassify = KNeighborsClassifier(train_X=X_train, train_Y=y_train, test_X=X_test, k=2)
knnClassify.fit()
print("Score for issue # 1")
print(knnClassify.score(y_test))

knnRegressor = KNeighborsRegressor(train_X=X_train, train_Y=y_train, test_X=X_test, k=2)
knnRegressor.fit()
print(knnRegressor.predict())
print("Predict for issue # 1")
print("\n\n")
#############################################

data = pd.read_csv(getcwd() + "/data/data.csv")
X = data.drop("status", 1)
Y = data["status"]

X = np.array(X)
Y = np.array(Y)

X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X, Y, test_size=0.25, random_state=42)

knnClassify = KNeighborsClassifier(train_X=X_train, train_Y=Y_train, test_X=X_test, k=1)
knnClassify.fit()
print("Score for issue # 2")
print(knnClassify.score(Y_test))

knnRegressor = KNeighborsRegressor(train_X=X_train, train_Y=Y_train, test_X=X_test, k=1)
knnRegressor.fit()
print("Predict for issue # 2")
print(knnRegressor.predict())
