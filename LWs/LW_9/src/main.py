# from lib.data_analysis.metrics.Metrics import Metrics
from lib.data_analysis.algorithms.neighbors.KNeighborsClassifier import KNeighborsClassifier
from lib.data_analysis.algorithms.neighbors.KNeighborsRegressor import KNeighborsRegressor
#
# print(KNeighborsClassifier([], [], 4))
# print(KNeighborsRegressor([], [], 6))
# print(Metrics.euclide_distance(12, -120))
#

# knn

import math
from collections import defaultdict
from operator import itemgetter

# Some handmade train/test data
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
print(knnClassify.score(y_test))

knnRegressor = KNeighborsRegressor(train_X=X_train, train_Y=y_train, test_X=X_test, k=2)
knnRegressor.fit()
print(knnRegressor.predict())