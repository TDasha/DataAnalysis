from lib.data_analysis.metrics.Metrics import Metrics
from lib.data_analysis.algorithms.neighbors.KNeighborsClassifier import KNeighborsClassifier
from lib.data_analysis.algorithms.neighbors.KNeighborsRegressor import KNeighborsRegressor

print(KNeighborsClassifier([], [], 4))
print(KNeighborsRegressor([], [], 6))
print(Metrics.euclide_distance(12, -120))

