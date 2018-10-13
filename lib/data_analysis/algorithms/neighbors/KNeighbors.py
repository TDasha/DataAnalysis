from abc import ABC
from lib.data_analysis.metrics.Metrics import Metrics
from collections import defaultdict
from operator import itemgetter


class KNeighbors(ABC):

    def __init__(self, train_X, train_Y, test_X, k: int):
        self.__train_X = train_X
        self.__train_Y = train_Y
        self.__test_X = test_X
        self._test_Answers = []
        self.__k = k

    def fit(self):
        y_test = []
        for test_row in self.__test_X:
            eucl_dist = [Metrics.euclide_distance(train_row, test_row) for train_row in self.__train_X]
            sorted_eucl_dist = sorted(eucl_dist)
            closest_knn = [eucl_dist.index(sorted_eucl_dist[i]) for i in range(0, self.__k)] if self.__k > 1 else [
                eucl_dist.index(min(eucl_dist))]
            closest_labels_knn = [self.__train_Y[x] for x in closest_knn]
            y_test.append(self.__get_most_common_item(closest_labels_knn))
        self._test_Answers = y_test

    def __get_most_common_item(self, array):
        count_dict = defaultdict(int)
        for key in array:
            count_dict[key] += 1
        key, count = max(count_dict.items(), key=itemgetter(1))
        return key

