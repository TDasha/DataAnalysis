import numpy as np


class Metrics:

    @staticmethod
    def euclide_distance(x, y):
        return np.linalg.norm(x - y)
