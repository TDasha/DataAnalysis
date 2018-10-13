import numpy as np


class Metrics:

    @staticmethod
    def euclide_distance(x, y):
        ans = []
        for i in range(len(x)):
            ans.append(np.linalg.norm(x[i] - y[i]))
        return ans
