import numpy as np


class _Metric(object):
    
    def call(self, o: np.ndarray, y: np.ndarray) -> bool:
        return False


class Accuracy(_Metric):

    def __init__(self, decimals: int = 0):
        self.decimals = decimals

    def call(self, o: np.ndarray, y: np.ndarray) -> bool:
        return np.all(np.round(o, self.decimals) == y)

class Delta(_Metric):

    def __init__(self, delta: float = 0.0):
        self.delta = delta

    def call(self, o: np.ndarray, y: np.ndarray) -> bool:
        return np.all(np.abs(o - y) <= self.delta)