import numpy as np


def henon(vecx, param):
    x, y = vecx
    a = param["a"]
    b = param["b"]

    return np.array([1 - a * x**2 + y, b * x])


def henon_inverse(vecx, param):
    x, y = vecx
    a = param["a"]
    b = param["b"]

    return np.array([y / b, x + (a / b**2) * y**2 - 1])
