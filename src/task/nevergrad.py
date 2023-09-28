import numpy as np


def sphere(x):
    return np.sum(np.square(x))


def rastrigin(x):
    return 10.0 * len(x) + np.sum(x ** 2 - 10.0 * np.cos(2 * np.pi * x))


def rosenbrock(x):
    return sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x)))
