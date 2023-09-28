import numpy as np


def sphere(x):
    return np.sum(np.square(x))


def schwefel(x):
    return 418.9829 * len(x) - np.sum(x * np.sin(np.sqrt(np.abs(x))))


def rastrigin(x, a=10):
    return a * len(x) + sum(x_i ** 2 - a * np.cos(2 * np.pi * x_i) for x_i in x)


def rosenbrock(x):
    return sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2 + (1 - x[i]) ** 2 for i in range(len(x) - 1))
