import abc
import copy
import random

import numpy as np

from ..selection.filters import Filter


class GeneticOperator(object):

    def __init__(self, genotype_filter: Filter):
        self.genotype_filter = genotype_filter

    def apply(self, *args):
        new_born = self.propose(*args)
        while not self.genotype_filter(new_born):
            new_born = self.propose(*args)
        return new_born

    @abc.abstractmethod
    def propose(self, *args):
        pass

    @abc.abstractmethod
    def get_arity(self) -> int:
        pass

    @classmethod
    def create_genetic_operator(cls, name: str, genotype_filter: Filter, **kwargs):
        if name == "gaussian_mut":
            return GaussianMutation(genotype_filter=genotype_filter, mu=kwargs["mu"], sigma=kwargs["sigma"])
        elif name == "insert_mut":
            return InsertMutation(genotype_filter=genotype_filter, lower=kwargs["lower"], upper=kwargs["upper"])
        elif name == "delete_mut":
            return DeleteMutation(genotype_filter=genotype_filter)
        elif name == "geometric_cx":
            return GeometricCrossover(genotype_filter=genotype_filter, upper=kwargs["upper"], lower=kwargs["lower"],
                                      mu=kwargs["mu"], sigma=0.1)
        elif name == "uniform_cx":
            return UniformCrossover(genotype_filter=genotype_filter, upper=kwargs["upper"], lower=kwargs["lower"],
                                    mu=kwargs["mu"], sigma=1000)
        raise ValueError("Invalid genetic operator name: {}".format(name))


class GaussianMutation(GeneticOperator):

    def __init__(self, genotype_filter, mu: float, sigma: float):
        super().__init__(genotype_filter)
        self.mu = mu
        self.sigma = sigma

    def propose(self, *args) -> np.ndarray:
        if len(args) != 1:
            raise ValueError("Need one parent for mutation")
        child = copy.deepcopy(args[0][0])
        mutation = np.random.normal(self.mu, self.sigma, len(child))
        for i in range(len(child)):
            child[i] += mutation[i]
        return child

    def get_arity(self):
        return 1


class InsertMutation(GeneticOperator):

    def __init__(self, genotype_filter, lower, upper):
        super().__init__(genotype_filter)
        self.lower = lower
        self.upper = upper

    def propose(self, *args) -> np.ndarray:
        if len(args) != 1:
            raise ValueError("Need one parent for mutation")
        child = copy.deepcopy(args[0][0])
        return np.append(child, np.random.uniform(low=self.lower, high=self.upper, size=1))

    def get_arity(self):
        return 1


class DeleteMutation(GeneticOperator):

    def __init__(self, genotype_filter):
        super().__init__(genotype_filter)

    def propose(self, *args) -> np.ndarray:
        if len(args) != 1:
            raise ValueError("Need one parent for mutation")
        child = copy.deepcopy(args[0][0])
        return np.delete(child, random.randint(0, len(child) - 1))

    def get_arity(self):
        return 1


class GeometricCrossover(GeneticOperator):

    def __init__(self, genotype_filter, upper: float, lower: float, mu: float, sigma: float):
        super().__init__(genotype_filter)
        self.upper = upper
        self.lower = lower
        self.mutation = GaussianMutation(genotype_filter=genotype_filter, mu=mu, sigma=sigma)

    def propose(self, *args) -> np.ndarray:
        if len(args[0]) != 2:
            raise ValueError("Need two parents for crossover")
        parent1, parent2 = args[0]
        return self.mutation.apply([parent1 + (parent2 - parent1) * (np.random.random(size=len(parent1)) *
                                                                     (self.upper - self.lower) + self.lower)])

    def get_arity(self):
        return 2


class UniformCrossover(GeneticOperator):

    def __init__(self, genotype_filter, upper: float, lower: float, mu: float, sigma: float):
        super().__init__(genotype_filter)
        self.upper = upper
        self.lower = lower
        self.mutation = GaussianMutation(genotype_filter=genotype_filter, mu=mu, sigma=sigma)

    def propose(self, *args) -> np.ndarray:
        if len(args[0]) != 2:
            raise ValueError("Need two parents for crossover")
        parent1, parent2 = args[0]
        child = np.copy(parent1)
        idx = np.where(np.random.rand(child.size) > 0.5)
        child[idx] = parent2[idx]
        return child

    def get_arity(self):
        return 2
