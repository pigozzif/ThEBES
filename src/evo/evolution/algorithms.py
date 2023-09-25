import abc
import random
from abc import ABC
from typing import Dict

import numpy as np
from sklearn.decomposition import PCA
from scipy.linalg import cholesky
from numpy.linalg import LinAlgError
from numpy.random import standard_normal

from .objectives import ObjectiveDict
from .operators.operator import GeneticOperator
from .optimizers import Adam
from .selection.filters import Filter
from .selection.selector import Selector
from ..representations.factory import GenotypeFactory
from ..representations.population import Population, Individual, Comparator
from ..utils.utilities import weighted_random_by_dct, exp_decay, reshape_fitness


class StochasticSolver(abc.ABC):

    def __init__(self, seed, num_params, pop_size):
        self.seed = seed
        self.num_params = num_params
        self.pop_size = pop_size

    @abc.abstractmethod
    def ask(self):
        pass

    @abc.abstractmethod
    def tell(self, fitness_list):
        pass

    @abc.abstractmethod
    def result(self):
        pass

    @classmethod
    def create_solver(cls, name: str, **kwargs):
        if name == "ga":
            return GeneticAlgorithm(**kwargs)
        elif name == "afpo":
            return AFPO(**kwargs)
        elif name == "nsgaii":
            return NSGAII(**kwargs)
        elif name == "openes":
            return OpenAIES(**kwargs)
        elif name == "thebes":
            return ThEBES(**kwargs)
        elif name == "rs":
            return RandomSearch(**kwargs)
        raise ValueError("Invalid solver name: {}".format(name))


class PopulationBasedSolver(StochasticSolver, ABC):

    def __init__(self, seed: int,
                 num_params: int,
                 pop_size: int,
                 genotype_factory: str,
                 objectives_dict: ObjectiveDict,
                 remap: bool,
                 genetic_operators: Dict[str, float],
                 comparator: str,
                 genotype_filter: str = None,
                 **kwargs):
        super().__init__(seed=seed, num_params=num_params, pop_size=pop_size)
        self.pop_size = pop_size
        self.remap = remap
        self.pop = Population(pop_size=pop_size,
                              genotype_factory=GenotypeFactory.create_factory(name=genotype_factory,
                                                                              genotype_filter=Filter.create_filter(
                                                                                  genotype_filter), **kwargs),
                              objectives_dict=objectives_dict,
                              comparator=comparator)
        self.genetic_operators = {GeneticOperator.create_genetic_operator(name=k,
                                                                          genotype_filter=Filter.create_filter(
                                                                              genotype_filter), **kwargs):
                                      v for k, v in genetic_operators.items()}

    def get_best(self) -> Individual:
        return self.pop.get_best()


class RandomSearch(PopulationBasedSolver):

    def __init__(self, seed, num_params, pop_size, objectives_dict, range_min, range_max):
        super().__init__(seed=seed,
                         num_params=num_params,
                         pop_size=pop_size,
                         genotype_factory="uniform_float",
                         objectives_dict=objectives_dict,
                         remap=False,
                         genetic_operators={},
                         genotype_filter="none",
                         comparator="lexicase",
                         range=(range_min, range_max),
                         n=num_params)
        self.best_fitness = float("inf")
        self.best_genotype = None

    def ask(self):
        if self.pop.gen > 0:
            for g in self.pop.init_random_individuals(n=self.pop_size):
                self.pop.add_individual(g)
        return [ind.genotype for ind in self.pop]

    def tell(self, fitness_list):
        for ind, f in zip([ind for ind in self.pop if not ind.evaluated], fitness_list):
            ind.fitness = {"fitness": f}
            ind.evaluated = True
        best_idx = np.argmin(fitness_list)
        if self.best_fitness <= fitness_list[best_idx]:
            self.best_fitness = fitness_list[best_idx]
            self.best_genotype = self.pop[best_idx]
        self.pop.clear()

    def result(self):
        return self.best_genotype, self.best_fitness


class GeneticAlgorithm(PopulationBasedSolver):

    def __init__(self, seed, num_params, pop_size, genotype_factory, objectives_dict, survival_selector: str,
                 parent_selector: str, offspring_size: int, overlapping: bool, remap, genetic_operators,
                 genotype_filter, **kwargs):
        super().__init__(seed=seed,
                         num_params=num_params,
                         pop_size=pop_size,
                         genotype_factory=genotype_factory,
                         objectives_dict=objectives_dict,
                         remap=remap,
                         genetic_operators=genetic_operators,
                         genotype_filter=genotype_filter,
                         comparator="lexicase",
                         **kwargs)
        self.survival_selector = Selector.create_selector(name=survival_selector, **kwargs)
        self.parent_selector = Selector.create_selector(name=parent_selector, **kwargs)
        self.offspring_size = offspring_size
        self.overlapping = overlapping

    def _build_offspring(self) -> list:
        children_genotypes = []
        while len(children_genotypes) < self.offspring_size:
            operator = weighted_random_by_dct(dct=self.genetic_operators)
            parents = [parent.genotype for parent in self.parent_selector.select(population=self.pop,
                                                                                 n=operator.get_arity())]
            children_genotypes.append(operator.apply(tuple(parents)))
        return children_genotypes

    def _trim_population(self) -> None:
        while len(self.pop) > self.pop_size:
            self.pop.remove_individual(self.survival_selector.select(population=self.pop, n=1)[0])

    def ask(self):
        if self.pop.gen != 0:
            for child_genotype in self._build_offspring():
                self.pop.add_individual(genotype=child_genotype)
        return [ind.genotype for ind in self.pop if not ind.evaluated]

    def tell(self, fitness_list):
        for ind, f in zip([ind for ind in self.pop if not ind.evaluated], fitness_list):
            ind.fitness = {"fitness": f}
            ind.evaluated = True
        if self.pop.gen != 0:
            self._trim_population()
        self.pop.gen += 1

    def result(self):
        best = self.get_best()
        return best.genotype, best.fitness["fitness"]


class AFPO(GeneticAlgorithm):

    def __init__(self, seed, num_params, pop_size, genotype_factory, objectives_dict, offspring_size: int,
                 remap, genetic_operators, genotype_filter, **kwargs):
        super().__init__(seed=seed,
                         num_params=num_params,
                         pop_size=pop_size,
                         genotype_factory=genotype_factory,
                         objectives_dict=objectives_dict,
                         remap=remap,
                         genetic_operators=genetic_operators,
                         survival_selector="tournament_pareto",
                         parent_selector="tournament",
                         offspring_size=offspring_size - 1,
                         overlapping=True,
                         genotype_filter=genotype_filter,
                         **kwargs)
        self.pareto_comparator = Comparator.create_comparator(name="pareto", objective_dict=self.pop.objectives_dict)

    def _select_individual(self, population: Population) -> Individual:
        ind1, ind2 = tuple(population.sample(n=2))
        c = self.pareto_comparator.compare(ind1=ind1, ind2=ind2)
        if c == -1:
            return ind1
        elif c == 1:
            return ind2
        return random.choice([ind1, ind2])

    def _build_offspring(self) -> list:
        children_genotypes = super()._build_offspring()
        children_genotypes.extend(self.pop.init_random_individuals(n=1))
        return children_genotypes

    def _trim_population(self) -> None:
        while len(self.pop) > self.pop_size:
            self.pop.remove_individual(self._select_individual(population=self.pop))

    def tell(self, fitness_list):
        for ind, f in zip([ind for ind in self.pop if not ind.evaluated], fitness_list):
            ind.fitness = {"fitness": f, "age": 0}
            ind.evaluated = True
        if self.pop.gen != 0:
            self._trim_population()
        self.pop.gen += 1
        self.pop.update_ages()


class OpenAIES(PopulationBasedSolver):

    def __init__(self, seed, num_params, pop_size, objectives_dict, sigma, sigma_decay, sigma_limit, l_rate_init,
                 l_rate_decay, l_rate_limit, **kwargs):
        super().__init__(seed=seed,
                         num_params=num_params,
                         pop_size=pop_size,
                         genotype_factory="uniform_float",
                         objectives_dict=objectives_dict,
                         remap=False,
                         genetic_operators={},
                         genotype_filter="none",
                         comparator="lexicase",
                         range=(-1, 1),
                         n=num_params,
                         **kwargs)
        self.sigma = sigma
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.optimizer = Adam(num_dims=num_params, l_rate_init=l_rate_init, l_rate_decay=l_rate_decay,
                              l_rate_limit=l_rate_limit)
        self.mode = np.zeros(num_params)
        self.best_fitness = float("inf")
        self.best_genotype = None

    def _sample_offspring(self):
        z_plus = np.random.normal(loc=0.0, scale=self.sigma, size=(self.pop_size // 2, self.num_params))
        z = np.concatenate([z_plus, -1.0 * z_plus])
        return [self.mode + x * self.sigma for x in z]

    def ask(self):
        self.pop.clear()
        for child_genotype in self._sample_offspring():
            self.pop.add_individual(genotype=child_genotype)
        return [ind.genotype for ind in self.pop if not ind.evaluated]

    def _update_mode(self) -> None:
        noise = np.array([(x.genotype - self.mode) / self.sigma for x in self.pop])
        fitness = np.array([x.fitness["fitness"] for x in self.pop])
        reshaped_fitness = reshape_fitness(fitness=fitness)
        best_idx = np.argmin(reshaped_fitness)
        if self.best_fitness >= fitness[best_idx]:
            self.best_fitness, self.best_genotype = fitness[best_idx], (noise[best_idx] * self.sigma) + self.mode
        theta_grad = (1.0 / (self.pop_size * self.sigma)) * np.dot(noise.T, reshaped_fitness)
        self.mode = self.optimizer.optimize(mean=self.mode,
                                            t=self.pop.gen,
                                            theta_grad=theta_grad)

    def tell(self, fitness_list):
        for ind, f in zip([ind for ind in self.pop if not ind.evaluated], fitness_list):
            ind.fitness = {"fitness": f}
            ind.evaluated = True
        self._update_mode()
        for ind in self.pop:
            self.pop.remove_individual(ind=ind)
        self.sigma = exp_decay(self.sigma, self.sigma_decay, self.sigma_limit)
        self.pop.gen += 1

    def result(self):
        return self.best_genotype, self.best_fitness


class ThEBES(OpenAIES):

    def __init__(self, seed, num_params, pop_size, objectives_dict, sigma, sigma_decay, sigma_limit, l_rate_init,
                 l_rate_decay, l_rate_limit, **kwargs):
        super().__init__(seed, num_params, pop_size, objectives_dict, sigma, sigma_decay, sigma_limit, l_rate_init,
                         l_rate_decay, l_rate_limit, **kwargs)

    def _update_mode(self) -> None:
        super()._update_mode()
        self.mode += np.random.normal(loc=0.0, scale=np.sqrt(self.optimizer.l_rate) * self.sigma, size=self.num_params)


class CMAES(StochasticSolver):

    def __init__(self, seed, num_params, pop_size, sigma_init):
        super().__init__(seed, num_params, pop_size)
        self.num_params = num_params
        self.sigma_init = sigma_init
        self.solutions = None
        self.it = 0
        self.best_fitness = float("inf")
        self.best_genotype = None

        import cma
        self.es = cma.CMAEvolutionStrategy(self.num_params * [0], self.sigma_init, {"popsize": self.pop_size})

    def ask(self):
        self.solutions = self.es.ask()
        self.it += 1
        return self.solutions

    def tell(self, fitness_list):
        self.es.tell(self.solutions, [-f for f in fitness_list])

    def result(self):
        r = self.es.result
        return r[0], r[1]

    def get_num_evaluated(self):
        return self.it * self.pop_size


class xNES(StochasticSolver):

    def __init__(self, seed, num_params, pop_size, sigma, l_rate=0.01):
        super().__init__(seed, num_params, pop_size)
        self.solutions = None
        self.zs = None
        self.sigma = sigma
        self.eta_mu = 1
        self.eta_A = 0.6 * (3 + np.log(num_params)) * num_params ** -1.5
        self.us = np.maximum(0, np.log(pop_size / 2 + 1) - np.log(1 + np.arange(pop_size)))
        self.us /= np.sum(self.us)
        self.us -= 1 / pop_size
        self.mu = np.zeros(num_params)
        self.cov_matrix = np.eye(num_params) * sigma
        self.l_rate = l_rate
        self.cov_matrix = np.eye(num_params)
        self.best_fitness = float("inf")
        self.best_genotype = None

    def ask(self):
        self.zs = np.array([np.random.normal(0, 1, self.num_params) for _ in range(self.pop_size)])
        self.solutions = np.array([self.mu + np.dot(self.cov_matrix, self.zs[i]) for i in range(self.pop_size)])
        return self.solutions

    def tell(self, fitness_list):
        order = np.argsort(fitness_list)
        self.zs = self.zs[order]
        gd = np.dot(self.us, self.zs)
        self.mu += self.eta_mu * np.dot(self.cov_matrix, gd)
        best_idx = np.argmin(fitness_list)
        if self.best_fitness >= fitness_list[best_idx]:
            self.best_fitness, self.best_genotype = fitness_list[best_idx], self.solutions[best_idx]
        gm = np.dot(
            np.array([np.outer(z, z).T - np.eye(self.num_params) for z in self.zs]).T,
            self.us)
        gs = np.trace(gm) / self.num_params
        gb = gm - gs * np.eye(self.num_params)
        self.cov_matrix *= np.exp(0.5 * self.eta_A * gs) * np.exp(0.5 * self.eta_A * gb)

    def result(self):
        return self.best_genotype, self.best_fitness


class sNES(StochasticSolver):

    def __init__(self, seed, num_params, pop_size, init_l_rate=1.0, init_sigma=1.0):
        super().__init__(seed, num_params, pop_size)
        self.mean_l_rate = init_l_rate
        self.cov_l_rate = 0.6 * (3 + np.log(self.num_params)) / 3 / np.sqrt(self.num_params)
        self.sigmas = np.full(self.num_params, fill_value=init_sigma)
        self.mean = np.zeros(num_params)
        self.cov_matrix = np.eye(num_params)
        self.noise = None
        self.solutions = None
        self.best_genotype = None
        self.best_fitness = float("inf")

    def _compute_utilities(self, fitness_list):
        ranks = np.maximum(0.0, np.log(self.pop_size / 2 + 1) - np.log(self.pop_size - np.argsort(fitness_list)))
        ranks /= np.sum(ranks)
        ranks -= 1 / self.pop_size
        return ranks

    def ask(self):
        self.noise = np.random.multivariate_normal(np.zeros(self.num_params), np.diag(self.sigmas), self.pop_size)
        self.solutions = self.mean + self.sigmas * self.noise
        return self.solutions

    def tell(self, fitness_list):
        self.noise = self.noise[np.argsort(fitness_list)]
        us = self._compute_utilities(fitness_list=fitness_list)
        gradient_mean = np.dot(us, self.noise)
        gradient_cov = np.dot(us, (self.noise ** 2 - 1))
        best_idx = np.argmin(fitness_list)
        if self.best_fitness >= fitness_list[best_idx]:
            self.best_fitness, self.best_genotype = fitness_list[best_idx], self.solutions[best_idx]
        self.mean += self.mean_l_rate * self.sigmas * gradient_mean
        self.sigmas *= np.exp(0.5 * self.cov_l_rate * gradient_cov)

    def result(self):
        return self.best_genotype, self.best_fitness


class CRFMNES(StochasticSolver):

    def __init__(self, seed, num_params, pop_size, sigma, **kwargs):
        super().__init__(seed, num_params, pop_size)
        self.m = np.zeros([num_params, 1])
        self.sigma = sigma
        assert (self.pop_size > 0 and self.pop_size % 2 == 0), f"The value of 'lamb' must be an even, positive " \
                                                               f"integer greater than 0 "

        self.v = kwargs.get('v', np.random.randn(self.num_params, 1) / np.sqrt(self.num_params))
        self.D = np.ones([self.num_params, 1])

        self.w_rank_hat = (np.log(self.pop_size / 2 + 1) - np.log(np.arange(1, self.pop_size + 1))).reshape(
            self.pop_size, 1)
        self.w_rank_hat[np.where(self.w_rank_hat < 0)] = 0
        self.w_rank = self.w_rank_hat / sum(self.w_rank_hat) - (1. / self.pop_size)
        self.mueff = 1 / ((self.w_rank + (1 / self.pop_size)).T @ (self.w_rank + (1 / self.pop_size)))[0][0]
        self.cs = (self.mueff + 2.) / (self.num_params + self.mueff + 5.)
        self.cc = (4. + self.mueff / self.num_params) / (self.num_params + 4. + 2. * self.mueff / self.num_params)
        self.c1_cma = 2. / ((self.num_params + 1.3) ** 2 + self.mueff)
        # initialization
        self.chiN = np.sqrt(self.num_params) * (
                1. - 1. / (4. * self.num_params) + 1. / (21. * self.num_params * self.num_params))
        self.pc = np.zeros([self.num_params, 1])
        self.ps = np.zeros([self.num_params, 1])
        # distance weight parameter
        self.h_inv = self.get_h_inv()
        self.alpha_dist = lambda lambF: self.h_inv * min(1., np.sqrt(float(self.pop_size) / self.num_params)) * np.sqrt(
            float(lambF) / self.pop_size)
        self.w_dist_hat = lambda z, lambF: np.exp(self.alpha_dist(lambF) * np.linalg.norm(z))
        # learning rate
        self.eta_m = 1.0
        self.eta_move_sigma = 1.
        self.eta_stag_sigma = lambda lambF: np.tanh(
            (0.024 * lambF + 0.7 * self.num_params + 20.) / (self.num_params + 12.))
        self.eta_conv_sigma = lambda lambF: 2. * np.tanh(
            (0.025 * lambF + 0.75 * self.num_params + 10.) / (self.num_params + 4.))
        self.c1 = lambda lambF: self.c1_cma * (self.num_params - 5) / 6 * (float(lambF) / self.pop_size)
        self.eta_B = lambda lambF: np.tanh(
            (min(0.02 * lambF, 3 * np.log(self.num_params)) + 5) / (0.23 * self.num_params + 25))

        self.g = 0
        self.no_of_evals = 0

        self.idxp = np.arange(self.pop_size / 2, dtype=int)
        self.idxm = np.arange(self.pop_size / 2, self.pop_size, dtype=int)
        self.z = np.zeros([self.num_params, self.pop_size])
        self.solutions = None
        self.best_fitness = float('inf')
        self.best_genotype = None

    def get_h_inv(self):
        f = lambda a, b: ((1. + a * a) * np.exp(a * a / 2.) / 0.24) - 10. - self.num_params
        f_prime = lambda a: (1. / 0.24) * a * np.exp(a * a / 2.) * (3. + a * a)
        h_inv = 6.0
        while abs(f(h_inv, self.num_params)) > 1e-10:
            last = h_inv
            h_inv = h_inv - 0.5 * (f(h_inv, self.num_params) / f_prime(h_inv))
            if abs(h_inv - last) < 1e-16:
                # Exit early since no further improvements are happening
                break
        return h_inv

    def sort_indices_by(self, evals, z):
        lam = len(evals)
        sorted_indices = np.argsort(evals)
        sorted_evals = evals[sorted_indices]
        no_of_feasible_solutions = np.where(sorted_evals != np.inf)[0].size
        if no_of_feasible_solutions != lam:
            infeasible_z = z[:, np.where(evals == np.inf)[0]]
            distances = np.sum(infeasible_z ** 2, axis=0)
            infeasible_indices = sorted_indices[no_of_feasible_solutions:]
            indices_sorted_by_distance = np.argsort(distances)
            sorted_indices[no_of_feasible_solutions:] = infeasible_indices[indices_sorted_by_distance]
        return sorted_indices

    def ask(self):
        zhalf = np.random.randn(self.num_params, int(self.pop_size / 2))
        self.z[:, self.idxp] = zhalf
        self.z[:, self.idxm] = -zhalf
        self.normv = np.linalg.norm(self.v)
        self.normv2 = self.normv ** 2
        self.vbar = self.v / self.normv
        self.y = self.z + (np.sqrt(1 + self.normv2) - 1) * self.vbar @ (self.vbar.T @ self.z)
        self.solutions = self.m + self.sigma * self.y * self.D
        return self.solutions.T

    def tell(self, fitness_list):
        fitness_list = np.array(fitness_list)
        sorted_indices = self.sort_indices_by(fitness_list, self.z)
        best_eval_id = sorted_indices[0]
        f_best = fitness_list[best_eval_id]
        x_best = self.solutions[:, best_eval_id]
        self.z = self.z[:, sorted_indices]
        y = self.y[:, sorted_indices]
        x = self.solutions[:, sorted_indices]

        self.no_of_evals += self.pop_size
        self.g += 1
        if f_best < self.best_fitness:
            self.best_fitness = f_best
            self.best_genotype = x_best

        # This operation assumes that if the solution is infeasible, infinity comes in as input.
        lambF = np.sum(fitness_list < np.finfo(float).max)

        # evolution path p_sigma
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2. - self.cs) * self.mueff) * (self.z @ self.w_rank)
        ps_norm = np.linalg.norm(self.ps)
        # distance weight
        w_tmp = np.array(
            [self.w_rank_hat[i] * self.w_dist_hat(np.array(self.z[:, i]), lambF) for i in
             range(self.pop_size)]).reshape(
            self.pop_size, 1)
        weights_dist = w_tmp / sum(w_tmp) - 1. / self.pop_size
        # switching weights and learning rate
        weights = weights_dist if ps_norm >= self.chiN else self.w_rank
        eta_sigma = self.eta_move_sigma if ps_norm >= self.chiN else self.eta_stag_sigma(
            lambF) if ps_norm >= 0.1 * self.chiN else self.eta_conv_sigma(lambF)
        # update pc, m
        wxm = (x - self.m) @ weights
        self.pc = (1. - self.cc) * self.pc + np.sqrt(self.cc * (2. - self.cc) * self.mueff) * wxm / self.sigma
        self.m += self.eta_m * wxm
        # calculate s, t
        # step1
        normv4 = self.normv2 ** 2
        exY = np.append(y, self.pc / self.D, axis=1)  # dim x lamb+1
        yy = exY * exY  # dim x lamb+1
        ip_yvbar = self.vbar.T @ exY
        yvbar = exY * self.vbar  # dim x lamb+1. exYのそれぞれの列にvbarがかかる
        gammav = 1. + self.normv2
        vbarbar = self.vbar * self.vbar
        alphavd = np.min(
            [1, np.sqrt(normv4 + (2 * gammav - np.sqrt(gammav)) / np.max(vbarbar)) / (2 + self.normv2)])  # scalar
        t = exY * ip_yvbar - self.vbar * (ip_yvbar ** 2 + gammav) / 2  # dim x lamb+1
        b = -(1 - alphavd ** 2) * normv4 / gammav + 2 * alphavd ** 2
        H = np.ones([self.num_params, 1]) * 2 - (b + 2 * alphavd ** 2) * vbarbar  # dim x 1
        invH = H ** (-1)
        s_step1 = yy - self.normv2 / gammav * (yvbar * ip_yvbar) - np.ones(
            [self.num_params, self.pop_size + 1])  # dim x lamb+1
        ip_vbart = self.vbar.T @ t  # 1 x lamb+1
        s_step2 = s_step1 - alphavd / gammav * (
                (2 + self.normv2) * (t * self.vbar) - self.normv2 * vbarbar @ ip_vbart)  # dim x lamb+1
        invHvbarbar = invH * vbarbar
        ip_s_step2invHvbarbar = invHvbarbar.T @ s_step2  # 1 x lamb+1
        s = (s_step2 * invH) - b / (
                1 + b * vbarbar.T @ invHvbarbar) * invHvbarbar @ ip_s_step2invHvbarbar  # dim x lamb+1
        ip_svbarbar = vbarbar.T @ s  # 1 x lamb+1
        t = t - alphavd * ((2 + self.normv2) * (s * self.vbar) - self.vbar @ ip_svbarbar)  # dim x lamb+1
        # update v, D
        exw = np.append(self.eta_B(lambF) * weights, np.array([self.c1(lambF)]).reshape(1, 1),
                        axis=0)  # lamb+1 x 1
        self.v = self.v + (t @ exw) / self.normv
        self.D = self.D + (s @ exw) * self.D
        # calculate detA
        nthrootdetA = \
            np.exp(np.sum(np.log(self.D)) / self.num_params + np.log(1 + self.v.T @ self.v) / (2 * self.num_params))[0][
                0]
        self.D = self.D / nthrootdetA
        # update sigma
        G_s = np.sum((self.z * self.z - np.ones([self.num_params, self.pop_size])) @ weights) / self.num_params
        self.sigma = self.sigma * np.exp(eta_sigma / 2 * G_s)

    def result(self):
        return self.best_genotype, self.best_fitness


class ASEBO(StochasticSolver):

    def __init__(self, seed, num_params, pop_size, subspace_dims, l_rate_init, l_rate_decay, l_rate_limit, sigma_init,
                 sigma_decay, sigma_limit):
        super().__init__(seed, num_params, pop_size)
        self.it = 0
        self.optimizer = Adam(num_dims=num_params, l_rate_init=l_rate_init, l_rate_decay=l_rate_decay,
                              l_rate_limit=l_rate_limit)
        self.subspace_dims = min(subspace_dims, self.num_params)
        self.lrate_init = l_rate_init
        self.lrate_decay = l_rate_decay
        self.lrate_limit = l_rate_limit
        self.sigma = sigma_init
        self.sigma_decay = sigma_decay
        self.sigma_limit = sigma_limit
        self.alpha = 1.0
        self.mean = np.zeros(self.num_params)
        self.grad_subspace = np.zeros((self.subspace_dims, self.num_params))
        self.uut = np.zeros((self.num_params, self.num_params)),
        self.uut_ort = np.zeros((self.num_params, self.num_params))
        self.solutions = None
        self.best_fitness = float("inf")
        self.best_genotype = None

    def ask(self):
        self.grad_subspace -= np.mean(self.grad_subspace, axis=0)
        u, s, vt = np.linalg.svd(self.grad_subspace, full_matrices=False)

        def svd_flip(u, v):
            max_abs_cols = np.argmax(np.abs(u), axis=0)
            signs = np.sign(u[max_abs_cols, np.arange(u.shape[1])])
            u *= signs
            v *= signs[:, np.newaxis]
            return u, v

        u, vt = svd_flip(u, vt)
        u = vt[: int(self.pop_size / 2)]
        self.uut = np.matmul(u.T, u)
        u_ort = vt[int(self.pop_size / 2):]
        self.uut_ort = np.matmul(u_ort.T, u_ort)
        self.uut = self.uut if self.it > self.subspace_dims else np.zeros((self.num_params, self.num_params))
        cov = (self.sigma * (self.alpha / self.num_params) * np.eye(self.num_params)
               + ((1 - self.alpha) / int(self.pop_size / 2)) * self.uut
               )
        chol = np.linalg.cholesky(cov)
        noise = np.random.normal(loc=0.0, scale=1.0, size=(self.num_params, int(self.pop_size / 2)))
        z_plus = np.swapaxes(chol @ noise, 0, 1)
        z_plus /= np.linalg.norm(z_plus, axis=-1)[:, np.newaxis]
        z = np.concatenate([z_plus, -1.0 * z_plus])
        self.solutions = self.mean + z
        return self.solutions

    def tell(self, fitness_list):
        fitness_list = np.array(fitness_list)
        noise = (self.solutions - self.mean) / self.sigma
        noise_1 = noise[: int(self.pop_size / 2)]
        fit_1 = fitness_list[: int(self.pop_size / 2)]
        fit_2 = fitness_list[int(self.pop_size / 2):]
        fit_diff_noise = np.dot(noise_1.T, fit_1 - fit_2)
        theta_grad = 1.0 / 2.0 * fit_diff_noise
        self.alpha = np.linalg.norm(
            np.dot(theta_grad, self.uut_ort)
        ) / np.linalg.norm(np.dot(theta_grad, self.uut)) if self.it > self.subspace_dims else 1.0
        self.grad_subspace[:-1, :] = self.grad_subspace[1:, :]
        self.grad_subspace[-1, :] = np.zeros(self.num_params)
        self.grad_subspace[-1, :] = theta_grad
        theta_grad /= np.linalg.norm(theta_grad) / self.num_params + 1e-8
        self.mean = self.optimizer.optimize(mean=self.mean,
                                            t=self.it,
                                            theta_grad=theta_grad)
        self.sigma = exp_decay(self.sigma, self.sigma_decay, self.sigma_limit)
        self.it += 1

    def result(self):
        return self.best_genotype, self.best_fitness


class NSGAII(PopulationBasedSolver):

    def __init__(self, seed, pop_size, genotype_factory, offspring_size: int, remap, genetic_operators,
                 genotype_filter, **kwargs):
        super().__init__(seed=seed, pop_size=pop_size,
                         genotype_factory=genotype_factory,
                         remap=remap,
                         genetic_operators=genetic_operators,
                         genotype_filter=genotype_filter,
                         comparator="pareto", **kwargs)
        self.offspring_size = offspring_size
        self.fronts = {}
        self.dominates = {}
        self.dominated_by = {}
        self.crowding_distances = {}
        self.parent_selector = Selector.create_selector(name="tournament_crowded",
                                                        crowding_distances=self.crowding_distances, fronts=self.fronts,
                                                        **kwargs)
        self.best_sensing = None
        self.best_locomotion = None

    def _fast_non_dominated_sort(self) -> None:
        self.fronts.clear()
        self.dominates.clear()
        self.dominated_by.clear()
        for p in self.pop:
            self.dominated_by[p.id] = 0
            for q in self.pop:
                if p.id == q.id:
                    continue
                elif p > q:
                    if p.id not in self.dominates:
                        self.dominates[p.id] = [q]
                    else:
                        self.dominates[p.id].append(q)
                elif p < q:
                    self.dominated_by[p.id] += 1
            if self.dominated_by[p.id] == 0:
                if 0 not in self.fronts:
                    self.fronts[0] = [p]
                else:
                    self.fronts[0].append(p)
        if not self.fronts:
            self.fronts[0] = [ind for ind in self.pop]
            return
        i = 0
        while len(self.fronts[i]):
            self.fronts[i + 1] = []
            for p in self.fronts[i]:
                for q in self.dominates.get(p.id, []):
                    self.dominated_by[q.id] -= 1
                    if self.dominated_by[q.id] == 0:
                        self.fronts[i + 1].append(q)
            i += 1
        self.fronts.pop(i)
        self.crowding_distances.clear()
        for front in self.fronts.values():
            self._crowding_distance_assignment(individuals=front)

    def _crowding_distance_assignment(self, individuals: list) -> None:
        for individual in individuals:
            self.crowding_distances[individual.id] = 0.0
        for rank, goal in self.pop.objectives_dict.items():
            individuals.sort(key=lambda x: x.fitness[goal["name"]], reverse=goal["maximize"])
            self.crowding_distances[individuals[0].id] = float("inf")
            self.crowding_distances[individuals[len(individuals) - 1].id] = float("inf")
            for i in range(1, len(individuals) - 1):
                self.crowding_distances[individuals[i].id] += (individuals[i + 1].fitness[goal["name"]] -
                                                               individuals[i - 1].fitness[goal["name"]]) / \
                                                              (abs(goal["best_value"] - goal["worst_value"]))

    def _build_offspring(self) -> list:
        children_genotypes = []
        while len(children_genotypes) < self.offspring_size:
            operator = weighted_random_by_dct(dct=self.genetic_operators)
            parents = [parent.genotype for parent in self.parent_selector.select(population=self.pop,
                                                                                 n=operator.get_arity())]
            children_genotypes.append(operator.apply(tuple(parents)))
        return children_genotypes

    def _trim_population(self) -> None:
        self._fast_non_dominated_sort()
        i = 0
        n = 0
        while n + len(self.fronts[i]) <= self.pop_size:
            n += len(self.fronts[i])
            i += 1
        self.fronts[i].sort(key=lambda x: self.crowding_distances[x.id])
        for j in range(len(self.fronts[i]) - self.pop_size + n):
            self.pop.remove_individual(ind=self.fronts[i][j])
        i += 1
        while i in self.fronts:
            for ind in self.fronts[i]:
                self.pop.remove_individual(ind=ind)
            i += 1

    def ask(self):
        if self.pop.gen != 0:
            for child_genotype in self._build_offspring():
                self.pop.add_individual(genotype=child_genotype)
        else:
            self._fast_non_dominated_sort()
        return [ind.genotype for ind in self.pop]

    def tell(self, fitness_list):
        for ind, f in zip([ind for ind in self.pop if not ind.evaluated], fitness_list):
            ind.fitness = f
            ind.evaluated = True
        if self.pop.gen != 0:
            self._trim_population()
        self.pop.gen += 1

    def result(self):
        return [best.genotype for best in self.fronts[0]], [best.fitness["fitness"] for best in self.fronts[0]]
