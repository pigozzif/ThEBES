import abc
import random
from abc import ABC
from typing import Dict

import numpy as np

from .objectives import ObjectiveDict
from .operators.operator import GeneticOperator
from .optimizers import Adam
from .selection.filters import Filter
from .selection.selector import Selector
from ..representations.factory import GenotypeFactory
from ..representations.population import Population, Individual, Comparator
from ..utils.utilities import weighted_random_by_dct, exp_decay


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


class RandomSearch(StochasticSolver):

    def __init__(self, seed, num_params, sigma, objectives_dict):
        super().__init__(seed, num_params, 1)
        self.comparator = Comparator.create_comparator(name="lexicase", objective_dict=objectives_dict)
        self.genetic_operator = GeneticOperator.create_genetic_operator(name="gaussian_mut", mu=0.0, sigma=sigma)
        self.best_fitness = objectives_dict[0]["worst_value"]
        self.best = Individual(id=0, genotype=np.zeros(num_params), comparator=self.comparator)
        self.move = None

    def ask(self):
        self.move = Individual(id=0,
                               genotype=self.genetic_operator.apply(tuple(self.best.genotype)),
                               comparator=self.comparator)
        return [self.move.genotype]

    def tell(self, fitness_list):
        self.move.fitness = {"fitness": fitness_list[0]}
        if self.comparator.compare(ind1=self.best, ind2=self.move) == -1:
            self.best = self.move

    def result(self):
        return self.best.genotype, self.best.fitness["fitness"]


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
        for child_genotype in self._sample_offspring():
            self.pop.add_individual(genotype=child_genotype)
        return [ind.genotype for ind in self.pop if not ind.evaluated]

    def _update_mode(self) -> None:
        noise = np.array([(x.genotype - self.mode) / self.sigma for x in self.pop])
        fitness = np.array([x.fitness["fitness_score"] for x in self.pop])
        best_idx = np.argmin(fitness)
        self.best_fitness, self.best_genotype = fitness[best_idx], (noise[best_idx] * self.sigma) + self.mode
        theta_grad = (1.0 / (self.pop_size * self.sigma)) * np.dot(noise.T, fitness)
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
