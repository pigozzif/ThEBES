import multiprocessing


class SimulationManager(object):

    def __init__(self, config, pop_size, evaluation):
        self.config = config
        self.pop_size = pop_size
        self.evaluation = evaluation

    def eval_solutions(self, solutions):
        with multiprocessing.Pool(self.config.np) as pool:
            results = pool.map(self._parallel_wrapper, [(self.config, solutions[i], i, False)
                                                        for i in range(self.pop_size)])
        return results

    def _parallel_wrapper(self, arg):
        c, solution, i, test = arg
        fitness = self.evaluation(config=c, solution=solution, seed=i if test else c.s)
        return i, fitness

    def test_solution(self, solution):
        with multiprocessing.Pool(self.config.np) as pool:
            results = pool.map(self._parallel_wrapper, [(self.config, solution, i, True) for i in range(self.pop_size)])
        return [value for _, value in sorted(results, key=lambda x: x[0])]
