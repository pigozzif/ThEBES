import argparse
import multiprocessing
import os
import time
import logging

import numpy as np

from evo.listeners.listener import FileListener
from task.sim import SimulationManager
from utilities import create_task, create_policy, create_solver


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--s", type=int, default=0, help="seed")
    parser.add_argument("--np", type=int, default=multiprocessing.cpu_count(), help="parallel optimization processes")
    parser.add_argument("--solver", type=str, default="afpo", help="solver")
    parser.add_argument("--evals", type=int, default=120000, help="fitness evaluations")
    parser.add_argument("--hidden-size", type=int, default=64, help="policy hidden size")
    parser.add_argument("--num-tests", type=int, default=100, help="number of test rollouts")
    parser.add_argument("--test-interval", type=int, default=10, help="test interval")
    parser.add_argument("--log-interval", type=int, default=20, help="logging interval")
    parser.add_argument("--task", type=str, default="car", help="task")
    parser.add_argument("--sigma", type=float, default=0.1, help="step size")
    parser.add_argument("--l-rate", type=float, default=0.01, help="learning rate")
    return parser.parse_args()


def parallel_solve(solver, config, listener):
    best_result = None
    best_fitness = float("-inf")
    start_time = time.time()
    evaluated = 0
    j = 0
    sim = SimulationManager(config=config, pop_size=solver.pop_size, evaluation=evaluate)
    while evaluated < config.evals:
        solutions = solver.ask()
        results = sim.eval_solutions(solutions=solutions)
        fitness_list = [value for _, value in sorted(results, key=lambda x: x[0])]
        solver.tell(fitness_list)
        result = solver.result()  # first element is the best solution, second element is the best fitness
        if (j + 1) % config.test_interval == 0:
            logging.warning("fitness at iteration {}: {}".format(j + 1, result[1]))
        listener.listen(**{"iteration": j, "elapsed.sec": time.time() - start_time,
                           "evaluations": evaluated, "best.fitness": -result[1], "avg.test": np.nan,
                           "std.test": np.nan, "best.solution": np.nan})
        if result[1] <= best_fitness or best_result is None:
            best_result = result[0]
            best_fitness = result[1]
        evaluated += len(solutions)
        j += 1
    test_scores = -sim.test_solution(solution=result[0])
    score_avg = np.mean(test_scores)
    score_std = np.std(test_scores)
    listener.listen(**{"iteration": j, "elapsed.sec": time.time() - start_time,
                       "evaluations": evaluated, "best.fitness": -best_fitness, "avg.test": score_avg,
                       "std.test": score_std, "best.solution": "/".join([str(x) for x in best_result])})
    return best_result, best_fitness


def evaluate(config, solution, seed, render=False):
    env = create_task(config=config)
    env.set_seed(seed)
    policy = create_policy(config=config, env=env)
    obs = env.reset()[0]
    policy.set_params(params=solution)
    fitness = 0.0
    for i in range(env.get_max_steps()):
        action = policy.act(obs=obs)
        obs, reward, done, *_ = env.step(action=action)
        fitness += reward
        if done:
            _ = env.reset()
            break
        elif render:
            env.render()
    env.close()
    return -fitness


if __name__ == "__main__":
    args = parse_args()
    file_name = os.path.join("output", "-".join([args.solver, str(args.s), args.task, str(args.sigma), "txt"]))
    listener = FileListener(file_name=file_name, header=["iteration", "elapsed.sec", "evaluations", "best.fitness",
                                                         "avg.test", "std.test", "best.solution"])
    solver = create_solver(config=args)
    best = parallel_solve(solver=solver, config=args, listener=listener)
