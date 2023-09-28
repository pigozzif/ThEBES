import argparse
import multiprocessing
import os
import time
import logging

from evo.listeners.listener import FileListener
from task.sim import SimulationManager
from task.nevergrad import *
from utilities import create_task, create_policy, create_solver, NEVERGRAD

# cmaes: 0.2 (bipedal), 0.2 (lunar)
# openes: n/a (bipedal), 0.2 (lunar)


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
    best_fitness = float("inf")
    start_time = time.time()
    evaluated = 0
    j = 0
    sim = SimulationManager(config=config, pop_size=solver.pop_size, evaluation=evaluate)
    while evaluated < config.evals:
        solutions = solver.ask()
        results = sim.eval_solutions(solutions=solutions)
        fitness_list = [value for _, value in sorted(results, key=lambda x: x[0])]
        solver.tell(fitness_list)
        result_idx, result_f = sorted(results, key=lambda x: x[1])[0]
        result_g = solutions[result_idx]
        if (j + 1) % config.test_interval == 0:
            logging.warning("fitness at iteration {}: {}".format(j + 1, -result_f))
        if result_f <= best_fitness:
            best_result = result_g
            best_fitness = result_f
        listener.listen(**{"iteration": j, "elapsed.sec": time.time() - start_time,
                           "evaluations": evaluated, "best.fitness": -best_fitness, "avg.test": np.nan,
                           "std.test": np.nan, "best.solution": "/".join([str(x) for x in best_result])})
        evaluated += len(solutions)
        j += 1
    test_scores = - np.array(sim.test_solution(solution=best_result))
    score_avg = np.mean(test_scores)
    score_std = np.std(test_scores)
    listener.listen(**{"iteration": j, "elapsed.sec": time.time() - start_time,
                       "evaluations": evaluated, "best.fitness": -best_fitness, "avg.test": score_avg,
                       "std.test": score_std, "best.solution": "/".join([str(x) for x in best_result])})
    return best_result, best_fitness


def evaluate(config, solution, seed, render=False):
    if config.task in NEVERGRAD:
        return eval(config.task + "({})".format(list(solution)))
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
    file_name = os.path.join("output", ".".join([args.solver, str(args.s), args.task, "txt"]))
    listener = FileListener(file_name=file_name, header=["iteration", "elapsed.sec", "evaluations", "best.fitness",
                                                         "avg.test", "std.test", "best.solution"])
    solver = create_solver(config=args)
    best = parallel_solve(solver=solver, config=args, listener=listener)
