from policy.policy import ConvPolicy, MLPPolicy
from evo.evolution.algorithms import OpenAIES, ThEBES, CMAES, RandomSearch
from evo.evolution.objectives import ObjectiveDict
from task.envs import CartPoleHard, BipedalWalker, LunarLander, MountainCar


def is_classic(task):
    return "cartpole" in task or "mountain" in task


def create_solver(config):
    objectives_dict = ObjectiveDict()
    objectives_dict.add_objective(name="fitness_score", maximize=False, best_value=0.0, worst_value=5.0)
    pop_size = get_pop_size(task=config.task)
    num_params = get_number_of_params(config=config)
    if config.solver == "openes":
        return OpenAIES(seed=config.s,
                        num_params=num_params,
                        pop_size=pop_size,
                        objectives_dict=objectives_dict,
                        sigma=0.03 if is_classic(task=config.task) else 0.04,
                        sigma_decay=0.999,
                        sigma_limit=0.01 if is_classic(task=config.task) else 0.001,
                        l_rate_init=0.02 if is_classic(task=config.task) else 0.01,
                        l_rate_decay=0.999,
                        l_rate_limit=0.001 if is_classic(task=config.task) else 0.005
                        )
    elif config.solver == "thebes":
        return ThEBES(seed=config.s,
                      num_params=num_params,
                      pop_size=pop_size,
                      objectives_dict=objectives_dict,
                      sigma=0.03 if is_classic(task=config.task) else 0.04,
                      sigma_decay=0.999,
                      sigma_limit=0.01 if is_classic(task=config.task) else 0.001,
                      l_rate_init=0.02 if is_classic(task=config.task) else 0.01,
                      l_rate_decay=0.999,
                      l_rate_limit=0.001 if is_classic(task=config.task) else 0.005
                      )
    elif config.solver == "cmaes":
        return CMAES(seed=config.s,
                     num_params=num_params,
                     pop_size=pop_size,
                     sigma_init=0.03 if is_classic(task=config.task) else 0.04)
    elif config.solver == "rs":
        return RandomSearch(seed=config.s,
                            num_params=num_params,
                            sigma=0.03 if is_classic(task=config.task) else 0.05,
                            objectives_dict=objectives_dict)
    raise ValueError("Invalid solver name: {}".format(config.solver))


def create_task(config):
    task_name = config.task
    if task_name.startswith("cartpole"):
        return CartPoleHard()
    elif task_name == "bipedal":
        return BipedalWalker()
    elif task_name == "lunar":
        return LunarLander()
    elif task_name == "mountain":
        return MountainCar()
    raise ValueError("Invalid task name: {}".format(task_name))


def create_policy(config, env):
    if config.task == "car":
        return ConvPolicy(input_size=env.observation_space.shape[0],
                          output_size=env.action_space.shape[0],
                          hidden_size=config.hidden_size)
    return MLPPolicy(input_size=env.observation_space.shape[0],
                     output_size=env.action_space.shape[0],
                     hidden_size=config.hidden_size)


def get_number_of_inputs(task):
    if task.startswith("cartpole"):
        return 4
    elif task == "bipedal":
        return 24
    elif task == "lunar":
        return 8
    elif task == "car":
        return 96 * 96 * 3
    elif task == "mountain":
        return 2
    raise ValueError("Invalid task name: {}".format(task))


def get_number_of_outputs(task):
    if task.startswith("cartpole"):
        return 1
    elif task == "bipedal":
        return 4
    elif task == "lunar":
        return 2
    elif task == "car":
        return 3
    elif task == "mountain":
        return 1
    raise ValueError("Invalid task name: {}".format(task))


def get_number_of_params(config):
    input_dim = get_number_of_inputs(task=config.task)
    output_dim = get_number_of_outputs(task=config.task)
    if config.task == "car":
        return 1629347
    return input_dim * config.hidden_size + config.hidden_size + config.hidden_size * output_dim + output_dim


def get_pop_size(task):
    if is_classic(task=task):
        return 100
    return 256