import abc
from typing import Tuple, Optional, Union, List

import gym as gym
import numpy as np
from gym.core import ActType, ObsType, RenderFrame

from .utils import seeding


class EnvWrapper(gym.Env):

    def __init__(self, env_name, **kwargs):
        self.env = gym.make(env_name, **kwargs)
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def set_seed(self, seed):
        self.env.np_random = seeding(seed)[0]

    def get_max_steps(self):
        return self.env.spec.max_episode_steps

    def get_max_fitness(self):
        return self.env.reward_range[1] * self.get_max_steps()

    def get_min_fitness(self):
        return self.env.reward_range[0] * self.get_max_steps()

    @abc.abstractmethod
    def reset(self, **kwargs):
        pass

    def render(self, mode="human") -> Optional[Union[RenderFrame, List[RenderFrame]]]:
        return self.env.render()

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        return self.env.step(action=action)

    def close(self):
        self.env.close()


class CartPoleHard(EnvWrapper):
    CART_X_LIMIT = 2.4

    def __init__(self):
        super().__init__("CartPole-v1")

    def reset(self, **kwargs):
        # Set custom initial conditions here
        self.env.reset()
        self.env.state = self.get_init_state()
        return self.env.state  # Return the initial state

    def get_init_state(self):
        return np.multiply(self.env.np_random.uniform(low=-1, high=1, size=(4,)),
                           np.array([self.CART_X_LIMIT, 10.0, np.pi / 2.0, 10.0])) + np.array(
            [0, 0, np.pi, 0])


class MountainCar(EnvWrapper):

    def __init__(self):
        super().__init__("MountainCarContinuous-v0")

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class BipedalWalker(EnvWrapper):

    def __init__(self):
        super().__init__("BipedalWalker-v3")

    def reset(self, **kwargs):
        return self.env.reset()


class LunarLander(EnvWrapper):

    def __init__(self):
        super().__init__("LunarLander-v2", continuous=True, gravity=-9.81, enable_wind=False)

    def reset(self, **kwargs):
        return self.env.reset()


class CarRacing(EnvWrapper):

    def __init__(self):
        super().__init__("CarRacing-v2", domain_randomize=False)

    def reset(self, **kwargs):
        return self.env.reset()
