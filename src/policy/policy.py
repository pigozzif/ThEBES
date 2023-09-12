import abc

import numpy as np
import torch
import cv2


class Policy(object):

    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size

    @abc.abstractmethod
    def set_params(self, params):
        pass

    @abc.abstractmethod
    def act(self, obs):
        pass


class MLPPolicy(Policy):

    def __init__(self, input_size, output_size, hidden_size):
        super().__init__(input_size=input_size, output_size=output_size)
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(in_features=self.input_size, out_features=hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=hidden_size, out_features=self.output_size)
            )

    def set_params(self, params):
        state_dict = self.nn.state_dict()
        start = 0
        for key, coeffs in state_dict.items():
            num = coeffs.numel()
            state_dict[key] = torch.tensor(np.array(params[start:start + num]).reshape(state_dict[key].shape))
            start += num
        self.nn.load_state_dict(state_dict)

    def act(self, obs):
        obs = torch.from_numpy(obs).float()
        return self.nn(obs).detach().numpy()


class ConvPolicy(MLPPolicy):

    def __init__(self, input_size, output_size, hidden_size):
        super().__init__(input_size=input_size, output_size=output_size, hidden_size=hidden_size)
        self.nn = torch.nn.Sequential(
            torch.nn.Conv2d(4, 32, kernel_size=(8,), stride=(4,)),  # Conv1
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=(4,), stride=(2,)),  # Conv2
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 64, kernel_size=(3,), stride=(1,)),  # Conv3
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(7 * 7 * 64, 512),  # Fully Connected Layer
            torch.nn.ReLU(),
            torch.nn.Linear(512, output_size)
        )
        self.history = []

    def act(self, obs):
        obs = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        obs = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_LINEAR)
        self.history.append(obs)
        if len(self.history) > 4:
            self.history.pop(0)
        super().act(obs=np.stack([self.history], axis=-1))
