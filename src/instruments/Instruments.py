from abc import abstractmethod

import torch
from instruments.Claims import Claim


class Instrument(Claim):

    @abstractmethod
    def value(self, primary_path) -> torch.Tensor:
        # returns value of the instrument at each timestep
        pass

    def payoff(self, primary_path):
        # returns payoff of the instrument at final timestep
        return self.value(primary_path)[:,-1]
