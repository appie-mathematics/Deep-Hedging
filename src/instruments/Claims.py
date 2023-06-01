from abc import ABC, abstractmethod

import torch
# Payoff functions of derivatives

# claim -> instrument -> [derivatives, primary]

# claim: primary

# Claims has a single payoff depending on the path
class Claim(ABC):

    @abstractmethod
    def payoff(self, primary_path) -> torch.Tensor:
        # returns payoff of the claim at final timestep
        pass

    @abstractmethod
    def primary(self):
         # return primary of the claim
        pass
