from abc import ABC, abstractmethod

import torch
# Payoff functions of derivatives

# claim -> instrument -> [derivatives, primary]

# claim: primary


class Claim(ABC):

    @abstractmethod
    def payoff(self, primary_path):
        # returns payoff of the claim at final timestep
        pass

    def primary(self):
        if self._primary is None:
            raise Exception("Primary not attached")
        return self._primary

    def attach_primary(self, primary):
        self._primary = primary
        return self

class EuropeanCall(Claim):

    def __init__(self, strike):
        self.strike = strike

    def payoff(self, primary_path):
        # primary path is P x T
        return torch.clamp(primary_path[:,-1] - self.strike, min=0)
