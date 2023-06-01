

import torch
from instruments.Claims import Claim
from instruments.Instruments import Instrument
from instruments.Primaries import Primary



class Derivative(Claim):

    def __init__(self, primary: Primary):
        self._primary = primary

    def primary(self) -> Primary:
        return self._primary


class EuropeanOption(Derivative):

        def __init__(self, primary: Primary, strike: float):
            super().__init__(primary)
            self.strike = strike


class EuropeanCall(EuropeanOption):

    def payoff(self, primary_path):
        # primary path is P x T
        return torch.clamp(primary_path[:,-1] - self.strike, min=0)


class BSCall(EuropeanCall, Instrument):
    # Black-Scholes Call TODO

    # def value(self, primary_path) -> torch.Tensor:
    pass
