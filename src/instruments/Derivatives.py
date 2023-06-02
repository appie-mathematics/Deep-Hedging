

import torch
from torch.distributions import Normal
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

    def value(self, primary_path, expiry, drift, volatility) -> torch.Tensor:
        expiries = expiry - torch.arange(primary_path.shape[1]) * (expiry / primary_path.shape[1])

        d1 = (torch.log(primary_path/self.strike) + (drift + 0.5 * volatility**2) * expiries) / (volatility * torch.sqrt(expiries))

        d2 = d1 - volatility * torch.sqrt(expiries)

        return primary_path * Normal(0, 1).cdf(d1) - self.strike * torch.exp(-drift * expiries) * Normal(0, 1).cdf(d2)

    def delta(self, primary_path, expiry, drift, volatility) -> torch.Tensor:
        expiries = expiry - torch.arange(primary_path.shape[1]) * (expiry / primary_path.shape[1])

        d1 = (torch.log(primary_path / self.strike) + (drift + 0.5 * volatility ** 2) * expiries) / (volatility * torch.sqrt(expiries))

        return Normal(0, 1).cdf(d1)
