

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

    def delta(self, primary_path) -> torch.Tensor:
        pass


class EuropeanOption(Derivative):

        def __init__(self, primary: Primary, strike: float, expiry: int):
            super().__init__(primary)
            self.strike = strike
            self.expiry = expiry


class EuropeanCall(EuropeanOption):

    def payoff(self, primary_path):
        # primary path is P x T
        return torch.clamp(primary_path[:,-1] - self.strike, min=0)

class EuropeanPut(EuropeanOption):

        def payoff(self, primary_path):
            # primary path is P x T
            return torch.clamp(self.strike - primary_path[:,-1], min=0)

class BSCall(EuropeanCall, Instrument):

    def __init__(self, primary: Primary, strike: float, expiry: float, drift: float, volatility: float):
        super().__init__(primary, strike, expiry)
        self.drift = drift
        self.volatility = volatility

    def value(self, primary_path) -> torch.Tensor:
        expiries = self.expiry - (torch.arange(primary_path.shape[1]) + 1) * (self.expiry / primary_path.shape[1])
        expiries[-1] = 0.01

        d1 = (torch.log(primary_path/self.strike) + (self.drift + 0.5 * self.volatility**2) * expiries) / (self.volatility * torch.sqrt(expiries))
        d2 = d1 - self.volatility * torch.sqrt(expiries)
        value = primary_path * Normal(0, 1).cdf(d1) - self.strike * torch.exp(-self.drift * expiries) * Normal(0, 1).cdf(d2)
        value[:,-1] = torch.clamp(primary_path[:,-1] - self.strike, min=0)

        return value

    def delta(self, primary_path) -> torch.Tensor:
        expiries = self.expiry - (torch.arange(primary_path.shape[1]) + 1) * (self.expiry / primary_path.shape[1])

        d1 = (torch.log(primary_path / self.strike) + (self.drift + 0.5 * self.volatility ** 2) * expiries) / (self.volatility * torch.sqrt(expiries))

        return Normal(0, 1).cdf(d1)


class BSPut(EuropeanPut, Instrument):

        def __init__(self, primary: Primary, strike: float, expiry: float, drift: float, volatility: float):
            super().__init__(primary, strike, expiry)
            self.drift = drift
            self.volatility = volatility

        def value(self, primary_path) -> torch.Tensor:
            expiries = self.expiry - (torch.arange(primary_path.shape[1]) + 1) * (self.expiry / primary_path.shape[1])

            d1 = (torch.log(primary_path / self.strike) + (self.drift + 0.5 * self.volatility ** 2) * expiries) / (
                        self.volatility * torch.sqrt(expiries))
            d2 = d1 - self.volatility * torch.sqrt(expiries)
            value = self.strike * torch.exp(-self.drift * expiries) * Normal(0, 1).cdf(-d2) - primary_path * Normal(0, 1).cdf(-d1)
            value[:,-1] = torch.clamp(self.strike - primary_path[:,-1], min=0)

            return value

        def delta(self, primary_path) -> torch.Tensor:
            expiries = self.expiry - (torch.arange(primary_path.shape[1]) + 1) * (self.expiry / primary_path.shape[1])

            d1 = (torch.log(primary_path / self.strike) + (self.drift + 0.5 * self.volatility ** 2) * expiries) / (
                        self.volatility * torch.sqrt(expiries))

            return Normal(0, 1).cdf(d1) - 1
