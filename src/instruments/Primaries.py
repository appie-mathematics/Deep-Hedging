from abc import abstractmethod

import torch
from instruments.Instrument import Instrument


class Primary(Instrument):

    @abstractmethod
    def simulate(self, P, T):
        # returns P x T tensor of primary paths
        pass

    @abstractmethod
    def name() -> str:
        pass

    def __str__(self) -> str:
        return self.name()

    def value(self, primary_path):
        return primary_path

    def primary(self):
        return self

class GeometricBrownianStock(Primary):

    def __init__(self, S0, mu, sigma):
        self.S0 = S0
        self.mu = mu
        self.sigma = sigma

    def name(self):
        return f"Geometric Brownian Stock with S0={self.S0}, mu={self.mu}, sigma={self.sigma}"

    def simulate(self, P, T):
        # returns P x T tensor of primary paths
        return self.S0 * torch.exp(torch.cumsum((self.mu - 0.5 * self.sigma ** 2) * torch.ones(P, T) + self.sigma * torch.randn(P, T), dim=-1))
