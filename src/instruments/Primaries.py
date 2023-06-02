from abc import abstractmethod

import torch
from instruments.Instruments import Instrument


class Primary(Instrument):

    @abstractmethod
    def simulate(self, P, T):
        # returns P x T tensor of primary paths
        pass

    def value(self, primary_path):
        return primary_path

    def primary(self) -> Instrument:
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
        # the first one should be S0
        return torch.cat([
            self.S0 * torch.ones(P, 1),
            self.S0 * torch.exp(torch.cumsum((self.mu - 0.5 * self.sigma ** 2) * torch.ones(P, T - 1) + self.sigma * torch.randn(P, T - 1), dim=1))
        ], dim=1)


class HestonStock(Primary):
    #TODO: implement Heston stock
    pass
