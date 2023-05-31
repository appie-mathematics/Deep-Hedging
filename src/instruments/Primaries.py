from abc import abstractmethod
from instruments.Instrument import Instrument


class Primary(Instrument):

    @abstractmethod
    def simulate(self, P, T):
        # returns P x T tensor of primary paths
        pass

    @abstractmethod
    def name():
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
