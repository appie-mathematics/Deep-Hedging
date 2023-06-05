from abc import abstractmethod

import torch
import numpy as np
from instruments.Instruments import Instrument


class Primary(Instrument):

    @abstractmethod
    def simulate(self, P, T) -> torch.Tensor:
        # returns P x T tensor of primary paths
        pass

    def value(self, primary_path):
        return primary_path

    def primary(self) -> Instrument:
        return self

    def delattr(self, primary_path):
        return torch.ones_like(primary_path)

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
    # only returns stock price, not variance
    # as this would need some editing

    def __init__(self, S0, V0, mu, kappa, theta, xi, rho):
        self.S0 = S0
        self.V0 = V0
        self.mu = mu
        self.kappa = kappa
        self.theta = theta
        self.xi = xi
        self.rho = rho

    def name(self):
        return f"Heston Stock with S0={self.S0}, V0={self.V0}, mu={self.mu}, kappa={self.kappa}, theta={self.theta}, xi={self.xi}, rho={self.rho}"

    def simulate(self, P, T):
        # returns P x T tensor of primary paths
        # the first one should be S0
        # the second one should be V0
        S = torch.zeros(P, T)
        V = torch.zeros(P, T)

        #initial values (P x 1)
        S[:, 0] = self.S0
        V[:, 0] = self.V0

        step_size = 1/T

        #generate increments (normal distributed with mean 0, variance sqrt(step_size))
        stock_increments = torch.randn(P, T - 1) * np.sqrt(step_size)
        variance_increments = torch.randn(P, T - 1) * np.sqrt(step_size)

        for t in range(1, T):
            S[:, t] = S[:, t-1] + self.mu * S[:, t-1] * step_size + torch.sqrt(torch.clamp(V[:,t-1], min=0)) * S[:, t-1] * stock_increments[:, t-1]
            V[:, t] = V[:, t-1] + self.kappa * (self.theta - torch.clamp(V[:,t-1], min=0)) * step_size + self.xi * torch.sqrt(torch.clamp(V[:,t-1], min=0)) * (self.rho * variance_increments[:, t-1] + np.sqrt(1 - self.rho**2) * stock_increments[:, t-1])

        return S
