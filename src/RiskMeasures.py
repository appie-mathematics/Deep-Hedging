

from abc import ABC, abstractmethod

import torch


class RiskMeasure(ABC, torch.nn.Module):
    pass



class WorstCase(RiskMeasure):

    def forward(self, portfolio_value: torch.Tensor):
        # portifolio_value: P x 1 (final portfolio value for every path)
        # return: 1 x 1
        return portfolio_value.min()


class Expectation(RiskMeasure):

    def forward(self, portfolio_value: torch.Tensor):
        # portifolio_value: P x 1 (final portfolio value for every path)
        # return: 1 x 1
        return portfolio_value.mean()


class Entropy(RiskMeasure):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, portfolio_value: torch.Tensor):
        # portifolio_value: P x 1 (final portfolio value for every path)
        # return: 1 x 1
        return - 1/self.lambd * torch.log(torch.exp(-self.lambd*portfolio_value).mean())

class CVaR(RiskMeasure):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, portfolio_value: torch.Tensor):
        # portifolio_value: P x 1 (final portfolio value for every path)
        # return: 1 x 1
        # return expected shortfall
        return 1 / (1 - self.alpha) * torch.relu(portfolio_value).mean()
