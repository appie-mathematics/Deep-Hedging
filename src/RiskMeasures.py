

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

    def forward(self, portfolio_value: torch.Tensor):
        # portifolio_value: P x 1 (final portfolio value for every path)
        # return: 1 x 1
        return -torch.sum(portfolio_value * torch.exp(portfolio_value)) / portfolio_value.shape[0]


class CVaR(RiskMeasure):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def forward(self, portfolio_value: torch.Tensor):
        # portifolio_value: P x 1 (final portfolio value for every path)
        # return: 1 x 1
        # return expected shortfall
        # TODO: check if this is correct
        return portfolio_value.mean() + (1 / (1 - self.alpha)) * portfolio_value[portfolio_value < portfolio_value.quantile(self.alpha)].mean()
