

from abc import ABC, abstractmethod

import torch


class RiskMeasure(ABC, torch.nn.Module):
    pass







class CVaR(RiskMeasure):
    def __init__(self, alpha):
        super().__init__()
        self.alpha = alpha

    def __call__(self, portfolio_value: torch.Tensor):
        # portifolio_value: P x 1 (final portfolio value for every path)
        # return: 1 x 1
        # return expected shortfall
        # TODO: check if this is correct
        return portfolio_value.mean() + (1 / (1 - self.alpha)) * portfolio_value[portfolio_value < portfolio_value.quantile(self.alpha)].mean()
