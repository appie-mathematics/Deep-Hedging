

from abc import ABC, abstractmethod

import torch


class RiskMeasure(ABC, torch.nn.Module):
    pass



class WorstCase(RiskMeasure):

    def forward(self, portfolio_value: torch.Tensor):
        # portifolio_value: P x 1 (final portfolio value for every path)
        # return: 1 x 1
        return portfolio_value.min()


class TailValue(RiskMeasure):

    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, portfolio_value: torch.Tensor):
        # portifolio_value: P x 1 (final portfolio value for every path)
        # returns the mean of the worst alpha% of the paths
        # remember that portfolio_value is not sorted and can be negative
        # return: 1 x 1
        k = int(self.alpha * portfolio_value.shape[0])
        return portfolio_value.topk(k, largest=False).values.mean()


class Median(RiskMeasure):

    def forward(self, portfolio_value: torch.Tensor):
        # portifolio_value: P x 1 (final portfolio value for every path)
        # return: 1 x 1
        return portfolio_value.median()




class Expectation(RiskMeasure):

    def forward(self, portfolio_value: torch.Tensor):
        # portifolio_value: P x 1 (final portfolio value for every path)
        # return: 1 x 1
        return portfolio_value.mean()


class Entropy(RiskMeasure):
    def __init__(self, lambd: float):
        super().__init__()
        self.lambd = lambd

    def forward(self, portfolio_value: torch.Tensor):
        # portifolio_value: P x 1 (final portfolio value for every path)
        # return: 1 x 1
        return - 1/self.lambd * torch.log(torch.exp(-self.lambd*portfolio_value).mean())

class CVaR(RiskMeasure):
    def __init__(self, alpha: float):
        super().__init__()
        self.alpha = alpha

    def forward(self, portfolio_value: torch.Tensor):
        # portifolio_value: P x 1 (final portfolio value for every path)
        # return: 1 x 1
        # return expected shortfall
        return 1 / (1 - self.alpha) * torch.relu(portfolio_value).mean()
