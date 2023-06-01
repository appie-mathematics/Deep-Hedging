

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
    def __init__(self, lambd: float):
        super().__init__()
        self.lambd = lambd

    def forward(self, portfolio_value: torch.Tensor):
        # portifolio_value: P x 1 (final portfolio value for every path)
        # return: 1 x 1
        # return expected shortfall

        def target_function(omega: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return omega + self.lambd * torch.clamp(-omega - x, min=0).mean()

        # try different omega's and return the one that gives the max of target_function (expected shortfall)
        omega = torch.linspace(-1, 1, portfolio_value.shape[0], device=portfolio_value.device)
        target = target_function(omega, portfolio_value)
        omega = omega[target.argmax()]

        return -(omega + self.lambd * torch.clamp(-omega - portfolio_value, min=0).mean())
