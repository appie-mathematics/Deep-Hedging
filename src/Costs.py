from abc import ABC, abstractmethod
from typing import Any
import torch


class CostFunction(ABC):

    @abstractmethod
    def cost(self, action, state = None) -> torch.Tensor:
        pass

    def __call__(self, action, state = None):
        return self.cost(action, state)


    def __add__(self, other):
        return SumCost(self, other)

class SumCost(CostFunction):

    def __init__(self, *costs):
        self.costs = costs

    def cost(self, action, state):
        return sum(c.cost(action, state) for c in self.costs)


class PorportionalCost(CostFunction):

    def __init__(self, cost_rate):
        self.cost_rate = cost_rate

    def cost(self, action, state):
        # state = hedge_paths[:,:t], cash_account[:,:t], positions[:,:t], portfolio_value[:,:t]
        purchase_price = state[0][:, -1, :] * action
        return self.cost_rate * purchase_price.abs().sum(dim=-1)



class FixedCost(CostFunction):

    def __init__(self, fixed_cost):
        self.fixed_cost = fixed_cost

    def cost(self, action, state):
        # cost is fixed
        # count the number of non-zero actions
        return self.fixed_cost * (action != 0).sum(dim=-1)
