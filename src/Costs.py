from abc import ABC, abstractmethod
import torch


class CostFunction(ABC):

    @abstractmethod
    def cost(self, action, state = None) -> torch.Tensor:
        pass

    def __call__(self, action, state = None):
        return self.cost(action, state)




class PorportionalCost(CostFunction):

    def __init__(self, cost_rate):
        self.cost_rate = cost_rate

    def cost(self, action, state):
        # state = hedge_paths[:,:t], cash_account[:,:t], positions[:,:t], portfolio_value[:,:t]
        purchase_price = state[0][:, -1, :] * action
        return self.cost_rate * purchase_price.abs().sum(dim=-1)
