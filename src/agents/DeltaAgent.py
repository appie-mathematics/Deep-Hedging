from collections import OrderedDict
from typing import List
import torch
import numpy as np
from Costs import CostFunction
from agents.Agent import Agent
from instruments.Instruments import Instrument
from torch.distributions import Normal

class DeltaAgent(Agent):

    def __init__(self,
                 criterion: torch.nn.Module,
                 cost_function: CostFunction,
                 hedging_instruments: List[Instrument],
                 stock_params,
                 interest_rate = 0.05,
                 lr=0.005,
                 pref_gpu=True,
                 h_dim=15,
                 ):
        super().__init__(criterion, cost_function, hedging_instruments, interest_rate, lr, pref_gpu)
        self.strike, self.expiry, self.drift, self.volatility = stock_params
        
        #initialize deltas to zero
        self.current_delta = torch.zeros(1, len(hedging_instruments), device=self.device)

    def get_delta(self, last_prices, days_to_expiry):
        d1 = (torch.log(last_prices / self.strike) + (self.drift + 0.5 * self.volatility ** 2) * days_to_expiry) / (self.volatility * np.sqrt(days_to_expiry))

        return Normal(0, 1).cdf(d1)
    
    def feature_transform(self, state: tuple) -> torch.Tensor:
        """
        :param state: tuple
        :return: torch.Tensor
        """
        # only know the current price
        paths, cash_account, positions = state
        P, t, N = paths.shape

        last_prices = paths[:, -1, :]

        #get delta from last prices
        days_to_expiry = self.expiry - t
        delta = self.get_delta(last_prices, days_to_expiry)
        delta_diff = delta - self.current_delta
        self.current_delta = delta

        return delta_diff.to(self.device)
    
    def forward(self, state: tuple) -> torch.Tensor:
        action = self.feature_transform(state) # D x input_dim
        return action



        
