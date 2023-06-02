from collections import OrderedDict
from typing import List
import torch
from Costs import CostFunction
from agents.Agent import Agent
from instruments.Instruments import Instrument


class SimpleAgent(Agent):

    def __init__(self,
                 criterion: torch.nn.Module,
                 cost_function: CostFunction,
                 hedging_instruments: List[Instrument],
                 interest_rate = 0.05,
                 lr=0.005,
                 pref_gpu=True,
                 h_dim=15,):

        self.N = len(hedging_instruments)
        network_input_dim = self.input_dim()

        super().__init__(criterion, cost_function, hedging_instruments, interest_rate, lr, pref_gpu)
        self.network = torch.nn.Sequential(
        OrderedDict([
            ('fc1', torch.nn.Linear(network_input_dim, h_dim)),
            ('relu1', torch.nn.ReLU()),
            ('fc2', torch.nn.Linear(h_dim, h_dim)),
            ('relu2', torch.nn.ReLU()),
            ('fc3', torch.nn.Linear(h_dim, self.N))
        ])
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr)


    def input_dim(self) -> int:
        return self.N + 1

    def feature_transform(self, state: tuple) -> torch.Tensor:
        """
        :param state: tuple
        :return: torch.Tensor
        """
        # only know the current price
        paths, cash_account, positions = state
        P, t, N = paths.shape

        last_prices = paths[:, -1, :] # (P, N)
        # log prices
        log_prices = torch.log(last_prices) # (P, N)

        times = torch.ones(P, 1, device=self.device) * t # (P, 1)
        # features is log_prices and t
        features = torch.cat([log_prices, times], dim=1) # (P, N+1)

        return features.to(self.device)

    def forward(self, state: tuple) -> torch.Tensor:
        features = self.feature_transform(state) # D x input_dim
        return self.network(features)
