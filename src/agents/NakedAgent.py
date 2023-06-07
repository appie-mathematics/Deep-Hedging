from collections import OrderedDict
from typing import List
import torch
import numpy as np
from Costs import CostFunction
from agents.Agent import Agent
from instruments.Instruments import Instrument
from torch.distributions import Normal

class NakedAgent(Agent):

    def __init__(self,
                 criterion: torch.nn.Module,
                 cost_function: CostFunction,
                 hedging_instruments: List[Instrument],
                 stock_params,
                 interest_rate = 0.0,
                 lr=0.005,
                 pref_gpu=True,
                 h_dim=15,
                 ):
        super().__init__(criterion, cost_function, hedging_instruments, interest_rate, lr, pref_gpu)

    def forward(self, state: tuple) -> torch.Tensor:
        # do nothing
        return torch.zeros(self.N, device=self.device)
