from collections import OrderedDict
from typing import List
import torch
from Agent import Agent, SimpleAgent

from Costs import CostFunction, PorportionalCost
from RiskMeasures import Expectation
from instruments.Claims import Claim, EuropeanCall
from instruments.Instrument import Instrument
from instruments.Primaries import GeometricBrownianStock


contingent_claim: Claim = EuropeanCall(100)
stock = GeometricBrownianStock(100, 0.05, 0.2)
contingent_claim.attach_primary(stock)
hedging_instruments: List[Instrument] = [stock]
N = len(hedging_instruments)

epochs = 100
paths = 10000
verbose = True
T = 10

h_dim = 15
simple_model: torch.nn.Module = torch.nn.Sequential(
    OrderedDict([
        ('fc1', torch.nn.Linear(N, h_dim)),
        ('relu1', torch.nn.ReLU()),
        ('fc2', torch.nn.Linear(h_dim, h_dim)),
        ('relu2', torch.nn.ReLU()),
        ('fc3', torch.nn.Linear(h_dim, N))
    ])
)
optimizer: torch.optim.Optimizer = torch.optim.Adam(simple_model.parameters(), lr=0.005)
criterion: torch.nn.Module = Expectation()
device: torch.device = torch.device('cpu')
cost_function: CostFunction = PorportionalCost(0.01)
interest_rate = 0.05


agent: Agent = SimpleAgent(simple_model, optimizer, criterion, device, cost_function, interest_rate)


agent.fit(contingent_claim, hedging_instruments, epochs, paths, verbose, T)
