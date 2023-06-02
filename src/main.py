from typing import List
import torch
from agents.Agent import Agent


from Costs import CostFunction, PorportionalCost
from agents.RecurrentAgent import RecurrentAgent
from agents.SimpleAgent import SimpleAgent
from instruments.Claims import Claim
from instruments.Derivatives import EuropeanCall, BSCall
from instruments.Instruments import Instrument
from instruments.Primaries import GeometricBrownianStock
import RiskMeasures

T = 10
total_rate = 0.0
step_interest_rate = (total_rate + 1) ** (1 / T) - 1
drift = step_interest_rate
volatility = 0.2
S0 = 1
stock = GeometricBrownianStock(S0, drift, volatility)
contingent_claim: Claim = BSCall(stock, S0, T, drift, volatility)
hedging_instruments: List[Instrument] = [BSCall(stock, S0*1.5, T, drift, volatility)]
N = len(hedging_instruments)

epochs = 50
paths = int(1e5)
verbose = True



criterion: torch.nn.Module = RiskMeasures.TailValue(.05)
cost_function: CostFunction = PorportionalCost(0.00)

pref_gpu = True




agent: Agent = RecurrentAgent(criterion, cost_function, hedging_instruments, step_interest_rate, h_dim=15, pref_gpu=pref_gpu)


agent.fit(contingent_claim, epochs, paths, verbose, T)
