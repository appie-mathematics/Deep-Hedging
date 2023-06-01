from collections import OrderedDict
from typing import List
import torch
from agents.Agent import Agent


from Costs import CostFunction, PorportionalCost
from agents.RecurrentAgent import RecurrentAgent
from agents.SimpleAgent import SimpleAgent
from instruments.Claims import Claim
from instruments.Derivatives import EuropeanCall
from instruments.Instruments import Instrument
from instruments.Primaries import GeometricBrownianStock
import RiskMeasures

T = 10
total_rate = 0.0
step_interest_rate = (total_rate + 1) ** (1 / T) - 1
print(step_interest_rate)
drift = step_interest_rate
volatility = 0.2
S0 = 1
stock = GeometricBrownianStock(S0, drift, volatility)
contingent_claim: Claim = EuropeanCall(stock, S0)
hedging_instruments: List[Instrument] = [stock]
N = len(hedging_instruments)

epochs = 100
paths = int(1e5)
verbose = True



criterion: torch.nn.Module = RiskMeasures.CVaR(19)
cost_function: CostFunction = PorportionalCost(0.00)

pref_gpu = False

device: torch.device = torch.device('cpu')
if pref_gpu:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA device found.")

    # mac device
    try:
        device = torch.device("mps")
        print("MPS device found.")
    except:
        pass


agent: Agent = SimpleAgent(criterion, device, cost_function, hedging_instruments, step_interest_rate, h_dim=15)


agent.fit(contingent_claim, epochs, paths, verbose, T)
