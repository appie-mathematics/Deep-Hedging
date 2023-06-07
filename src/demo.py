from typing import List
from matplotlib import pyplot as plt
import torch
from agents.Agent import Agent


from Costs import CostFunction, PorportionalCost
from agents.RecurrentAgent import RecurrentAgent
from agents.SimpleAgent import SimpleAgent
from instruments.Claims import Claim
from instruments.Derivatives import EuropeanCall, BSCall, BSPut, EuropeanPut
from instruments.Instruments import Instrument
from instruments.Primaries import GeometricBrownianStock, HestonStock
import RiskMeasures
from ExperimentRunner import ExperimentRunner, SimpleRunner, plot_dists

seed = 1337
torch.manual_seed(seed)

T = 10 # Number of time steps
total_rate = 0.0
step_interest_rate = (total_rate + 1) ** (1 / T) - 1

# Define an underlying stock
drift = step_interest_rate
volatility = 0.2
S0 = 1
stock = GeometricBrownianStock(S0, drift, volatility)

# Define the claim, hedging instruments, criterion, and costs for the experiment
contingent_claim: Claim = BSCall(stock, S0, T, drift, volatility)
hedging_instruments: List[Instrument] = [stock]
criterion: torch.nn.Module = RiskMeasures.WorstCase()
cost_function: CostFunction = PorportionalCost(0.0)

# Define the agent
epochs = 50
h_dim = 15
paths = int(1e4)
verbose = False
runner = ExperimentRunner("recurrent", pref_gpu=True)

# Run the experiment
runner.run(contingent_claim, hedging_instruments, criterion, T, step_interest_rate, epochs, paths, verbose, cost_function, h_dim)

# Plot the results
runner.plot_runner(animate=True, save=False, n=3)
