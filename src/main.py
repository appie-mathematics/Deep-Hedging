from typing import List
from matplotlib import pyplot as plt
import torch
from agents.Agent import Agent


from Costs import CostFunction, PorportionalCost, FixedCost
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

T = 10
total_rate = 0.0
step_interest_rate = (total_rate + 1) ** (1 / T) - 1
drift = step_interest_rate
volatility = 0.2
S0 = 1
stock = GeometricBrownianStock(S0, drift, volatility)
contingent_claim: Claim = BSCall(stock, S0, T, drift, volatility)
hedging_instruments: List[Instrument] = [stock]
epochs = 50
paths = int(5e4)
verbose = True
criterion: torch.nn.Module = RiskMeasures.WorstCase()
prop_cost = 0.00
cost_function: CostFunction = PorportionalCost(prop_cost)

runners = []
naked_runner = SimpleRunner("naked", pref_gpu=True)
res = naked_runner.run(contingent_claim, hedging_instruments, criterion, T, step_interest_rate, epochs, paths, verbose, cost_function)
print(res)
runners.append(naked_runner)

stock_params = [S0, T, drift, volatility] #strike, expiry, rate, volatility
delta_runner = SimpleRunner("delta", pref_gpu=True)
res = delta_runner.run(contingent_claim, hedging_instruments, criterion, T, step_interest_rate, epochs, paths, verbose, cost_function, extra_params=stock_params)
print(res)
runners.append(delta_runner)

# runner = ExperimentRunner("recurrent", pref_gpu=True)
# h_dim = 15
# res = runner.run(contingent_claim, hedging_instruments, criterion, T, step_interest_rate, epochs, paths, verbose, cost_function, h_dim)
# print(res)
# filename = f"output/{runner.agent_type}_hd_{h_dim}_e_{epochs}_p_{paths}_s_{seed}_pc_{prop_cost: .1f}"
# runner.plot_runner(animate=False, save=False, file_prefix=filename, n=2)
# runners.append(runner)

plot_dists(runners, save=True, file_prefix="output/naked_delta")
plt.show()
