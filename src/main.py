from typing import List
from matplotlib import pyplot as plt
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
from ExperimentRunner import ExperimentRunner

T = 50
total_rate = 0.0
step_interest_rate = (total_rate + 1) ** (1 / T) - 1
drift = step_interest_rate
volatility = 0.2
S0 = 1
stock = GeometricBrownianStock(S0, drift, volatility)


contingent_claim: Claim = BSCall(stock, S0, T, drift, volatility)
hedging_instruments: List[Instrument] = [stock]
epochs = 2
paths = int(1e5)
verbose = False
criterion: torch.nn.Module = RiskMeasures.TailValue(.05)
cost_function: CostFunction = PorportionalCost(0.00)



runner = ExperimentRunner("recurrent", pref_gpu=True)
res = runner.run(contingent_claim, hedging_instruments, criterion, T, step_interest_rate, epochs, paths, verbose, cost_function, 100)
print(res)
# runner.plot_training_loss()
# plt.show()

for i in range(5):
    runner.plot_path(i)
    plt.show()

# ani = runner.training_pnl_animation()
# plt.show()
