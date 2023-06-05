from typing import List
from matplotlib import pyplot as plt
import torch
from agents import Agent


from Costs import CostFunction, PorportionalCost
from instruments.Claims import Claim
from instruments.Derivatives import EuropeanCall, BSCall, BSPut
from instruments.Instruments import Instrument
from instruments.Primaries import GeometricBrownianStock, HestonStock
import RiskMeasures
from ExperimentRunnerSimple import ExperimentRunnerSimple

T = 31
total_rate = 0.0
step_interest_rate = (total_rate + 1) ** (1 / T) - 1
drift = step_interest_rate
volatility = 0.2
S0 = 1
stock = GeometricBrownianStock(S0, drift, volatility)
stock_params = [S0, T, drift, volatility] #strike, expiry, rate, volatility


contingent_claim: Claim = BSCall(stock, S0, T, drift, volatility)
hedging_instruments: List[Instrument] = [stock]
epochs = 100
paths = int(1e5)
verbose = True
criterion: torch.nn.Module = RiskMeasures.TailValue(.05)
cost_function: CostFunction = PorportionalCost(0.0)



runner = ExperimentRunnerSimple("delta", pref_gpu=True)
h_dim = 30
res = runner.run(contingent_claim, hedging_instruments, criterion, T, step_interest_rate, epochs, paths, verbose, cost_function, h_dim, extra_params=stock_params)
print(res)
#runner.plot_training_loss()
#plt.show()

runner.plot_val_dist()

for i in range(2):
    runner.plot_path(i)

#ani = runner.training_pnl_animation()
plt.show()
