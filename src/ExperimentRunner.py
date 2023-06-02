

from typing import List
from matplotlib.animation import FuncAnimation

import torch
from Costs import CostFunction, PorportionalCost
from RiskMeasures import RiskMeasure
from agents.Agent import Agent
from instruments.Claims import Claim
from instruments.Instruments import Instrument
from agents.SimpleAgent import SimpleAgent
from agents.RecurrentAgent import RecurrentAgent
import matplotlib.pyplot as plt
import seaborn as sns


agents = {
    "simple": SimpleAgent,
    "recurrent": RecurrentAgent
}

class ExperimentRunner:

    def __init__(self, agent_type: str, pref_gpu=True) -> None:
        self.pref_gpu = pref_gpu
        self.agent_type = agent_type


    def run(self,
            contingent_claim: Claim,
            hedging_instruments: List[Instrument],
            criterion: torch.nn.Module,
            T = 10,
            step_interest_rate = 0.0,
            epochs = 50,
            paths = int(1e5),
            verbose = True,
            cost_function: CostFunction = PorportionalCost(0.00)
            ) -> None:

        self.agent: Agent = agents[self.agent_type](criterion, cost_function, hedging_instruments, step_interest_rate, h_dim=15, pref_gpu=self.pref_gpu)
        self.agent.fit(contingent_claim, epochs, paths, verbose, T, logging=True)
        self.training_logs = self.agent.training_logs
        loss = self.agent.validate(contingent_claim, paths, T, logging=True)
        self.validation_logs = self.agent.validation_logs
        self.portfolio_logs = self.agent.portfolio_logs
        return loss



    def training_pnl_animation(self):
        training_pl = self.training_logs["training_PL"]
        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()
            sns.histplot(training_pl[i].numpy(), ax=ax, stat='density', kde=True, color='blue', label='P&L')
            ax.set_xlim(-5, 5)
            ax.set_ylim(0, 2)
            ax.grid()
            ax.set_title(f"Epoch {i+1}")
            ax.set_xlabel("Profit")
            ax.set_ylabel("Density")

        return FuncAnimation(fig, animate, frames=len(training_pl), repeat=True)

    def plot_training_loss(self):
        losses = self.training_logs["training_losses"]
        plot = sns.lineplot(x=range(len(losses)), y=losses)
        plot.set_title("Training Loss")
        plot.set_xlabel("Epoch")
        plot.set_ylabel("Loss")
        return plot



    def plot_path(self, i):
        portfolio_logs = self.agent.portfolio_logs
        portfolio_value = portfolio_logs["portfolio_value"][i]
        cash_account = portfolio_logs["cash_account"][i]
        positions = portfolio_logs["positions"][i]
        hedge_paths = portfolio_logs["hedge_paths"][i]
        #total_cost = portfolio_logs["total_cost"][i]

        fig, ax = plt.subplots(2, 2, figsize=(10, 10))

        sns.lineplot(x=range(len(portfolio_value)), y=portfolio_value, ax=ax[0, 0])
        ax[0, 0].set_title("Portfolio Value")
        ax[0, 0].set_xlabel("Time")
        ax[0, 0].set_ylabel("Value")

        sns.lineplot(x=range(len(cash_account)), y=cash_account, ax=ax[0, 1])
        ax[0, 1].set_title("Cash Account")
        ax[0, 1].set_xlabel("Time")
        ax[0, 1].set_ylabel("Value")

        # positions.shape = (T, N) where N is the number of hedging instruments
        for i in range(positions.shape[1]):
            sns.lineplot(x=range(len(positions[:, i])), y=positions[:, i], ax=ax[1, 0])
        ax[1, 0].set_title("Positions")
        ax[1, 0].set_xlabel("Time")
        ax[1, 0].set_ylabel("Value")

        for i in range(hedge_paths.shape[1]):
            sns.lineplot(x=range(len(hedge_paths[:, i])), y=hedge_paths[:, i], ax=ax[1, 1])
        ax[1, 1].set_title("Hedge Paths")
        ax[1, 1].set_xlabel("Time")
        ax[1, 1].set_ylabel("Value")

        #fig.suptitle(f"Total Cost: {total_cost}")
        return fig
