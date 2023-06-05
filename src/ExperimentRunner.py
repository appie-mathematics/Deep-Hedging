

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
            cost_function: CostFunction = PorportionalCost(0.00),
            h_dim = 15
            ) -> None:

        self.agent: Agent = agents[self.agent_type](criterion, cost_function, hedging_instruments, step_interest_rate, h_dim=h_dim, pref_gpu=self.pref_gpu)
        self.agent.fit(contingent_claim, epochs, paths, verbose, T, logging=True)
        self.training_logs = self.agent.training_logs
        loss = self.agent.validate(contingent_claim, paths, T, logging=True)
        self.validation_logs = self.agent.validation_logs
        self.portfolio_logs = self.agent.portfolio_logs
        return loss


    def plot_val_dist(self):
        # self.validation_logs["validation_profit"] = profit.detach().cpu()
        # self.validation_logs["validation_claim_payoff"] = claim_payoff.detach().cpu()
        # self.validation_logs["validation_loss"] = loss.detach().cpu()
        val_profit = self.validation_logs["validation_profit"]
        val_payoff = self.validation_logs["validation_claim_payoff"]
        val_loss = self.validation_logs["validation_loss"]
        plot = sns.histplot((val_profit).numpy(), stat='density', kde=True, color='blue', label='P&L', binwidth=0.1)
        plot.set_title(f"Validation P&L, Loss: {val_loss:.2f}")
        plot.set_xlim(-5, 5)
        plot.set_ylim(0, 2)
        plot.grid()
        plot.set_xlabel("Profit")
        plot.set_ylabel("Density")
        return plot

    def training_pnl_animation(self):
        training_pl = self.training_logs["training_PL"]
        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()
            sns.histplot(training_pl[i].numpy(), ax=ax, stat='density', kde=True, color='blue', label='P&L', binwidth=0.1)
            ax.set_title(f"Epoch {i+1}")
            ax.set_xlim(-5, 5)
            ax.set_ylim(0, 2)
            ax.grid()
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
        claim_payoff = portfolio_logs["claim_payoff"][i]
        claim_delta = portfolio_logs["claim_delta"][i]

        fig, ax = plt.subplots(2, 2)

        for a in ax.flatten():
            a.grid(True)  # Adding grid to each subplot

        for i in range(hedge_paths.shape[1]):
            sns.lineplot(x=range(len(hedge_paths[:, i])), y=hedge_paths[:, i], ax=ax[0, 0])
        ax[0,0].set_title("Hedge Prices")
        ax[0,0].set_xlabel("Time")
        ax[0,0].set_ylabel("Value")


        for i in range(positions.shape[1]):
            sns.lineplot(x=range(len(positions[:, i])), y=positions[:, i], ax=ax[0, 1])
        ax[0, 1].set_title("Positions")
        ax[0, 1].set_xlabel("Time")
        ax[0, 1].set_ylabel("Value")
        # add line for y = 0
        ax[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        # make diverging from 0
        max_value = max(abs(positions.min()), abs(positions.max())) * 1.1
        ax[0, 1].set_ylim(-max_value, max_value)

        if claim_delta is not None:
            # add the delta of the claim to the graph
            ax[0, 1] = sns.lineplot(x=range(len(claim_delta)), y=claim_delta, ax=ax[0, 1], color='black', label='Claim Delta')




        pnl = portfolio_value + cash_account

        sns.lineplot(x=range(len(pnl)), y=pnl, ax=ax[1, 0])
        ax[1, 0].set_title("P&L")
        ax[1, 0].set_xlabel("Time")
        ax[1, 0].set_ylabel("Value")
        # add line for y = 0
        ax[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        # make diverging from 0
        max_value = max(abs(pnl.min()), abs(pnl.max())) * 1.1
        ax[1, 0].set_ylim(-max_value, max_value)

        final_cash = cash_account[-1]
        final_portfolio_value = portfolio_value[-1]

        # Setting up the diverging bars
        categories = ['Cash', 'PV', 'CC', 'P&L']
        values = [final_cash, final_portfolio_value, -claim_payoff, final_cash + final_portfolio_value -claim_payoff]
        colors = ['red', 'blue', 'green', 'orange']

        ax[1,1].bar(categories, values, color=colors)
        ax[1,1].set_title("Final Portfolio Breakdown")
        ax[1,1].set_xlabel("Category")
        ax[1,1].set_ylabel("Value")

        # Adding a horizontal line at y=0
        ax[1,1].hlines(0, -1, len(categories), colors='black', linestyles='dashed')

        # Set y limits to center the plot around 0
        max_value = max(abs(min(values)), abs(max(values))) * 1.1
        ax[1,1].set_ylim(-max_value, max_value)

        # Adding value annotations to the bars
        for i, v in enumerate(values):
            ax[1,1].text(i, v, f" {v:.2f}", color='black', ha='center', fontweight='bold')

        return fig
