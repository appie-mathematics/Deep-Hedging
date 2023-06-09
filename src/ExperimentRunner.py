

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

from agents.DeltaAgent import DeltaAgent
from agents.NakedAgent import NakedAgent


agents = {
    "simple": SimpleAgent,
    "recurrent": RecurrentAgent,
    "delta": DeltaAgent,
    "naked": NakedAgent
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
        loss = self.agent.validate(contingent_claim, int(1e6), T, logging=True)
        self.validation_logs = self.agent.validation_logs
        self.portfolio_logs = self.agent.portfolio_logs
        self.claim = contingent_claim
        self.hedging_instruments = hedging_instruments

        return loss


    def plot_val_dist(self, save=False, file_prefix='plot'):
        return plot_dists([self], save, file_prefix)

    def training_pnl_animation(self):
        training_pl = self.training_logs["training_PL"]
        fig, ax = plt.subplots()

        def animate(i):
            ax.clear()
            ax.set_title(f"Training P&L, N: {len(training_pl[i])}, Epoch: {i+1}")
            ax.set_xlim(-4, 4)
            ax.grid()
            ax.set_xlabel("P&L")
            ax.set_ylabel("Frequency")
            ax.set_ylim(0, len(training_pl[i]) / 2)
            sns.histplot(training_pl[i].numpy(), ax=ax, stat='count', kde=False, color='blue', label='P&L', binwidth=0.1)


        return FuncAnimation(fig, animate, frames=len(training_pl), repeat=True)

    def plot_training_loss(self):
        losses = self.training_logs["training_losses"]
        plot = sns.lineplot(x=range(len(losses)), y=losses)
        plot.set_title("Training Loss")
        plot.set_xlabel("Epoch")
        plot.set_ylabel("Loss")
        plot.grid()
        plot.set_yscale('log')
        return plot



    def plot_path(self, i):
        portfolio_logs = self.agent.portfolio_logs
        portfolio_value = portfolio_logs["portfolio_value"][i]
        cash_account = portfolio_logs["cash_account"][i]
        positions = portfolio_logs["positions"][i]
        hedge_paths = portfolio_logs["hedge_paths"][i]
        claim_payoff = portfolio_logs["claim_payoff"][i]
        claim_price = portfolio_logs["claim_payoff"].mean()
        claim_delta = None
        claim_name = self.claim.__class__.__name__
        hedge_names = [h.__class__.__name__ for h in self.hedging_instruments]

        if "claim_delta" in portfolio_logs and portfolio_logs["claim_delta"] is not None:
            claim_delta = portfolio_logs["claim_delta"][i]

        fig, ax = plt.subplots(2, 2)

        for a in ax.flatten():
            a.grid(True)  # Adding grid to each subplot

        for i in range(hedge_paths.shape[1]):
            sns.lineplot(x=range(len(hedge_paths[:, i])), y=hedge_paths[:, i], ax=ax[0, 0], label=hedge_names[i])
        ax[0,0].set_title("Price of Hedging Instruments")
        ax[0,0].set_xlabel("Time")
        ax[0,0].set_ylabel("Price [€]")
        ax[0,0].legend()


        for i in range(positions.shape[1]):
            sns.lineplot(x=range(len(positions[:, i])), y=positions[:, i], ax=ax[0, 1], label=hedge_names[i])
        ax[0, 1].set_title("Agent Positions")
        ax[0, 1].set_xlabel("Time")
        ax[0, 1].set_ylabel("Positions")
        # add line for y = 0
        ax[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.8)
        # make diverging from 0
        max_value = max(abs(positions.min()), abs(positions.max()))

        if claim_delta is not None:
            # add the delta of the claim to the graph
            ax[0, 1] = sns.lineplot(x=range(len(claim_delta)), y=claim_delta, ax=ax[0, 1], color='red', label=f'Delta of {claim_name}', alpha=0.9, linestyle='-.')
            max_delta = max(abs(claim_delta.min()), abs(claim_delta.max()))
            lim = max(max_value, max_delta) * 1.1
        else:
            lim = max_value * 1.1
        ax[0, 1].set_ylim(-lim, lim)


        pnl = portfolio_value + cash_account
        pnl += claim_price
        pnl[-1] -= claim_payoff

        sns.lineplot(x=range(len(pnl)), y=pnl, ax=ax[1, 0])
        ax[1, 0].set_title("Total P&L (Including Claim)")
        ax[1, 0].set_xlabel("Time")
        ax[1, 0].set_ylabel("P&L [€]")
        # add line for y = 0
        ax[1, 0].axhline(y=0, color='black', linestyle='--', alpha=0.5)
        # make diverging from 0
        max_value = max(abs(pnl.min()), abs(pnl.max())) * 1.1
        ax[1, 0].set_ylim(-max_value, max_value)

        final_cash = cash_account[-1] + claim_price
        final_portfolio_value = portfolio_value[-1]

        # Setting up the diverging bars
        categories = ['Cash', 'PV', 'CC', 'P&L']
        values = [final_cash, final_portfolio_value, -claim_payoff, final_cash + final_portfolio_value -claim_payoff]
        colors = ['red', 'blue', 'green', 'orange']

        ax[1,1].bar(categories, values, color=colors)
        ax[1,1].set_title("Final Portfolio Value Breakdown")
        ax[1,1].set_xlabel("Category")
        ax[1,1].set_ylabel("Value [€]")

        # Adding a horizontal line at y=0
        ax[1,1].hlines(0, -1, len(categories), colors='black', linestyles='dashed')

        # Set y limits to center the plot around 0
        max_value = max(abs(min(values)), abs(max(values))) * 1.1
        ax[1,1].set_ylim(-max_value, max_value)

        # Adding value annotations to the bars
        for i, v in enumerate(values):
            ax[1,1].text(i, v, f" {v:.2f}", color='black', ha='center', fontweight='bold')

        # make sure the text does not overlap
        fig.tight_layout()

        return fig


    def plot_runner(self, animate=False, save=False, file_prefix='plot', n = 5, compare = []):
        self.plot_training_loss()
        if save:
            plt.savefig(f'{file_prefix}_training_loss.pdf')
        plt.show()

        if animate:
            ani = self.training_pnl_animation()
            if save:
                ani.save(f'{file_prefix}_training_animation.mp4', writer='ffmpeg')
            plt.show()


        plot_dists([*compare, self], save=save, file_prefix=file_prefix)

        for i in range(n):
            self.plot_path(i)
            if save:
                plt.savefig(f'{file_prefix}_path_{i}.pdf')
        plt.show()



class SimpleRunner(ExperimentRunner):


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
            h_dim = 15,
            extra_params = None,
            ) -> None:

        self.agent = agents[self.agent_type](criterion, cost_function, hedging_instruments, extra_params,  step_interest_rate, h_dim=h_dim, pref_gpu=self.pref_gpu)
        loss = self.agent.validate(contingent_claim, int(1e6), T, logging=True)
        self.validation_logs = self.agent.validation_logs
        self.portfolio_logs = self.agent.portfolio_logs
        self.claim = contingent_claim
        self.hedging_instruments = hedging_instruments
        return loss

    def plot_runner(self, animate=False, save=False, file_prefix='plot'):
        self.plot_val_dist()
        if save:
            plt.savefig(f'{file_prefix}_val_dist.pdf')
        plt.show()



def plot_dists(runners: List[ExperimentRunner], save=False, file_prefix='plot', x_lim=2):
    # plot_val_dist for multiple runners
    plot = plt.figure()
    for runner in runners:
        val_profit = runner.validation_logs["validation_profit"]
        val_payoff = runner.validation_logs["validation_claim_payoff"]
        price = val_payoff.mean()
        val_loss = runner.validation_logs["validation_loss"]
        plot = sns.histplot((val_profit+price).numpy(), stat='count', kde=False, label=f'{runner.agent_type}, N: {len(val_profit)}, Loss: {val_loss:.2f}', binwidth=0.03, alpha=0.5)
    plot.set_title("P&L Distribution")
    plot.set_xlim(-x_lim, x_lim)
    plot.grid()
    plot.legend()
    if save:
        plt.savefig(f'{file_prefix}_comp_dist.pdf')
    return plot
