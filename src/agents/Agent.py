from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import List, Set
from matplotlib.animation import FuncAnimation
import numpy as np
import torch
from Costs import CostFunction
from instruments.Claims import Claim
from tqdm import tqdm
from instruments.Instruments import Instrument
from instruments.Primaries import Primary
import matplotlib.pyplot as plt
import seaborn as sns

class Agent(torch.nn.Module, ABC):
    """
    Base class for deep hedge agent
    """
    network: torch.nn.Module
    optimizer: torch.optim.Optimizer

    def __init__(self,
                 criterion: torch.nn.Module,
                 device: torch.device,
                 cost_function: CostFunction,
                 hedging_instruments: List[Instrument],
                 interest_rate = 0.05,
                 lr=0.005):
        """
        :param model: torch.nn.Module
        :param optimizer: torch.optim
        :param criterion: torch.nn
        :param device: torch.device
        """
        super(Agent, self).__init__()
        self.device = device
        self.lr = lr
        self.criterion = criterion.to(device)
        self.cost_function = cost_function
        self.interest_rate = interest_rate
        self.hedging_instruments = hedging_instruments
        self.N = len(hedging_instruments)
        self.to(device)
        self.profit_logs = torch.Tensor()

    @abstractmethod
    def forward(self, state: tuple) -> torch.Tensor: # (P, N)
        pass


    def policy(self, state: tuple) -> torch.Tensor:
        """
        :param x: torch.Tensor
        :return: torch.Tensor
        """
        return self.forward(state)

    # returns the final p&l
    def compute_portfolio(self, hedge_paths) -> torch.Tensor:
        # number of time steps
        P, T, N = hedge_paths.shape

        cash_account = torch.zeros(P, T, device=self.device)
        portfolio_value = torch.zeros(P, T, device=self.device)
        positions = torch.zeros(P, T, N, device=self.device)

        for t in range(1, T):
            # define state
            state = hedge_paths[:,:t], cash_account[:,:t], positions[:,:t]
            # compute action
            action = self.policy(state) # (P, N)
            # update positions
            positions[:, t] = positions[:, t-1] + action # (P, N)
            # compute cost of action
            cost_of_action = self.cost_function(action, state) # (P, 1)
            # TODO: check if other operations are possible
            spent = (action * hedge_paths[:, t]).sum(dim=-1) + cost_of_action # (P, 1)
            # update cash account
            cash_account[:,t] = cash_account[:, t-1] * (1+self.interest_rate) - spent # (P, 1)
            # update portfolio value
            portfolio_value[:,t] = (positions[:,t] * hedge_paths[:,t]).sum(dim=-1) # (P, 1)

        # # plot portfolio value, cash account, positions
        # plt.plot(portfolio_value.detach().mean(dim=0), label='portfolio value')
        # plt.plot(cash_account.detach().mean(dim=0), label='cash account')
        # plt.plot(positions.detach().mean(dim=0), label='positions')
        # plt.plot(hedge_paths.detach().mean(dim=0), label='hedge paths')
        # plt.legend()
        # plt.grid()
        # plt.show()

        # print(f"average portfolio value: {portfolio_value.squeeze().mean().item(): .2f}")

        # return final portfolio value
        # print("pfv", portfolio_value[:,-1].mean(dim=0).item())
        # print("cash", cash_account[:,-1].mean(dim=0).item())
        return portfolio_value[:,-1] + cash_account[:,-1]


    def loss(self, contingent_claim: Claim, P, T, i):
        """
        :param contingent_claim: Instrument
        :param paths: int
        :return: None
        """
        # number of time steps: T
        # number of hedging instruments: N
        # number of paths: P

        # 1. check how many primaries are invloved
        primaries: set = set([hedge.primary() for hedge in self.hedging_instruments])
        primaries.add(contingent_claim.primary())

        # 2. generate paths for all the primaries
        primary_paths = {primary: primary.simulate(P, T) for primary in primaries}

        # 3. generate paths for all derivatives based on the primary paths
        hedge_paths = [instrument.value(primary_paths[instrument.primary()]) for instrument in self.hedging_instruments] # N x tensor(P x T)
        # convert to P x T x N tensor
        hedge_paths = torch.stack(hedge_paths, dim=-1) # P x T x N

        # 4. compute claim payoff based on primary paths
        claim_payoff = contingent_claim.payoff(primary_paths[contingent_claim.primary()]).to(self.device) # P x 1

        portfolio_value = self.compute_portfolio(hedge_paths.to(self.device)) # P
        profit = portfolio_value - claim_payoff # P
        with torch.no_grad():
            self.profit_logs[i,:] = profit.detach().cpu()

        return - self.criterion(profit).mean()




    def fit(self, contingent_claim: Claim, epochs = 50, paths = 100, verbose = False, T = 365):
        """
        :param contingent_claim: Instrument
        :param epochs: int
        :param batch_size: int
        :param verbose: bool
        :return: None
        """
        losses = []
        self.profit_logs = torch.Tensor(epochs, paths).cpu()

        for epoch in tqdm(range(epochs), desc="Training", total=epochs):
            self.train()
            loss = self.loss(contingent_claim, paths, T, epoch)
            print(loss.item())
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            #if verbose:
            #    print("Epoch: {}, Loss: {}".format(epoch, loss.item()))

            # eventually add validation

        if verbose:
            plt.plot(losses, label='loss')
            plt.show()

            fig, ax = plt.subplots()

            def animate(i):
                ax.clear()
                sns.histplot(self.profit_logs[i].numpy(), ax=ax, stat='density', kde=True, color='blue', label='P&L', binwidth=.05)
                ax.set_xlim(-5, 5)
                ax.set_ylim(0, 2)
                ax.grid()
                ax.set_title(f"Epoch {i+1}")
                ax.set_xlabel("Profit")
                ax.set_ylabel("Density")

            ani = FuncAnimation(fig, animate, frames=len(self.profit_logs), repeat=True)

            plt.show()
