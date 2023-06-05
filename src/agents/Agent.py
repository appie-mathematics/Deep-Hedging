from abc import ABC, abstractmethod
from typing import List
from matplotlib.animation import FuncAnimation
import torch
from Costs import CostFunction
from instruments.Claims import Claim
from tqdm import tqdm
from instruments.Instruments import Instrument
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
                 cost_function: CostFunction,
                 hedging_instruments: List[Instrument],
                 interest_rate = 0.05,
                 lr=0.005,
                 pref_gpu=True):
        """
        :param model: torch.nn.Module
        :param optimizer: torch.optim
        :param criterion: torch.nn
        :param device: torch.device
        """
        super(Agent, self).__init__()
        device: torch.device = torch.device('cpu')
        if pref_gpu:
            if torch.cuda.is_available():
                device = torch.device('cuda')
                print("Running on CUDA GPU")

            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = torch.device("mps")
                print("Running on MPS GPU")

        self.device = device
        self.lr = lr
        self.criterion = criterion.to(device)
        self.cost_function = cost_function
        self.interest_rate = interest_rate
        self.hedging_instruments = hedging_instruments
        self.N = len(hedging_instruments)
        self.to(device)
        self.training_logs = dict()
        self.portfolio_logs = dict()
        self.validation_logs = dict()

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
    def compute_portfolio(self, hedge_paths, logging = False) -> torch.Tensor:
        # number of time steps
        P, T, N = hedge_paths.shape

        cash_account = torch.zeros(P, T, device=self.device)
        portfolio_value = torch.zeros(P, T, device=self.device)
        positions = torch.zeros(P, T, N, device=self.device)

        state = hedge_paths[:,:1], cash_account[:,:1], positions[:,:1]
        action = self.policy(state)
        positions[:, 0] = action
        cost_of_action = self.cost_function(action, state)
        purchase = (action * hedge_paths[:, 0]).sum(dim=-1)
        spent = purchase + cost_of_action # (P, 1)
        # update cash account
        cash_account[:,0] = - spent # (P, 1)
        # update portfolio value
        portfolio_value[:,0] = purchase # (P, 1)

        for t in range(1, T):
            # define state
            state = hedge_paths[:,:t+1], cash_account[:,:t], positions[:,:t]
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


        if logging:
            self.portfolio_logs = {
                "portfolio_value": portfolio_value.detach().cpu(),
                "cash_account": cash_account.detach().cpu(),
                "positions": positions.detach().cpu(),
                "hedge_paths": hedge_paths.detach().cpu(),
            }


        return portfolio_value[:,-1] + cash_account[:,-1]

    def generate_paths(self, P, T, contingent_claim):
        # 1. check how many primaries are invloved
        primaries: set = set([hedge.primary() for hedge in self.hedging_instruments])
        primaries.add(contingent_claim.primary())

        # 2. generate paths for all the primaries
        primary_paths = {primary: primary.simulate(P, T) for primary in primaries}

        # 3. generate paths for all derivatives based on the primary paths
        hedge_paths = [instrument.value(primary_paths[instrument.primary()]).to(self.device) for instrument in self.hedging_instruments] # N x tensor(P x T)
        # convert to P x T x N tensor
        hedge_paths = torch.stack(hedge_paths, dim=-1) # P x T x N

        # 4. compute claim payoff based on primary paths
        claim_payoff = contingent_claim.payoff(primary_paths[contingent_claim.primary()]).to(self.device) # P x 1
        return hedge_paths, claim_payoff

    def pl(self, contingent_claim: Claim, P, T, logging = False):
        """
        :param contingent_claim: Instrument
        :param paths: int
        :return: None
        """
        # number of time steps: T
        # number of hedging instruments: N
        # number of paths: P

        hedge_paths, claim_payoff = self.generate_paths(P, T, contingent_claim) # P x T x N, P x 1

        portfolio_value = self.compute_portfolio(hedge_paths, logging) # P
        profit = portfolio_value - claim_payoff # P
        if logging:
            self.portfolio_logs["claim_payoff"] = claim_payoff.detach().cpu()

        return profit, claim_payoff


    def fit(self, contingent_claim: Claim, epochs = 50, paths = 100, verbose = False, T = 365, logging = True):
        """
        :param contingent_claim: Instrument
        :param epochs: int
        :param batch_size: int
        :param verbose: bool
        :return: None
        """
        losses = []

        for epoch in tqdm(range(epochs), desc="Training", total=epochs, leave=False, unit="epoch"):
            self.train()
            pl, _ = self.pl(contingent_claim, paths, T, False)
            loss = - self.criterion(pl)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            if verbose:
                print(f"Epoch: {epoch}, Loss: {loss.item(): .2f}")

            if logging:
                if "training_PL" not in self.training_logs:
                    self.training_logs["training_PL"] = torch.Tensor(epochs, paths).cpu()

                self.training_logs["training_PL"][epoch] = pl.detach().cpu()
        if logging:
            self.training_logs["training_losses"] = torch.Tensor(losses).cpu()

        return losses

    def validate(self, contingent_claim: Claim, paths = 10000, T = 365, logging = True):
        """
        :param contingent_claim: Instrument
        :param epochs: int
        :param batch_size: int
        :return: None
        """
        with torch.no_grad():
            self.eval()
            profit, claim_payoff = self.pl(contingent_claim, paths, T, True)
            loss = self.criterion(profit)
            if logging:
                self.validation_logs["validation_profit"] = profit.detach().cpu()
                self.validation_logs["validation_claim_payoff"] = claim_payoff.detach().cpu()
                self.validation_logs["validation_loss"] = loss.detach().cpu()

            return loss.item()
