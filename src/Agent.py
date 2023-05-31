from typing import List, Set
import torch
from Costs import CostFunction
from instruments.Claims import Claim
from tqdm import tqdm
from instruments.Instrument import Instrument
from instruments.Primaries import Primary
import matplotlib.pyplot as plt

class Agent(torch.nn.Module):
    """
    Base class for deep hedge agent
    """

    def __init__(self, model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 criterion: torch.nn.Module,
                 device: torch.device,
                 cost_function: CostFunction,
                 interest_rate = 0.05):
        """
        :param model: torch.nn.Module
        :param optimizer: torch.optim
        :param criterion: torch.nn
        :param device: torch.device
        """
        super(Agent, self).__init__()
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.cost_function = cost_function
        self.device = device
        self.interest_rate = interest_rate

    def forward(self, x):
        """
        :param x: torch.Tensor
        :return: torch.Tensor
        """
        return self.model(x)


    def policy(self, x):
        """
        :param x: torch.Tensor
        :return: torch.Tensor
        """
        return self.forward(x)


    # takes all the historic data and return feature vector
    def feature_transform(self, state: tuple) -> torch.Tensor:
        """
        :param state: tuple
        :return: torch.Tensor
        """
        return torch.cat(state, dim=-1)


    # returns the final p&l
    def compute_portfolio(self, hedge_paths) -> torch.Tensor:
        # number of time steps
        P, T, N = hedge_paths.shape

        cash_account = torch.zeros(P, T)
        portfolio_value = torch.zeros(P, T)
        positions = torch.zeros(P, T, N)

        for t in range(1, T):
            # define state
            state = hedge_paths[:,:t], cash_account[:,:t], positions[:,:t]
            # compute action
            feature_vector = self.feature_transform(state)
            # select action
            action = self.policy(feature_vector) # (P, N)
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

        # return final portfolio value
        # print("pfv", portfolio_value[:,-1].mean(dim=0).item())
        # print("cash", cash_account[:,-1].mean(dim=0).item())
        return portfolio_value[:,-1] + cash_account[:,-1]


    def loss(self, contingent_claim: Claim , hedging_instruments: List[Instrument], P, T):
        """
        :param contingent_claim: Instrument
        :param hedging_instruments: List[Instrument]
        :param paths: int
        :return: None
        """
        # number of time steps: T
        # number of hedging instruments: N
        # number of paths: P

        # 1. check how many primaries are invloved
        primaries: Set[Primary] = set([hedge.primary() for hedge in hedging_instruments])
        primaries.add(contingent_claim.primary())

        # 2. generate paths for all the primaries
        primary_paths = {primary: primary.simulate(P, T) for primary in primaries}

        # 3. generate paths for all derivatives based on the primary paths
        hedge_paths = [instrument.value(primary_paths[instrument.primary()]) for instrument in hedging_instruments] # N x tensor(P x T)
        # convert to P x T x N tensor
        hedge_paths = torch.stack(hedge_paths, dim=-1) # P x T x N

        # 4. compute claim payoff based on primary paths
        claim_payoff = contingent_claim.payoff(primary_paths[contingent_claim.primary()]) # P x 1

        portfolio_value = self.compute_portfolio(hedge_paths) # P
        profit = portfolio_value - claim_payoff # P
        print(profit.mean())
        return - self.criterion(profit).mean()




    def fit(self, contingent_claim: Claim, hedging_instruments: List[Instrument], epochs = 50, paths = 100, verbose = False, T = 365):
        """
        :param contingent_claim: Instrument
        :param hedging_instruments: List[Instrument]
        :param epochs: int
        :param batch_size: int
        :param verbose: bool
        :return: None
        """
        losses = []

        for epoch in tqdm(range(epochs), desc="Training", total=epochs):
            # TODO: define number of time steps
            loss = self.loss(contingent_claim, hedging_instruments, paths, T)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            if verbose:
                print("Epoch: {}, Loss: {}".format(epoch, loss.item()))

            # eventually add validation

        if verbose:
            plt.plot(losses, label='loss')
            plt.show()




class SimpleAgent(Agent):

    def feature_transform(self, state: tuple) -> torch.Tensor:
        """
        :param state: tuple
        :return: torch.Tensor
        """
        # only know the current price
        paths, cash_account, positions = state

        last_prices = paths[:, -1, :] # (P, N)

        # log prices
        log_prices = torch.log(last_prices) # (P, N)

        # return squeezed tensor
        return log_prices
