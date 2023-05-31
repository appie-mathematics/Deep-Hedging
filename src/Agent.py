import torch


class Agent(torch.Module):
    """
    Base class for deep hedge agent
    """

    def __init__(self, model, optimizer, criterion, device, simulator, cost_function, interest_rate = 0.05):
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
        self.simulator = simulator
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
    def feature_transform(self, state):
        return state[-1]


    # returns the final p&l
    def compute_portfolio(self, hedge_paths) -> float:
        # number of time steps
        P, T, N = hedge_paths.shape

        cash_account = torch.zeros(P, T)
        portfolio_value = torch.zeros(P, T)
        positions = torch.zeros(P, T, N)

        for t in range(1, T):
            state = hedge_paths[:,:t], cash_account[:,:t], positions[:,:t], portfolio_value[:,:t]
            feature_vector = self.feature_transform(state)
            action = self.policy(feature_vector) # (P, N)
            positions[:, t] = positions[:, t-1] + action
            cost_of_action = self.cost_function(action, state) # (P, 1)
            # TODO: check if other operations are possible
            spent = (action * hedge_paths[:, t].squeeze()).sum(dim=-1) + cost_of_action # (P, 1)
            cash_account[:,t] = cash_account[:, t-1] * (1+self.interest_rate) - spent # (P, 1)
            portfolio_value[:,t] = positions[:, t] @ hedge_paths[:, t].transpose(1,2) #(P, T)

        return portfolio_value[:,-1] + cash_account[:,-1]


    def loss(self, contingent_claim: Derivative , hedging_instruments: List[Primary], paths) -> float:
        """
        :param contingent_claim: Instrument
        :param hedging_instruments: List[Instrument]
        :param paths: int
        :return: None
        """
        # number of time steps: T
        # number of hedging instruments: N
        # number of paths: P

        primary_of_claim = contingent_claim.primary

        # generate paths
        hedge_paths = self.simulator.generate_paths(hedging_instruments, paths) # P x T x N

        primary_path = None # P x T x 1
        if primary_of_claim in hedging_instruments:
            primary_path = hedge_paths[:, :, hedging_instruments.index(primary_of_claim)]
        else:
            primary_path = self.simulator.generate_paths([primary_of_claim], paths)

        portfolio_value = self.compute_portfolio(hedge_paths) # P
        return self.criterion(portfolio_value + contingent_claim.payoff(primary_path)).mean() # scalar




    def fit(self, contingent_claim: Instrument, hedging_instruments: List[Instrument], epochs = 50, paths = 100, verbose = False):
        """
        :param contingent_claim: Instrument
        :param hedging_instruments: List[Instrument]
        :param epochs: int
        :param batch_size: int
        :param verbose: bool
        :return: None
        """

        for epoch in epochs:
            loss = self.loss(contingent_claim, hedging_instruments, paths)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if verbose:
                print("Epoch: {}, Loss: {}".format(epoch, loss.item()))
