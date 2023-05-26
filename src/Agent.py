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
    def compute_portfolio(self, claim_path, hedge_paths) -> float:
        # number of time steps
        T = claim_path.shape[0]
        # number of available hedging instruments
        N = hedge_paths.shape[0]

        cash_account = torch.zeros(1, T)
        portfolio_value = torch.zeros(1, T)
        positions = torch.zeros(N, T)

        for t in range(1, T):
            state = claim_path[:t], hedge_paths[:t], positions[:,:t], cash_account[:t], portfolio_value[:t]
            feature_vector = self.feature_transform(state)
            action = self.policy(feature_vector)
            positions[N, t] = positions[N, t-1] + action
            cost_of_action: int = self.cost_function(action, state)
            # TODO: fix matrix mult?
            spent: int = action @ hedge_paths[t].T + cost_of_action
            cash_account[t] = cash_account[t-1] * (1+self.interest_rate) - spent
            portfolio_value[t] = positions[N, t] @ hedge_paths[t].T

        return portfolio_value[-1] + cash_account[-1]


    def loss(self, contingent_claim: Derivative , hedging_instruments: List[Primary], paths) -> float:
        """
        :param contingent_claim: Instrument
        :param hedging_instruments: List[Instrument]
        :param paths: int
        :return: None
        """

        # generate paths
        hedge_paths = self.simulator.generate_paths(hedging_instruments, paths)
        portfolio_value = self.compute_portfolio(claim_path, hedge_paths)
        return self.criterion(portfolio_value).mean()




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
