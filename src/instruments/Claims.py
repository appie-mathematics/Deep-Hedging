from abc import ABC, abstractmethod
# Payoff functions of derivatives

# claim -> instrument -> [derivatives, primary]

# claim: primary


class Claim(ABC):

    primary = None

    @abstractmethod
    def payoff(self, primary_path):
        # returns payoff of the claim at final timestep
        pass

    def primary(self):
        return self.primary

    def attach_primary(self, primary):
        self.primary = primary
        return self

class EuropeanCall(Claim):

    def __init__(self, strike):
        self.strike = strike

    def payoff(self, primary_path):
        return max(primary_path[-1] - self.strike, 0)
