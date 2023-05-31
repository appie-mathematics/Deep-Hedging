from abc import abstractmethod
from instruments.Claims import Claim


class Instrument(Claim):

    @abstractmethod
    def value(self, primary_path):
        # returns value of the instrument at each timestep
        pass

    def payoff(self, primary_path):
        # returns payoff of the instrument at final timestep
        return self.value(primary_path)[:,-1]

# class BSEuropeanCall(Instrument):

#     def __init__(self, strike):
#         self.strike = strike

#     def payoff(self, primary_path):
#         return np.maximum(primary_path - self.strike, 0)
