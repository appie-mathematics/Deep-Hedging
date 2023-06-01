

import torch
from agents.SimpleAgent import SimpleAgent


class RecurrentAgent(SimpleAgent):

    def input_dim(self) -> int:
        return super().input_dim() + self.N

    def feature_transform(self, state: tuple) -> torch.Tensor:
        """
        :param state: tuple
        :return: torch.Tensor
        """

        simple_features = super().feature_transform(state) # (P, N+1)

        current_positions = state[2][:, -1] # (P, N)

        # add the current positions to the features

        features = torch.cat([simple_features, current_positions], dim=1) # (P, 2N+1)

        return features.to(self.device)
