from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812


@dataclass
class AggregatorArgs:
    input_dim: int  # default: layer1(64), layer2(64), layer3(32)
    output_dim: int  # default: layer1(64), layer2(32), layer3(16)
    dropout: float  # default: layer1(0.1), layer2(0.1), layer3(0.1)


class Aggregator(nn.Module):
    def __init__(self, args: AggregatorArgs) -> None:
        super().__init__()

        self._input_dim = args.input_dim
        self._output_dim = args.output_dim

        self.message_dropout = nn.Dropout(p=args.dropout)
        self.activation = nn.LeakyReLU()

        self.linear1 = nn.Linear(in_features=self._input_dim, out_features=self._output_dim)
        self.linear2 = nn.Linear(in_features=self._input_dim, out_features=self._output_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize weights.
        """
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)

    def forward(self, ego_embeddings: torch.Tensor, attentive_matrix: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        -----------
        ego_embeddings: torch.Tensor
            Ego embeddings.
        attentive_matrix: torch.Tensor
            Attentive matrix.

        Returns
        --------
        embeddings: torch.Tensor
            Aggregated embeddings.
        """
        # Aggregate neighbors
        side_embeddings = torch.matmul(attentive_matrix, ego_embeddings)

        # Bi-interaction
        sum_embeddings: torch.Tensor = self.activation(self.linear1(ego_embeddings + side_embeddings))
        multiplied_embeddings: torch.Tensor = self.activation(self.linear2(ego_embeddings * side_embeddings))
        aggregated_embeddings: torch.Tensor = sum_embeddings + multiplied_embeddings

        # Dropout
        aggregated_embeddings = self.message_dropout(aggregated_embeddings)

        # L2 Normalization
        return F.normalize(aggregated_embeddings, p=2, dim=1)
