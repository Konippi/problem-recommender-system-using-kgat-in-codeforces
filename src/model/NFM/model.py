from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from src.model.NFM.hidden_layer import HiddenLayer, HiddenLayerArgs


@dataclass
class NFMArgs:
    user_num: int
    item_num: int
    entity_num: int
    embedding_dim: int = 64
    loss_lambda: float = 1e-5
    hidden_dim: list[int] = field(default_factory=lambda: [64, 32, 16])
    message_dropout: list[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])


class NFMMode(IntEnum):
    TRAIN = 0
    PREDICT = 1


class NFM(nn.Module):
    def __init__(self, args: NFMArgs) -> None:
        super().__init__()

        self._user_num = args.user_num
        self._item_num = args.item_num
        self._entity_num = args.entity_num
        self._feature_num = self._user_num + self._entity_num
        self._embedding_dim = args.embedding_dim
        self._loss_lambda = args.loss_lambda
        self._hidden_dim = args.hidden_dim
        self._message_dropout = args.message_dropout
        self._layer_size = len(self._hidden_dim)

        self._linear = nn.Linear(self._feature_num, 1)
        nn.init.xavier_uniform_(self._linear.weight)

        self._feature_embedding = nn.Parameter(torch.Tensor(self._feature_num, self._embedding_dim))
        nn.init.xavier_uniform_(self._feature_embedding)

        # Hidden Layer
        self._hidden_laryes = nn.ModuleList()
        layers = [self._embedding_dim, *self._hidden_dim]
        for layer_id in range(self._layer_size):
            self._hidden_laryes.append(
                HiddenLayer(
                    HiddenLayerArgs(
                        input_dim=layers[layer_id],
                        output_dim=layers[layer_id + 1],
                        dropout=self._message_dropout[layer_id],
                    )
                )
            )

        # Output layer
        self._output_layer = nn.Linear(self._hidden_dim[-1], 1, bias=False)
        nn.init.xavier_uniform_(self._output_layer.weight)

    def _calc_score(
        self,
        feature_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate score.

        Parameters
        -----------
        feature_values: torch.Tensor
            Feature values.

        Returns
        --------
        score: torch.Tensor
            Score.
        """
        # Bi-interaction
        sum_embeddings = torch.mm(feature_values, self._feature_embedding).pow(2)
        squared_embeddings = torch.mm(torch.pow(feature_values, 2), torch.pow(self._feature_embedding, 2))
        z = 0.5 * (sum_embeddings - squared_embeddings)

        for layer in self._hidden_laryes:
            z = layer(z)

        # Output layer
        y: torch.Tensor = self._output_layer(z)
        y = self._linear(feature_values) + y
        return y.squeeze()

    def _calc_loss(
        self,
        positive_feature_values: torch.Tensor,
        negative_feature_values: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate loss.

        Parameters
        -----------
        positive_feature_values: torch.Tensor
            Positive feature values.
        negative_feature_values: torch.Tensor
            Negative feature values.

        Returns
        --------
        loss: torch.Tensor
            Loss.
        """
        positive_scores = self._calc_score(positive_feature_values)
        negative_scores = self._calc_score(negative_feature_values)

        loss: torch.Tensor = -torch.log(1e-10 + F.sigmoid(positive_scores - negative_scores)).mean()
        l2_loss: torch.Tensor = torch.norm(self._output_layer.weight, 2).pow(2) / 2

        return loss + self._loss_lambda * l2_loss

    def build_optimizer(self, lr: float) -> None:
        """
        Build the optimizer.

        Parameters
        ----------
        lr: float
            The learning rate.
        """
        self._optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

    def update_weights(self) -> None:
        """
        Update the weights for collaborative filtering.
        """
        self._optimizer.step()
        self._optimizer.zero_grad()

    def forward(self, *args: Any, mode: NFMMode) -> torch.Tensor:  # noqa: ANN401
        match mode:
            case NFMMode.TRAIN:
                return self._calc_loss(*args)
            case NFMMode.PREDICT:
                return self._calc_score(*args)
