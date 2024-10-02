from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class HiddenLayerArgs:
    input_dim: int
    output_dim: int
    dropout: float


class HiddenLayer(nn.Module):
    def __init__(self, args: HiddenLayerArgs) -> None:
        super().__init__()

        self._input_dim = args.input_dim
        self._output_dim = args.output_dim
        self._dropout = args.dropout

        self._linear = nn.Linear(self._input_dim, self._output_dim)
        self._activation = nn.ReLU()
        self._message_dropout = nn.Dropout(self._dropout)

        nn.init.xavier_uniform_(self._linear.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tmp = self._linear(x)
        tmp = self._message_dropout(tmp)
        out: torch.Tensor = self._activation(tmp)
        return out
