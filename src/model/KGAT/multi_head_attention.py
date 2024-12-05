import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(self, cf_embedding_dim: int, kg_embedding_dim: int, head_num: int = 4) -> None:
        super().__init__()

        self._head_num = head_num
        self._cf_embedding_dim = cf_embedding_dim
        self._kg_embedding_dim = kg_embedding_dim
        self._depth = self._cf_embedding_dim // self._head_num
        self._query_weight = nn.Linear(self._cf_embedding_dim, self._kg_embedding_dim)
        self._key_weight = nn.Linear(self._cf_embedding_dim, self._kg_embedding_dim)
        self._value_weight = nn.Linear(self._cf_embedding_dim, self._kg_embedding_dim)
        self._output = nn.Linear(self._kg_embedding_dim, self._kg_embedding_dim)

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """
        Initialize weights.
        """
        nn.init.xavier_uniform_(self._query_weight.weight)
        nn.init.xavier_uniform_(self._key_weight.weight)
        nn.init.xavier_uniform_(self._value_weight.weight)
        nn.init.xavier_uniform_(self._output.weight)

    def _split_heads(self, x: torch.Tensor, batch_size: int) -> torch.Tensor:
        x = x.view(batch_size, -1, self._head_num, self._depth)
        return x.transpose(1, 2)

    def forward(
        self,
        head_embedding: torch.Tensor,
        relation_embedding: torch.Tensor,
        tail_embedding: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = head_embedding.size(0)

        query = self._split_heads(self._query_weight(head_embedding), batch_size)
        key = self._split_heads(self._key_weight(tail_embedding + relation_embedding), batch_size)
        value = self._split_heads(self._value_weight(tail_embedding + relation_embedding), batch_size)

        # Scaled Dot-Product Attention
        attention = torch.matmul(query, key.transpose(-2, -1)) / (self._depth**0.5)
        attention = torch.softmax(attention, dim=-1)
        attention = torch.matmul(attention, value)
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self._kg_embedding_dim)
        output: torch.Tensor = self._output(attention)
        return output
