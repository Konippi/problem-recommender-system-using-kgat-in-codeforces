from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

from src.model.KGAT.aggregator import Aggregator, AggregatorArgs


@dataclass
class KGATArgs:
    user_num: int
    entity_num: int
    relation_num: int
    cf_embedding_dim: int = 64
    kg_embedding_dim: int = 64
    attentive_matrix: torch.Tensor | None = None
    message_dropout: list[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    layer_size: list[int] = field(default_factory=lambda: [64, 32, 16])
    regularization_params: list[float] = field(default_factory=lambda: [1e-5, 1e-5])


class KGATMode(IntEnum):
    TRAIN_CF = 0
    TRAIN_KG = 1
    UPDATE_ATTENTION = 2
    PREDICT = 3


class KGAT(nn.Module):
    def __init__(self, args: KGATArgs) -> None:
        super().__init__()

        ####################################
        # Dataset
        ####################################
        self._user_num = args.user_num
        self._entity_num = args.entity_num
        self._relation_num = args.relation_num

        ####################################
        # Knowledge Graph Attention Network
        ####################################
        # Part of Hyperparameters
        self._message_dropout = args.message_dropout
        self._layer_dims = args.layer_size
        self._layer_num = len(self._layer_dims)
        self._regularization_params = args.regularization_params

        # Part of Bipartite Graph
        self._cf_embedding_dim = args.cf_embedding_dim

        # Part of Knowledge Graph
        self._kg_embedding_dim = args.kg_embedding_dim

        self._user_entity_embedding = nn.Embedding(
            num_embeddings=self._user_num + self._entity_num,
            embedding_dim=self._cf_embedding_dim,
        )
        self._relation_embedding = nn.Embedding(
            num_embeddings=self._relation_num,
            embedding_dim=self._kg_embedding_dim,
        )
        self._trans_matrix = nn.Parameter(
            data=torch.Tensor(
                self._relation_num,
                self._cf_embedding_dim,
                self._kg_embedding_dim,
            )
        )

        # Initialize the weights of embeddings
        self._initialize_weights()

        # Initialize the aggregator
        self._aggregator_layers = nn.ModuleList()
        self._initialize_aggregator()

        # Public for visualization
        self.attentive_matrix = nn.Parameter(
            data=torch.sparse_coo_tensor(
                indices=torch.empty(size=(2, 0), dtype=torch.long),
                values=torch.empty(size=(0,), dtype=torch.float32),
                size=torch.Size([self._user_num + self._entity_num, self._user_num + self._entity_num]),
            )
        )
        if args.attentive_matrix is not None:
            self.attentive_matrix.data = args.attentive_matrix
        self.attentive_matrix.requires_grad = False

    def _initialize_weights(self) -> None:
        """
        Initialize the weights of embeddings.
        """
        nn.init.xavier_uniform_(tensor=self._user_entity_embedding.weight)
        nn.init.xavier_uniform_(tensor=self._relation_embedding.weight)
        nn.init.xavier_uniform_(tensor=self._trans_matrix)

    def _initialize_aggregator(self) -> None:
        """
        Initialize the Aggregator. (Bi-Interaction)
        """
        # User-entity embedding + L-hop neighbor embeddings(Default: L=3[64, 32, 16])
        layers = [self._cf_embedding_dim, *self._layer_dims]
        for layer_id in range(self._layer_num):
            self._aggregator_layers.append(
                Aggregator(
                    AggregatorArgs(
                        input_dim=layers[layer_id],
                        output_dim=layers[layer_id + 1],
                        dropout=self._message_dropout[layer_id],
                    )
                )
            )

    def _build_cf_embeddings(self) -> torch.Tensor:
        """
        Build the embeddings for collaborative filtering.

        Returns
        -------
        cf_embeddings: torch.Tensor
            The embeddings for collaborative filtering.
        """
        ego_embedding = self._user_entity_embedding.weight
        all_embeddings = [ego_embedding]

        for aggregator_layer in self._aggregator_layers:
            ego_embedding = aggregator_layer(ego_embedding, self.attentive_matrix)
            all_embeddings.append(ego_embedding)

        return torch.cat(tensors=all_embeddings, dim=1)  # (user_num + entity_num, concatentated_dim)

    def _l2_mean_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Calculate the L2 Mean Loss.

        Parameters
        ----------
        embeddings: torch.Tensor
            The embeddings.

        Returns
        -------
        l2_loss: torch.Tensor
            The L2 loss.
        """
        return torch.mean(
            torch.sum(
                input=torch.pow(input=embeddings, exponent=2),
                dim=1,
                keepdim=False,
            )
            / 2.0
        )

    def _calc_cf_loss(
        self,
        user_ids: torch.Tensor,
        positive_item_ids: torch.Tensor,
        negative_item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the loss for collaborative filtering.

        Parameters
        ----------
        user_ids: torch.Tensor
            The user IDs.
        positive_item_ids: torch.Tensor
            The positive item IDs.
        negative_item_ids: torch.Tensor
            The negative item IDs.

        Returns
        -------
        cf_loss: torch.Tensor
            The loss for collaborative filtering.
        """
        all_embeddings = self._build_cf_embeddings()  # (user_num + entity_num, concatentated_dim)
        user_embedding = all_embeddings[user_ids.long()]  # (cf_batch_size, concatentated_dim)
        positive_item_embedding = all_embeddings[positive_item_ids.long()]  # (cf_batch_size, concatentated_dim)
        negative_item_embedding = all_embeddings[negative_item_ids.long()]  # (cf_batch_size, concatentated_dim)
        positive_scores = torch.sum(user_embedding * positive_item_embedding, dim=1)  # (cf_batch_size,)
        negative_scores = torch.sum(user_embedding * negative_item_embedding, dim=1)  # (cf_batch_size,)

        cf_loss = -F.logsigmoid(positive_scores - negative_scores).mean()
        l2_loss = (
            self._l2_mean_loss(user_embedding)
            + self._l2_mean_loss(positive_item_embedding)
            + self._l2_mean_loss(negative_item_embedding)
        )

        return cf_loss + self._regularization_params[0] * l2_loss

    def _calc_kg_loss(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        positive_tails: torch.Tensor,
        negative_tails: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the loss for knowledge graph.

        Parameters
        ----------
        heads: torch.Tensor
            The head entities.
        relations: torch.Tensor
            The relations.
        positive_tails: torch.Tensor
            The positive tail entities.
        negative_tails: torch.Tensor
            The negative tail entities.

        Returns
        -------
        kg_loss: torch.Tensor
            The loss for knowledge graph.
        """
        _relation_embedding: torch.Tensor = self._relation_embedding(relations)  # (kg_batch_size, relation_dim)
        trans_matrix_by_relation: torch.Tensor = self._trans_matrix[
            relations
        ]  # (kg_batch_size, cf_embedding_dim, relation_dim)
        head_embedding: torch.Tensor = self._user_entity_embedding(heads)  # (kg_batch_size, cf_embedding_dim)
        positive_tails_embedding: torch.Tensor = self._user_entity_embedding(
            positive_tails
        )  # (kg_batch_size, cf_embedding_dim)
        negative_tails_embedding: torch.Tensor = self._user_entity_embedding(
            negative_tails
        )  # (kg_batch_size, cf_embedding_dim)

        # Transform to the relation space (Remove extra dimensions)
        transformed_head_embedding = torch.matmul(head_embedding.unsqueeze(1), trans_matrix_by_relation).squeeze(
            1
        )  # (kg_batch_size, relation_dim)
        transformed_positive_tail_embedding = torch.matmul(
            positive_tails_embedding.unsqueeze(1),
            trans_matrix_by_relation,
        ).squeeze(1)  # (kg_batch_size, relation_dim)
        transformed_negative_tail_embedding = torch.matmul(
            negative_tails_embedding.unsqueeze(1),
            trans_matrix_by_relation,
        ).squeeze(1)  # (kg_batch_size, relation_dim)

        # Calculate the score
        positive_scores = torch.sum(
            input=torch.pow(
                input=transformed_head_embedding + _relation_embedding - transformed_positive_tail_embedding,
                exponent=2,
            ),
            dim=1,
        )  # (kg_batch_size,)
        negative_scores = torch.sum(
            input=torch.pow(
                input=transformed_head_embedding + _relation_embedding - transformed_negative_tail_embedding,
                exponent=2,
            ),
            dim=1,
        )  # (kg_batch_size,)

        kg_loss = -F.logsigmoid(negative_scores - positive_scores).mean()
        l2_loss = (
            self._l2_mean_loss(transformed_head_embedding)
            + self._l2_mean_loss(_relation_embedding)
            + self._l2_mean_loss(transformed_positive_tail_embedding)
            + self._l2_mean_loss(transformed_negative_tail_embedding)
        )

        return kg_loss + self._regularization_params[1] * l2_loss

    def _calc_popularity_weights(
        self,
        tails: torch.Tensor,
        problem_with_submission_cnt: dict[int, int],
    ) -> torch.Tensor:
        """
        Calculate the popularity weights.

        Parameters
        ----------
        tails: torch.Tensor
            The tail entities.
        problem_with_submission_cnt: dict[int, int]
            The problem with submission count.

        Returns
        -------
        weights: torch.Tensor
            The weights.
        """
        return torch.tensor(
            [1.0 / (np.log1p(problem_with_submission_cnt[tail.item()]) ** 2 + 1) for tail in tails],
            dtype=torch.float32,
        )

    def _update_attention_by_batch(
        self,
        heads: torch.Tensor,
        tails: torch.Tensor,
        relation_idx: torch.Tensor,
        problem_with_submission_cnt: dict[int, int],
    ) -> torch.Tensor:
        """
        Update the attention matrix.

        Parameters
        ----------
        heads: torch.Tensor
            The head entities.
        tails: torch.Tensor
            The tail entities.
        relation: torch.Tensor
            The relation.
        problem_with_submission_cnt: dict[int, int]
            The problem with submission count.

        Returns
        -------
        updated_attention: torch.Tensor
            The updated attention.
        """
        _relation_embedding = self._relation_embedding.weight[relation_idx]
        trans_matrix_by_relation = self._trans_matrix[relation_idx]
        head_embedding = self._user_entity_embedding.weight[heads]
        tail_embedding = self._user_entity_embedding.weight[tails]

        transformed_head_embedding = torch.matmul(
            input=head_embedding,
            other=trans_matrix_by_relation,
        )
        transformed_tail_embedding = torch.matmul(
            input=tail_embedding,
            other=trans_matrix_by_relation,
        )

        attention = torch.sum(
            input=transformed_tail_embedding * torch.tanh(input=transformed_head_embedding + _relation_embedding),
            dim=1,
        )

        user_to_problem_relation_idx = 0
        if relation_idx == user_to_problem_relation_idx:
            attention *= self._calc_popularity_weights(tails, problem_with_submission_cnt)

        return attention

    def _update_attention(
        self,
        heads: torch.Tensor,
        relations: torch.Tensor,
        tails: torch.Tensor,
        relation_indices: torch.Tensor,
        problem_with_submission_cnt: dict[int, int],
    ) -> None:
        """
        Update the attention matrix.

        Parameters
        ----------
        heads: torch.Tensor
            The head entities.
        relations: torch.Tensor
            The relations.
        tails: torch.Tensor
            The tail entities.
        relation_indices: torch.Tensor
            The relation indices.
        problem_with_submission_cnt: dict[int, int]
            The problem with submission count.
        """
        device = self.attentive_matrix.device
        rows, cols, attentions = [], [], []

        for relation_idx in relation_indices:
            matched_indices = torch.where(relations == relation_idx)
            batch_heads = heads[matched_indices]
            batch_tails = tails[matched_indices]
            batch_attention = self._update_attention_by_batch(
                heads=batch_heads,
                tails=batch_tails,
                relation_idx=relation_idx,
                problem_with_submission_cnt=problem_with_submission_cnt,
            )
            rows.append(batch_heads)
            cols.append(batch_tails)
            attentions.append(batch_attention)

        concatenated_rows = torch.cat(tensors=rows)
        concatenated_cols = torch.cat(tensors=cols)
        concatenated_attentions = torch.cat(tensors=attentions)

        attentive_matrix: torch.Tensor = torch.sparse_coo_tensor(
            indices=torch.stack(tensors=[concatenated_rows, concatenated_cols]),
            values=concatenated_attentions,
            size=torch.Size(self.attentive_matrix.shape),
        )
        attentive_matrix = torch.sparse.softmax(input=attentive_matrix.cpu(), dim=1)

        self.attentive_matrix.data = attentive_matrix.to(device)

    def _calc_score(
        self,
        user_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate the score.

        Parameters
        ----------
        user_ids: torch.Tensor
            The user IDs.
        item_ids: torch.Tensor
            The item IDs.

        Returns
        -------
        score: torch.Tensor
            The score.
        """
        all_embeddings = self._build_cf_embeddings()  # (user_num + entity_num, concatentated_dim)
        user_embeddings = all_embeddings[user_ids]  # (user_num, concatentated_dim)
        item_embeddings = all_embeddings[item_ids]  # (item_num, concatentated_dim)
        return torch.matmul(input=user_embeddings, other=item_embeddings.transpose(0, 1))

    def build_optimizer(self, lr: float) -> None:
        """
        Build the optimizer.

        Parameters
        ----------
        lr: float
            The learning rate.
        """
        self._cf_optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)
        self._kg_optimizer = torch.optim.Adam(params=self.parameters(), lr=lr)

    def update_cf_weights(self) -> None:
        """
        Update the weights for collaborative filtering.
        """
        self._cf_optimizer.step()
        self._cf_optimizer.zero_grad()

    def update_kg_weights(self) -> None:
        """
        Update the weights for knowledge graph.
        """
        self._kg_optimizer.step()
        self._kg_optimizer.zero_grad()

    def forward(self, *args: Any, mode: KGATMode) -> torch.Tensor | None:  # noqa: ANN401
        match mode:
            case KGATMode.TRAIN_CF:
                return self._calc_cf_loss(*args)
            case KGATMode.TRAIN_KG:
                return self._calc_kg_loss(*args)
            case KGATMode.UPDATE_ATTENTION:
                self._update_attention(*args)
                return None
            case KGATMode.PREDICT:
                return self._calc_score(*args)
