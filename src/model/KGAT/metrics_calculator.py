from enum import StrEnum

import numpy as np
import torch
from numpy.typing import NDArray


class Metrics(StrEnum):
    PRECISION = "precision"
    RECALL = "recall"
    NDCG = "ndcg"


MetricsDict = dict[int, dict[Metrics, NDArray[np.float64]]]


def precision_at_k_batch(hits: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    """
    Calculate precision@k for a batch of hits.

    Parameters
    ----------
    hits: NDArray[np.float64]
        The hits.
    k: int
        The k.

    Returns
    -------
    score: NDArray[np.float64]
        The precision@k.
    """
    total: NDArray[np.float64] = hits[:, :k].sum(axis=1)
    return total / k


def recall_at_k_batch(hits: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    """
    Calculate recall@k for a batch of hits.

    Parameters
    ----------e
    hits: NDArray[np.float64]
        The hits.
    k: int
        The k.

    Returns
    -------
    score: NDArray[np.float64]
        The recall@k.
    """
    tp: NDArray[np.float64] = hits[:, :k].sum(axis=1)
    tp_with_fn: NDArray[np.float64] = hits.sum(axis=1)
    return tp / tp_with_fn


def ndcg_at_k_batch(hits: NDArray[np.float64], k: int) -> NDArray[np.float64]:
    """
    Calculate NDCG@k for a batch of hits.

    Parameters
    ----------
    hits: NDArray[np.float64]
        The hits.
    k: int
        The k.

    Returns
    -------
    score: NDArray[np.float64]
        The NDCG@k.
    """
    hits_k = hits[:, :k]
    dcg: NDArray[np.float64] = np.sum((2**hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    sorted_hits_k = np.flip(np.sort(hits), axis=1)[:, :k]
    idcg: NDArray[np.float64] = np.sum((2**sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    idcg[idcg == 0] = 1
    return dcg / idcg


def metrics_at_k(
    cf_scores: torch.Tensor,
    train_interaction_dict: dict[int, list[int]],
    test_interaction_dict: dict[int, list[int]],
    user_ids: NDArray[np.int64],
    item_ids: NDArray[np.int64],
    k_list: list[int],
) -> MetricsDict:
    """
    Calculate metrics@k.

    Parameters:
    cf_scores: torch.Tensor
        The CF scores.
    train_interaction_dict: dict[int, list[int]]
        The train user interaction dictionary.
    test_interaction_dict: dict[int, list[int]]
        The test user interaction dictionary.
    user_ids: np.ndarray[int]
        The user IDs.
    item_ids: np.ndarray[int]
        The item IDs.
    k_list: list[int]
        The k list.

    Returns
    -------
    metrics_dict: MetricsDict
        The metrics.
    """
    positive_items_binary = np.zeros([len(user_ids), len(item_ids)], dtype=np.float64)
    for idx, user_id in enumerate(user_ids):
        train_positive_items = train_interaction_dict[user_id]
        test_positive_items = test_interaction_dict[user_id]
        cf_scores[idx][train_positive_items] = -np.inf
        positive_items_binary[idx][test_positive_items] = 1

    _, rank_indices = torch.sort(cf_scores, descending=True)
    hits = np.array([positive_items_binary[idx][rank_indices[idx]] for idx in range(len(user_ids))], dtype=np.float64)

    metrics_dict: MetricsDict = {}
    for k in k_list:
        metrics_dict[k] = {}
        metrics_dict[k][Metrics.PRECISION] = precision_at_k_batch(hits, k)
        metrics_dict[k][Metrics.RECALL] = recall_at_k_batch(hits, k)
        metrics_dict[k][Metrics.NDCG] = ndcg_at_k_batch(hits, k)

    return metrics_dict
