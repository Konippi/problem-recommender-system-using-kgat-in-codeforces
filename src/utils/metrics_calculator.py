from enum import StrEnum

import numpy as np
import torch
from numpy.typing import NDArray


class Metrics(StrEnum):
    PRECISION = "precision"
    RECALL = "recall"
    NDCG = "ndcg"


MetricsDict = dict[int, dict[Metrics, NDArray[np.float32]]]


def _precision_at_k_batch(hits: NDArray[np.float32], k: int) -> NDArray[np.float32]:
    """
    Calculate precision@k for a batch of hits.

    Parameters
    ----------
    hits: NDArray[np.float32]
        The hits.
    k: int
        The k.

    Returns
    -------
    result: NDArray[np.float32]
        The precision@k.
    """
    result: NDArray[np.float32] = hits[:, :k].mean(axis=1)
    return result


def _recall_at_k_batch(hits: NDArray[np.float32], k: int) -> NDArray[np.float32]:
    """
    Calculate recall@k for a batch of hits.

    Parameters
    ----------e
    hits: NDArray[np.float32]
        The hits.
    k: int
        The k.

    Returns
    -------
    score: NDArray[np.float32]
        The recall@k.
    """
    tp: NDArray[np.float32] = hits[:, :k].sum(axis=1)
    tp_with_fn: NDArray[np.float32] = hits.sum(axis=1)
    return tp / tp_with_fn


def _ndcg_at_k_batch(hits: NDArray[np.float32], k: int) -> NDArray[np.float32]:
    """
    Calculate NDCG@k for a batch of hits.

    Parameters
    ----------
    hits: NDArray[np.float32]
        The hits.
    k: int
        The k.

    Returns
    -------
    score: NDArray[np.float32]
        The NDCG@k.
    """
    hits_k = hits[:, :k]
    dcg: NDArray[np.float32] = np.sum((2**hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    sorted_hits_k = np.flip(np.sort(hits), axis=1)[:, :k]
    idcg: NDArray[np.float32] = np.sum((2**sorted_hits_k - 1) / np.log2(np.arange(2, k + 2)), axis=1)

    idcg[idcg == 0] = np.inf
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
    positive_items_binary = np.zeros([len(user_ids), len(item_ids)], dtype=np.float32)
    for idx, user_id in enumerate(user_ids):
        train_positive_items = train_interaction_dict[user_id]
        test_positive_items = test_interaction_dict[user_id]
        cf_scores[idx][train_positive_items] = -np.inf
        positive_items_binary[idx][test_positive_items] = 1

    _, rank_indices = torch.sort(cf_scores, descending=True)
    hits = np.array([positive_items_binary[idx][rank_indices[idx]] for idx in range(len(user_ids))], dtype=np.float32)

    metrics_dict: MetricsDict = {}
    for k in k_list:
        metrics_dict[k] = {}
        metrics_dict[k][Metrics.PRECISION] = _precision_at_k_batch(hits, k)
        metrics_dict[k][Metrics.RECALL] = _recall_at_k_batch(hits, k)
        metrics_dict[k][Metrics.NDCG] = _ndcg_at_k_batch(hits, k)

    return metrics_dict
