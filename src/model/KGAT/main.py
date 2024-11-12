import sys
from pathlib import Path

# for Google Colab
if "google.colab" in sys.modules:
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))

import argparse
import warnings
from argparse import Namespace
from collections import Counter, defaultdict
from logging import INFO, basicConfig, getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from tqdm import tqdm

from src.constants import SEED
from src.model.KGAT import kg_visualizer
from src.model.KGAT.model import (
    KGAT,
    KGATArgs,
    KGATMode,
)
from src.model.KGAT.preprocess import Preprocess
from src.utils import bar_graph_visualizer
from src.utils.data_loader import DataLoader
from src.utils.figure_drawer import plot_loss, plot_metrics
from src.utils.metrics_calculator import Metrics, metrics_at_k

if TYPE_CHECKING:
    from src.type import Entity, Problem, User

# from src.model.KGAT.weights_visualizer import visualize_attention_scores

warnings.filterwarnings("ignore", category=RuntimeWarning)

basicConfig(level=INFO)
logger = getLogger(__name__)

EPOCH_NUM = 500
STOP_STEPS = 10
TRAIN_CF_BATCH_SIZE = 256
TRAIN_KG_BATCH_SIZE = 512
TEST_BATCH_SIZE = 256
EDGE_DROPOUT = [0.1, 0.1, 0.1]
NODE_DROPOUT = [0.1]
LEARNING_RATE = 0.0001
METRICS_K_LIST = [20, 40, 60, 80, 100]


def set_seed() -> None:
    """
    Set seed for reproducibility.

    Parameters
    ----------
    seed: Optional[int]
        The seed.
    """
    np.random.default_rng(seed=SEED)
    torch.manual_seed(seed=SEED)
    torch.cuda.manual_seed_all(seed=SEED)


def evaluate(
    model: KGAT,
    preprocess: Preprocess,
    train_interaction_dict: dict[int, list[int]],
    eval_interaction_dict: dict[int, list[int]],
    device: torch.device,
) -> tuple[torch.Tensor, dict[int, dict[Metrics, float]]]:
    model.eval()

    test_user_ids = list(eval_interaction_dict.keys())
    test_user_ids_batch_list = [
        torch.LongTensor(test_user_ids[i : i + TEST_BATCH_SIZE]) for i in range(0, len(test_user_ids), TEST_BATCH_SIZE)
    ]

    # Convert item ids to indices
    train_interaction_dict = dict(train_interaction_dict.items())
    eval_interaction_dict = dict(eval_interaction_dict.items())

    item_num = preprocess.item_num
    item_ids = torch.arange(item_num, dtype=torch.long).to(device)

    cf_scores = []
    metrics_dict: dict[int, dict[Metrics, list[float]]] = {
        k: {metrics: [] for metrics in Metrics} for k in METRICS_K_LIST
    }

    with tqdm(total=len(test_user_ids_batch_list), desc="Evaluating") as bar:
        for test_user_ids_batch in test_user_ids_batch_list:
            user_ids = test_user_ids_batch.to(device)
            with torch.no_grad():
                batch_scores: torch.Tensor = model(
                    test_user_ids_batch,
                    item_ids,
                    mode=KGATMode.PREDICT,
                )

            batch_scores = batch_scores.cpu()
            batch_metrics = metrics_at_k(
                cf_scores=batch_scores,
                train_interaction_dict=train_interaction_dict,
                test_interaction_dict=eval_interaction_dict,
                user_ids=user_ids.cpu().numpy(),
                item_ids=item_ids.cpu().numpy(),
                k_list=METRICS_K_LIST,
            )
            cf_scores.append(batch_scores.cpu())

            for k in METRICS_K_LIST:
                for metrics in Metrics:
                    metrics_dict[k][metrics].append(batch_metrics[k][metrics].tolist())

            bar.update(1)

    metrics_result_dict: dict[int, dict[Metrics, float]] = {}
    scores = torch.cat(cf_scores)
    for k in METRICS_K_LIST:
        metrics_result_dict[k] = {metrics: 0.0 for metrics in Metrics}
        for metrics in Metrics:
            metrics_result_dict[k][metrics] = float(np.concatenate(metrics_dict[k][metrics]).mean())

    return scores, metrics_result_dict


def evaluate_on_dataset(
    model: KGAT,
    device: torch.device,
    preprocess: Preprocess,
    train_interaction_dict: dict[int, list[int]],
    eval_interaction_dict: dict[int, list[int]],
    dataset_name: Literal["training", "test", "validation"],
    epoch_idx: int | None = None,
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    logger.info("Evaluating model on %s dataset...", dataset_name)
    _, metrics_dict = evaluate(
        model=model,
        preprocess=preprocess,
        train_interaction_dict=train_interaction_dict,
        eval_interaction_dict=eval_interaction_dict,
        device=device,
    )

    precision: dict[int, float] = {}
    recall: dict[int, float] = {}
    ndcg: dict[int, float] = {}

    for k in METRICS_K_LIST:
        precision[k] = metrics_dict[k][Metrics.PRECISION]
        recall[k] = metrics_dict[k][Metrics.RECALL]
        ndcg[k] = metrics_dict[k][Metrics.NDCG]

    if epoch_idx is not None:
        metrics_info = f"[{dataset_name}] Epoch: {epoch_idx}, "
        for k in METRICS_K_LIST:
            metrics_result_k = (
                f"Precision@{k}: {precision[k]:.4f}, " f"Recall@{k}: {recall[k]:.4f}, " f"nDCG@{k}: {ndcg[k]:.4f} "
            )
            metrics_info += metrics_result_k
        metrics_info += "\n"
        logger.info(metrics_info)

    return precision, recall, ndcg


def early_stopping(recalls: list[float]) -> tuple[float, bool]:
    """
    Early stopping.

    Parameters
    ----------
    recalls: list[float]
        The recalls.

    Returns
    -------
    best_recall: float
        The best recall.
    stop_flag: bool
        The flag for stopping.
    """
    best_recall = max(recalls)
    best_step = recalls.index(best_recall)
    stop_flag = False
    if len(recalls) - best_step - 1 >= STOP_STEPS:
        stop_flag = True
    return best_recall, stop_flag


def save_model(model: KGAT, save_dir: str) -> None:
    """
    Save the model.

    Parameters
    ----------
    model: KGAT
        The model.
    save_dir: str
        The save directory.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, f"{save_dir}/kgat.pth")


def load_model(model: KGAT, load_dir: str) -> KGAT:
    """
    Load the model.

    Parameters
    ----------
    model: KGAT
        The model.
    load_dir: str
        The load directory.

    Returns
    -------
    model: KGAT
        The loaded model
    """
    checkpoint = torch.load(f"{load_dir}/kgat.pth", map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def train(args: Namespace) -> None:
    # Set seed
    set_seed()

    # Set device (GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load dataset
    logger.info("Loading dataset...")
    dataset_loader = DataLoader(dataset_dir=Path("../dataset"))
    dataset = dataset_loader.load_dataset(args=args)
    logger.info("Dataset loaded!\n====================================")

    # Preprocess
    logger.info("Preprocessing...")
    preprocess = Preprocess(
        args=args,
        dataset=dataset,
        cf_batch_size=TRAIN_CF_BATCH_SIZE,
        kg_batch_size=TRAIN_KG_BATCH_SIZE,
        device=device,
    )
    preprocess.run(dataset_name="training")
    logger.info("Preprocessed!\n====================================")

    # Build model
    logger.info("Building model...")
    model_args = KGATArgs(
        user_num=preprocess.user_num,
        entity_num=preprocess.entity_num,
        relation_num=len(preprocess.adjacency_relations),
        attentive_matrix=preprocess.attentive_matrix,
    )
    model = KGAT(args=model_args)
    model.to(device)
    logger.info("Built model!\n%s\n====================================", model)

    # Build optimizer
    model.build_optimizer(lr=LEARNING_RATE)

    # Initialize metrics
    best_epoch = 0
    k_min = min(METRICS_K_LIST)
    train_cf_losses: list[float] = []
    train_kg_losses: list[float] = []
    train_precisions: dict[int, list[float]] = defaultdict(list)
    train_recalls: dict[int, list[float]] = defaultdict(list)
    train_ndcgs: dict[int, list[float]] = defaultdict(list)
    validation_precisions: dict[int, list[float]] = defaultdict(list)
    validation_recalls: dict[int, list[float]] = defaultdict(list)
    validation_ndcgs: dict[int, list[float]] = defaultdict(list)

    # Training
    logger.info("Start training...")

    problem_with_submission_cnt = dict(Counter([row[1] for row in preprocess.interaction_matrix]))
    for problem_id in range(preprocess.item_num):
        if problem_id not in problem_with_submission_cnt:
            problem_with_submission_cnt[problem_id] = 0

    for epoch_idx in range(1, EPOCH_NUM + 1):
        logger.info("--------------- Epoch: %d ---------------", epoch_idx)
        model.train()

        # Training for collaborative filtering
        logger.info("Training for collaborative filtering...")
        train_cf_loss = 0.0
        cf_batch_num = len(preprocess.interaction_matrix) // TRAIN_CF_BATCH_SIZE + 1
        with tqdm(initial=1, total=cf_batch_num + 1, desc="CF Training") as bar:
            for _ in range(1, cf_batch_num + 1):
                cf_batch_user_ids, cf_batch_positive_problems, cf_batch_negative_problems = (
                    preprocess.generate_cf_batch()
                )
                cf_batch_user_ids = cf_batch_user_ids.to(device)
                cf_batch_positive_problems = cf_batch_positive_problems.to(device)
                cf_batch_negative_problems = cf_batch_negative_problems.to(device)
                cf_batch_loss: torch.Tensor = model(
                    cf_batch_user_ids,
                    cf_batch_positive_problems,
                    cf_batch_negative_problems,
                    mode=KGATMode.TRAIN_CF,
                )
                cf_batch_loss.backward()
                model.update_cf_weights()
                train_cf_loss += cf_batch_loss.item()

                bar.update(1)

        train_cf_losses.append(train_cf_loss / cf_batch_num)
        logger.info("[training] Epoch: %d, CF Loss: %.4f\n", epoch_idx, train_cf_loss / cf_batch_num)

        # Training for knowledge graph
        logger.info("Training for knowledge graph...")
        train_kg_loss = 0.0
        kg_batch_num = len(preprocess.all_heads) // TRAIN_KG_BATCH_SIZE + 1
        with tqdm(initial=1, total=kg_batch_num + 1, desc="KG Training") as bar:
            for _ in range(1, kg_batch_num + 1):
                kg_batch_heads, kg_batch_relations, kg_batch_positive_tails, kg_batch_negative_tails = (
                    preprocess.generate_kg_batch()
                )
                heads = torch.tensor(kg_batch_heads).to(device)
                relations = torch.tensor(kg_batch_relations).to(device)
                positive_tails = torch.tensor(kg_batch_positive_tails).to(device)
                negative_tails = torch.tensor(kg_batch_negative_tails).to(device)
                kg_batch_loss: torch.Tensor = model(
                    heads,
                    relations,
                    positive_tails,
                    negative_tails,
                    mode=KGATMode.TRAIN_KG,
                )
                kg_batch_loss.backward()
                model.update_kg_weights()
                train_kg_loss += kg_batch_loss.item()

                bar.update(1)

        train_kg_losses.append(train_kg_loss / kg_batch_num)
        logger.info("[training] Epoch: %d, KG Loss: %.4f\n", epoch_idx, train_kg_loss / kg_batch_num)

        # Update attention
        heads = torch.tensor(preprocess.all_heads).to(device)
        relations = torch.tensor(preprocess.all_relation_indices).to(device)
        tails = torch.tensor(preprocess.all_tails).to(device)
        relation_indices = torch.tensor(preprocess.adjacency_relations).to(device)
        model(
            heads,
            relations,
            tails,
            relation_indices,
            mode=KGATMode.UPDATE_ATTENTION,
        )

        # Evaluate on test dataset
        train_precision, train_recall, train_ndcg = evaluate_on_dataset(
            model=model,
            device=device,
            preprocess=preprocess,
            train_interaction_dict=preprocess.train_interaction_dict,
            eval_interaction_dict=preprocess.test_interaction_dict,
            dataset_name="training",
            epoch_idx=epoch_idx,
        )
        for k in METRICS_K_LIST:
            train_precisions[k].append(train_precision[k])
            train_recalls[k].append(train_recall[k])
            train_ndcgs[k].append(train_ndcg[k])

        # Evaluate on validation dataset
        validation_precision, validation_recall, validation_ndcg = evaluate_on_dataset(
            model=model,
            device=device,
            preprocess=preprocess,
            train_interaction_dict=preprocess.train_interaction_dict,
            eval_interaction_dict=preprocess.validation_interaction_dict,
            dataset_name="validation",
            epoch_idx=epoch_idx,
        )
        for k in METRICS_K_LIST:
            validation_precisions[k].append(validation_precision[k])
            validation_recalls[k].append(validation_recall[k])
            validation_ndcgs[k].append(validation_ndcg[k])

        _, stop_flag = early_stopping(validation_recalls[k_min])

        if stop_flag:
            best_epoch = epoch_idx
            logger.info("Early stopping!")
            break

    # Save model
    save_model(
        model=model,
        save_dir="./result/model",
    )

    def save_metrics(dataset_name: Literal["training", "validation"]) -> None:
        Path("./result/metrics").mkdir(parents=True, exist_ok=True)
        with Path(f"./result/metrics/{dataset_name}-metrics.txt").open(mode="w") as f:
            f.write(f"{dataset_name.capitalize()} Metrics\n")
            f.write("Precisions\n")
            f.write(str(dict(train_precisions) if dataset_name == "training" else dict(validation_precisions)) + "\n")
            f.write("Recalls\n")
            f.write(str(dict(train_recalls) if dataset_name == "training" else dict(validation_recalls)) + "\n")
            f.write("nDCGs\n")
            f.write(str(dict(train_ndcgs) if dataset_name == "training" else dict(validation_ndcgs)) + "\n")

        # Output best metrics
        for k in METRICS_K_LIST:
            logger.info(
                "[%s] Best Precision@%d: %.4f",
                dataset_name,
                k,
                max(train_precisions[k]) if dataset_name == "training" else max(validation_precisions[k]),
            )
            logger.info(
                "[%s] Best Recall@%d: %.4f",
                dataset_name,
                k,
                max(train_recalls[k]) if dataset_name == "training" else max(validation_recalls[k]),
            )
            logger.info(
                "[%s] Best nDCG@%d: %.4f",
                dataset_name,
                k,
                max(train_ndcgs[k]) if dataset_name == "training" else max(validation_ndcgs[k]),
            )

    # Save metrics
    save_metrics(dataset_name="training")
    save_metrics(dataset_name="validation")

    # Plot losses and metrics
    plot_loss(
        epoch_num=best_epoch,
        losses=train_cf_losses,
        loss_type="cf",
    )
    plot_loss(
        epoch_num=best_epoch,
        losses=train_kg_losses,
        loss_type="kg",
    )
    plot_metrics(
        dataset_name="training",
        epoch_num=best_epoch,
        metrics=train_precisions,
        metrics_name="precision",
    )
    plot_metrics(
        dataset_name="training",
        epoch_num=best_epoch,
        metrics=train_recalls,
        metrics_name="recall",
    )
    plot_metrics(
        dataset_name="training",
        epoch_num=best_epoch,
        metrics=train_ndcgs,
        metrics_name="ndcg",
    )
    plot_metrics(
        dataset_name="validation",
        epoch_num=best_epoch,
        metrics=validation_precisions,
        metrics_name="precision",
    )
    plot_metrics(
        dataset_name="validation",
        epoch_num=best_epoch,
        metrics=validation_recalls,
        metrics_name="recall",
    )
    plot_metrics(
        dataset_name="validation",
        epoch_num=best_epoch,
        metrics=validation_ndcgs,
        metrics_name="ndcg",
    )


def predict(args: Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load dataset
    logger.info("Loading dataset...")
    dataset_loader = DataLoader(dataset_dir=Path("../dataset"))
    dataset = dataset_loader.load_dataset(args=args)
    logger.info("Dataset loaded!\n====================================")

    # Preprocess
    logger.info("Preprocessing...")
    preprocess = Preprocess(
        args=args,
        dataset=dataset,
        cf_batch_size=TRAIN_CF_BATCH_SIZE,
        kg_batch_size=TRAIN_KG_BATCH_SIZE,
        device=device,
    )
    preprocess.run(dataset_name="test")
    logger.info("Preprocessed!\n====================================")

    # Build model
    logger.info("Building model...")
    model_args = KGATArgs(
        user_num=preprocess.user_num,
        entity_num=preprocess.entity_num,
        relation_num=len(preprocess.adjacency_relations),
    )
    model = load_model(model=KGAT(args=model_args), load_dir="./result/model")
    model.to(device)

    # Predict
    logger.info("Predicting recommendations...")
    model.eval()

    test_precisions, test_recalls, test_ndcgs = evaluate_on_dataset(
        model=model,
        device=device,
        preprocess=preprocess,
        train_interaction_dict=preprocess.train_interaction_dict,
        eval_interaction_dict=preprocess.test_interaction_dict,
        dataset_name="test",
    )

    logger.info("Precision: %s", test_precisions)
    logger.info("Recall: %s", test_recalls)
    logger.info("nDCG: %s", test_ndcgs)


def recommend(args: Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load dataset
    logger.info("Loading dataset...")
    dataset_loader = DataLoader(dataset_dir=Path("../dataset"))
    dataset = dataset_loader.load_dataset(args=args)
    logger.info("Dataset loaded!\n====================================")

    # Preprocess
    logger.info("Preprocessing...")
    preprocess = Preprocess(
        args=args,
        dataset=dataset,
        cf_batch_size=TRAIN_CF_BATCH_SIZE,
        kg_batch_size=TRAIN_KG_BATCH_SIZE,
        device=device,
    )
    preprocess.run(dataset_name="test")
    logger.info("Preprocessed!\n====================================")

    # Build model
    logger.info("Building model...")
    model_args = KGATArgs(
        user_num=preprocess.user_num,
        entity_num=preprocess.entity_num,
        relation_num=len(preprocess.adjacency_relations),
    )
    model = load_model(model=KGAT(args=model_args), load_dir="./result/model")
    model.to(device)

    user_ids = torch.arange(preprocess.user_num, dtype=torch.long)
    item_ids = torch.arange(preprocess.item_num, dtype=torch.long).to(device)

    user_ids_batch_list = [
        torch.LongTensor(user_ids[i : i + TEST_BATCH_SIZE]) for i in range(0, len(user_ids), TEST_BATCH_SIZE)
    ]
    cf_scores = []

    for batch_user_ids in user_ids_batch_list:
        with torch.no_grad():
            batch_scores: torch.Tensor = model(
                batch_user_ids.to(device),
                item_ids,
                mode=KGATMode.PREDICT,
            )
        cf_scores.append(batch_scores.cpu())

    scores = torch.cat(cf_scores)

    train_problem_mask = torch.zeros_like(scores)
    for user_idx in range(preprocess.user_num):
        solved_problems = preprocess.interaction_dict[user_idx]
        train_problems = preprocess.train_interaction_dict[user_idx]
        mask = list(set(solved_problems + train_problems))
        if mask:
            train_problem_mask[user_idx, mask] = float("-inf")

    masked_scores = scores + train_problem_mask

    # Recommend top-k problems for each user
    k = 20
    _, all_top_k_problem_indices = torch.topk(masked_scores, k=k, dim=1)

    user_idx_with_recommended_problems: dict[int, list[Problem]] = defaultdict(list)
    problem_cnt_dict: dict[int, int] = defaultdict(int)
    for user_idx in range(preprocess.user_num):
        user = preprocess.user_id_map[user_idx]
        recommended_problems = [
            preprocess.problem_id_map[int(problem_id)] for problem_id in all_top_k_problem_indices[user_idx]
        ]
        # logger.info("Recommendations for user: %s", user.handle)
        for i, problem in enumerate(recommended_problems):
            user_idx_with_recommended_problems[user_idx].append(problem)
            problem_cnt_dict[problem.id] += 1
            # logger.info("%d. (%d, %s)", i + 1, problem.contest_id, problem.index)
        # logger.info("--------------------")

    for problem_id in range(preprocess.item_num):
        if problem_id not in problem_cnt_dict:
            problem_cnt_dict[problem_id] = 0

    problem_with_recommended_cnt = sorted(problem_cnt_dict.items())
    problem_ids, recommended_cnts = zip(*problem_with_recommended_cnt, strict=False)

    bar_graph_visualizer.visualize(
        x=list(problem_ids),
        y=list(recommended_cnts),
        title="Recommended Count for Each Problem",
        x_label="Problem ID",
        y_label="Recommended Count",
        x_interval=1000,
        y_interval=50,
        ticks="both",
    )


def visualize_popularity(args: Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load dataset
    logger.info("Loading dataset...")
    dataset_loader = DataLoader(dataset_dir=Path("../dataset"))
    dataset = dataset_loader.load_dataset(args=args)
    logger.info("Dataset loaded!\n====================================")

    # Preprocess
    logger.info("Preprocessing...")
    preprocess = Preprocess(
        args=args,
        dataset=dataset,
        cf_batch_size=TRAIN_CF_BATCH_SIZE,
        kg_batch_size=TRAIN_KG_BATCH_SIZE,
        device=device,
    )

    dataset_name = args.visualize_popularity
    if dataset_name is None:
        msg = "dataset_name must be provided."
        raise ValueError(msg)

    preprocess.run(dataset_name=dataset_name)
    logger.info("Preprocessed!\n====================================")

    # Visualize problem with submission count
    problem_with_submission_cnt = dict(Counter([row[1] for row in preprocess.interaction_matrix]))
    for problem_id in range(preprocess.item_num):
        if problem_id not in problem_with_submission_cnt:
            problem_with_submission_cnt[problem_id] = 0

    sorted_problem_with_submission_cnt = sorted(problem_with_submission_cnt.items())
    problem_ids, submission_cnts = zip(*sorted_problem_with_submission_cnt, strict=False)

    popularities = sorted(submission_cnts, reverse=True)

    bar_graph_visualizer.visualize(
        x=list(problem_ids),
        y=list(popularities),
        title="Submission Count for Each Problem",
        x_label="Problem ID",
        y_label="Submission Count",
        y_interval=25,
        ticks="y",
    )


def visualize_kg(args: Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load dataset
    logger.info("Loading dataset...")
    dataset_loader = DataLoader(dataset_dir=Path("../dataset"))
    dataset = dataset_loader.load_dataset(args=args)
    logger.info("Dataset loaded!\n====================================")

    # Preprocess
    logger.info("Preprocessing...")
    preprocess = Preprocess(
        args=args,
        dataset=dataset,
        cf_batch_size=TRAIN_CF_BATCH_SIZE,
        kg_batch_size=TRAIN_KG_BATCH_SIZE,
        device=device,
    )
    preprocess.run(dataset_name="training")
    logger.info("Preprocessed!\n====================================")

    triplets = preprocess.triplets
    entities = preprocess.entities

    # Visualize knowledge graph
    kg_visualizer.visualize(
        triplets=triplets,
        entities=entities,
        highlight_nodes=[],
    )


def visualize_attention(args: Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load dataset
    logger.info("Loading dataset...")
    dataset_loader = DataLoader(dataset_dir=Path("../dataset"))
    dataset = dataset_loader.load_dataset(args=args)
    logger.info("Dataset loaded!\n====================================")

    # Preprocess
    logger.info("Preprocessing...")
    preprocess = Preprocess(
        args=args,
        dataset=dataset,
        cf_batch_size=TRAIN_CF_BATCH_SIZE,
        kg_batch_size=TRAIN_KG_BATCH_SIZE,
        device=device,
    )
    preprocess.run(dataset_name="training")
    logger.info("Preprocessed!\n====================================")

    # Build model
    logger.info("Building model...")
    model_args = KGATArgs(
        user_num=preprocess.user_num,
        entity_num=preprocess.entity_num,
        relation_num=len(preprocess.adjacency_relations),
    )
    model = load_model(model=KGAT(args=model_args), load_dir="./result/model")
    model.to(device)

    idx_to_entity = dict(enumerate(preprocess.entities))
    attentive_matrix_indices = model.attentive_matrix.indices()  # [head_attentions[], tail_attentions[]]

    heads: list[int] = attentive_matrix_indices[0].tolist()  # len: user_num + entity_num
    tails: list[int] = attentive_matrix_indices[1].tolist()  # len: user_num + entity_num
    attentions: list[float] = model.attentive_matrix.values().tolist()

    result_file = Path("./result/attention_scores.txt")
    if result_file.exists():
        result_file.unlink()

    with result_file.open("a") as f:
        for head, tail, attention in zip(heads, tails, attentions, strict=False):
            head_entity: User | Entity
            tail_entity: User | Entity

            if head < preprocess.user_num:
                head_entity = preprocess.user_id_map[head]
            else:
                head_entity = idx_to_entity[head - preprocess.user_num]

            if tail < preprocess.user_num:
                tail_entity = preprocess.user_id_map[tail]
            else:
                tail_entity = idx_to_entity[tail - preprocess.user_num]

            f.write(f"{head_entity} -> {tail_entity}: {attention:.6f}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sm",
        help="for using small dataset",
        action="store_true",
    )
    parser.add_argument(
        "--predict",
        help="for prediction",
        action="store_true",
    )
    parser.add_argument(
        "--recommend",
        help="for recommendation",
        action="store_true",
    )
    parser.add_argument(
        "--visualize_popularity",
        help="for popularity visualization",
        type=str,
        choices=["training", "test", "validation"],
    )
    parser.add_argument(
        "--visualize_kg",
        help="for knowledge graph visualization",
        action="store_true",
    )
    parser.add_argument(
        "--visualize_attention",
        help="for attention visualization",
        action="store_true",
    )
    args = parser.parse_args()
    if args.predict:
        predict(args=args)
    elif args.recommend:
        recommend(args=args)
    elif args.visualize_popularity:
        visualize_popularity(args=args)
    elif args.visualize_kg:
        visualize_kg(args=args)
    elif args.visualize_attention:
        visualize_attention(args=args)
    else:
        train(args=args)
