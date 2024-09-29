import sys
from pathlib import Path

# for Google Colab
if "google.colab" in sys.modules:
    project_root = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(project_root))

import argparse
import warnings
from argparse import Namespace
from collections import defaultdict
from logging import INFO, basicConfig, getLogger
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import torch
from tqdm import tqdm

from src.model.KGAT.data_loader import DataLoader
from src.model.KGAT.graph_drawer import plot_loss, plot_metrics
from src.model.KGAT.metrics_calculator import Metrics, metrics_at_k
from src.model.KGAT.model import (
    KGAT,
    KGATArgs,
    KGATMode,
)
from src.model.KGAT.preprocess import Preprocess
from src.model.KGAT.weights_visualizer import visualize_attention_scores

if TYPE_CHECKING:
    from src.model.KGAT.dataset import Problem

# from src.model.KGAT.weights_visualizer import visualize_attention_scores

warnings.filterwarnings("ignore", category=RuntimeWarning)

basicConfig(level=INFO)
logger = getLogger(__name__)

SEED = 2024
EPOCH_NUM = 300
STOP_STEPS = 10
CF_BATCH_SIZE = 256
KG_BATCH_SIZE = 256
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
    train_interaction_dict: dict[int, list[int]],
    eval_interaction_dict: dict[int, list[int]],
    device: torch.device,
) -> tuple[list[float], dict[int, dict[Metrics, float]]]:
    model.eval()

    test_user_ids = list(eval_interaction_dict.keys())
    test_user_ids_batch_list = [
        torch.LongTensor(test_user_ids[i : i + TEST_BATCH_SIZE]) for i in range(0, len(test_user_ids), TEST_BATCH_SIZE)
    ]

    all_items = set()
    for item in train_interaction_dict.values():
        all_items.update(item)
    for item in eval_interaction_dict.values():
        all_items.update(item)
    item_to_idx = {item: idx for idx, item in enumerate(sorted(all_items))}

    # Convert item ids to indices
    train_interaction_dict = {
        user_id: [item_to_idx[item_id] for item_id in item_ids] for user_id, item_ids in train_interaction_dict.items()
    }
    eval_interaction_dict = {
        user_id: [item_to_idx[item_id] for item_id in item_ids] for user_id, item_ids in eval_interaction_dict.items()
    }

    item_num = len(all_items)
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
            cf_scores.append(batch_scores.numpy())

            for k in METRICS_K_LIST:
                for metrics in Metrics:
                    metrics_dict[k][metrics].append(batch_metrics[k][metrics].tolist())

            bar.update(1)

    metrics_result_dict: dict[int, dict[Metrics, float]] = {}
    cf_scores = np.concatenate(cf_scores, axis=0)
    for k in METRICS_K_LIST:
        metrics_result_dict[k] = {metrics: 0.0 for metrics in Metrics}
        for metrics in Metrics:
            metrics_result_dict[k][metrics] = float(np.concatenate(metrics_dict[k][metrics]).mean())

    return cf_scores, metrics_result_dict


def evaluate_on_dataset(
    model: KGAT,
    device: torch.device,
    train_interaction_dict: dict[int, list[int]],
    eval_interaction_dict: dict[int, list[int]],
    dataset_name: Literal["training", "test", "validation"],
    epoch_idx: int | None = None,
) -> tuple[dict[int, float], dict[int, float], dict[int, float]]:
    logger.info("Evaluating model on %s dataset...", dataset_name)
    _, metrics_dict = evaluate(
        model=model,
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
        cf_batch_size=CF_BATCH_SIZE,
        kg_batch_size=KG_BATCH_SIZE,
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
    for epoch_idx in range(1, EPOCH_NUM + 1):
        logger.info("--------------- Epoch: %d ---------------", epoch_idx)
        model.train()

        # Training for collaborative filtering
        logger.info("Training for collaborative filtering...")
        train_cf_loss = 0.0
        cf_batch_num = len(preprocess.interaction_matrix) // CF_BATCH_SIZE + 1
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
        kg_batch_num = len(preprocess.all_heads) // KG_BATCH_SIZE + 1
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
        model(heads, relations, tails, relation_indices, mode=KGATMode.UPDATE_ATTENTION)

        # Evaluate on test dataset
        train_precision, train_recall, train_ndcg = evaluate_on_dataset(
            model=model,
            device=device,
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
        cf_batch_size=CF_BATCH_SIZE,
        kg_batch_size=KG_BATCH_SIZE,
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
        cf_batch_size=CF_BATCH_SIZE,
        kg_batch_size=KG_BATCH_SIZE,
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

    user_ids = torch.arange(preprocess.user_num, dtype=torch.long).to(device)
    item_ids = torch.arange(preprocess.item_num, dtype=torch.long).to(device)

    with torch.no_grad():
        scores: torch.Tensor = model(user_ids, item_ids, mode=KGATMode.PREDICT)

    k = 10
    _, all_top_k_problem_indices = torch.topk(scores, k=k, dim=1)
    top_k_problems = [
        [top_k_problem_idx + 1 for top_k_problem_idx in top_k_problem_indices]
        for top_k_problem_indices in all_top_k_problem_indices.tolist()
    ]

    user_idx_with_recommended_problems: dict[int, list[Problem]] = defaultdict(list)
    for user_idx in range(preprocess.user_num):
        user = preprocess.user_id_map[user_idx + 1]
        recommended_problems = [preprocess.problem_id_map[problem_id] for problem_id in top_k_problems[user_idx]]
        logger.info("Recommendations for user: %s", user.handle)
        for i, problem in enumerate(recommended_problems, 1):
            user_idx_with_recommended_problems[user_idx].append(problem)
            logger.info("%d. (%d, %s)", i, problem.contest_id, problem.index)
        logger.info("--------------------")


def testing(args: Namespace) -> None:
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
        cf_batch_size=CF_BATCH_SIZE,
        kg_batch_size=KG_BATCH_SIZE,
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

    # problem_map = {problem.id: problem for problem in preprocess.problems}
    # tag_map = {tag.id: tag for problem in preprocess.problems for tag in problem.tags}
    # difficulty_map = {
    #     problem.id: problem.rating.value for problem in preprocess.problems if problem.rating is not None
    # }
    # idx_to_entity = dict(enumerate(preprocess.entities))

    # heads: list[int] = model.attentive_matrix.indices()[0].tolist()
    # tails: list[int] = model.attentive_matrix.indices()[1].tolist()
    # attentions: list[float] = model.attentive_matrix.values().tolist()

    # for i in range(len(heads)):
    #     head = idx_to_entity[heads[i]]
    #     tail = idx_to_entity[tails[i]]
    #     attention = attentions[i]
    #     print(head, tail, attention)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sm", help="for using small dataset", action="store_true")
    parser.add_argument("--predict", help="for prediction", action="store_true")
    parser.add_argument("--recommend", help="for recommendation", action="store_true")
    parser.add_argument("--testing", help="for testing", action="store_true")
    args = parser.parse_args()
    if args.predict:
        predict(args=args)
    elif args.recommend:
        recommend(args=args)
    elif args.testing:
        testing(args=args)
    else:
        train(args=args)
