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
from typing import Literal

import numpy as np
import torch
from scipy import sparse as sp
from tqdm import tqdm

from src.constants import SEED
from src.model.NFM.model import NFM, NFMArgs, NFMMode
from src.model.NFM.preprocess import Preprocess
from src.utils.data_loader import DataLoader
from src.utils.figure_drawer import plot_loss, plot_metrics
from src.utils.metrics_calculator import Metrics, metrics_at_k

warnings.filterwarnings("ignore", category=RuntimeWarning)

basicConfig(level=INFO)
logger = getLogger(__name__)

EPOCH_NUM = 300
STOP_STEPS = 50
TRAIN_BATCH_SIZE = 64
TEST_BATCH_SIZE = 16
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
    model: NFM,
    preprocess: Preprocess,
    train_interaction_dict: dict[int, list[int]],
    eval_interaction_dict: dict[int, list[int]],
    device: torch.device,
) -> tuple[list[torch.Tensor], dict[int, dict[Metrics, float]]]:
    model.eval()

    user_ids = list(eval_interaction_dict.keys())
    user_ids_batch_list = [
        torch.LongTensor(user_ids[i : i + TEST_BATCH_SIZE]) for i in range(0, len(user_ids), TEST_BATCH_SIZE)
    ]
    user_num = len(user_ids)
    user_map = dict(zip(user_ids, range(user_num), strict=False))
    item_ids = list(range(preprocess.item_num))
    item_num = len(item_ids)

    cf_users = []
    cf_items = []
    cf_scores = []
    metrics_dict: dict[int, dict[Metrics, list[float]]] = {
        k: {metrics: [] for metrics in Metrics} for k in METRICS_K_LIST
    }

    with tqdm(total=len(user_ids_batch_list), desc="Evaluating") as bar:
        for user_ids_batch in user_ids_batch_list:
            feature_values = preprocess.generate_test_batch(batch_user_ids=user_ids_batch).to(device)

            with torch.no_grad():
                batch_scores: torch.Tensor = model(
                    feature_values,
                    mode=NFMMode.PREDICT,
                )

            cf_users.extend(np.repeat(user_ids_batch, item_num).tolist())
            cf_items.extend(item_ids * len(user_ids_batch))
            cf_scores.append(batch_scores.cpu())
            bar.update(1)

    rows = [user_map[idx] for idx in cf_users]
    cols = cf_items
    scores = torch.cat(cf_scores)
    cf_score_matrix = torch.Tensor(sp.coo_matrix((scores, (rows, cols)), shape=(user_num, item_num)).todense())

    batch_metrics = metrics_at_k(
        cf_scores=cf_score_matrix,
        train_interaction_dict=train_interaction_dict,
        test_interaction_dict=eval_interaction_dict,
        user_ids=np.array(user_ids),
        item_ids=np.array(item_ids),
        k_list=METRICS_K_LIST,
    )

    for k in METRICS_K_LIST:
        for metrics in Metrics:
            metrics_dict[k][metrics].append(batch_metrics[k][metrics].tolist())

    metrics_result_dict: dict[int, dict[Metrics, float]] = {}
    for k in METRICS_K_LIST:
        metrics_result_dict[k] = {metrics: 0.0 for metrics in Metrics}
        for metrics in Metrics:
            metrics_result_dict[k][metrics] = float(np.concatenate(metrics_dict[k][metrics]).mean())

    return cf_score_matrix.numpy(), metrics_result_dict


def evaluate_on_dataset(
    model: NFM,
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


def save_model(model: NFM, save_dir: str) -> None:
    """
    Save the model.

    Parameters
    ----------
    model: NFM
        The model.
    save_dir: str
        The save directory.
    """
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    torch.save({"model_state_dict": model.state_dict()}, f"{save_dir}/nfm.pth")


def load_model(model: NFM, load_dir: str) -> NFM:
    """
    Load the model.

    Parameters
    ----------
    model: NFM
        The model.
    load_dir: str
        The load directory.

    Returns
    -------
    model: NFM
        The loaded model
    """
    checkpoint = torch.load(f"{load_dir}/nfm.pth", map_location=torch.device("cpu"))
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
        train_batch_size=TRAIN_BATCH_SIZE,
        test_batch_size=TEST_BATCH_SIZE,
    )
    preprocess.run(dataset_name="training")
    logger.info("Preprocessed!\n====================================")

    model_args = NFMArgs(
        user_num=preprocess.user_num,
        item_num=preprocess.item_num,
        entity_num=preprocess.entity_num,
    )
    model = NFM(args=model_args)
    model.to(device)
    logger.info("Built model!\n%s\n====================================", model)

    # Build optimizer
    model.build_optimizer(lr=LEARNING_RATE)

    # Initialize metrics
    best_epoch = 0
    k_min = min(METRICS_K_LIST)
    train_losses: list[float] = []
    train_precisions: dict[int, list[float]] = defaultdict(list)
    train_recalls: dict[int, list[float]] = defaultdict(list)
    train_ndcgs: dict[int, list[float]] = defaultdict(list)
    validation_precisions: dict[int, list[float]] = defaultdict(list)
    validation_recalls: dict[int, list[float]] = defaultdict(list)
    validation_ndcgs: dict[int, list[float]] = defaultdict(list)

    # Trainig
    logger.info("Training...")
    for epoch_idx in range(1, EPOCH_NUM + 1):
        logger.info("--------------- Epoch: %d ---------------", epoch_idx)
        model.train()

        train_loss = 0.0
        batch_num = len(preprocess.interaction_matrix[0]) // TRAIN_BATCH_SIZE + 1

        with tqdm(initial=1, total=batch_num + 1, desc="Training") as bar:
            for _ in range(1, batch_num + 1):
                positive_feature_values, negative_feature_values = preprocess.generate_train_batch()
                positive_feature_values = positive_feature_values.to(device)
                negative_feature_values = negative_feature_values.to(device)

                batch_loss = model(positive_feature_values, negative_feature_values, mode=NFMMode.TRAIN)
                batch_loss.backward()
                model.update_weights()
                train_loss += batch_loss.item()

                bar.update(1)

        train_losses.append(train_loss / batch_num)
        logger.info("[training] Epoch: %d, Loss: %.4f\n", epoch_idx, train_loss / batch_num)

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
        losses=train_losses,
        loss_type="cf",
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sm", help="for using small dataset", action="store_true")
    parser.add_argument("--predict", help="for prediction", action="store_true")
    args = parser.parse_args()
    # if args.predict:
    #     predict(args=args)
    # else:
    train(args=args)
