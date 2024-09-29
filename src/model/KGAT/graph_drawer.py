import sys

sys.path.append("../..")

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


def plot_loss(
    epoch_num: int,
    losses: list[float],
    loss_type: Literal["cf", "kg"],
) -> None:
    epochs = range(1, epoch_num + 1)
    label = "CF Part" if loss_type == "cf" else "KG Part"
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, losses, label=label)
    plt.title(f"Training Loss ({label})")
    plt.xlabel("Epoch")
    plt.xticks(ticks=np.arange(0, epoch_num + 1, 5))
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(visible=True)
    Path("./result/graph").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"./result/graph/training-{loss_type}-loss.png")
    plt.close()


def plot_metrics(
    dataset_name: Literal["training", "validation"],
    epoch_num: int,
    metrics: dict[int, list[float]],
    metrics_name: Literal["precision", "recall", "ndcg"],
) -> None:
    epochs = range(1, epoch_num + 1)
    label = "nDCG" if metrics_name == "ndcg" else metrics_name.capitalize()
    plt.figure(figsize=(10, 5))
    for k in metrics:
        plt.plot(epochs, metrics[k], label=f"{label}@{k}")
    plt.title(label)
    plt.xlabel("Epoch")
    plt.xticks(ticks=np.arange(0, epoch_num + 1, 5))
    plt.ylabel(label)
    plt.yticks(ticks=np.arange(0, 0.5, 0.05))
    plt.legend()
    plt.grid(visible=True)
    Path("./result/graph").mkdir(parents=True, exist_ok=True)
    plt.savefig(f"./result/graph/{dataset_name}-{metrics_name}.png")
    plt.close()
