from typing import Literal

import matplotlib.pyplot as plt
import numpy as np


def visualize(
    x: list[int],
    y: list[int],
    title: str,
    x_label: str = "",
    y_label: str = "",
    x_interval: int = 10,
    y_interval: int = 10,
    ticks: Literal["both", "x", "y", "none"] = "none",
) -> None:
    plt.figure(figsize=(20, 5))
    plt.title(title)

    plt.xlabel(x_label)
    plt.ylabel(y_label)

    match ticks:
        case "both":
            plt.xticks(ticks=np.arange(0, max(x) - 1, x_interval))
            plt.yticks(ticks=np.arange(0, max(y) + y_interval, y_interval))
        case "x":
            plt.xticks(ticks=np.arange(0, max(x) - 1, x_interval))
            plt.yticks([])
        case "y":
            plt.xticks([])
            plt.yticks(ticks=np.arange(0, max(y) + y_interval, y_interval))
        case "none":
            plt.xticks([])
            plt.yticks([])

    plt.bar(x=x, height=y, width=1.0, color="blue")
    plt.tight_layout()
    plt.subplots_adjust(left=0.05)
    plt.show()
