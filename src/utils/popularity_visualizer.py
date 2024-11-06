import matplotlib.pyplot as plt
import numpy as np


def visualize(
    problem_ids: list[int],
    cnts: list[int],
    title: str,
    x_label: str = "",
    y_label: str = "",
    x_interval: int = 10,
    y_interval: int = 10,
) -> None:
    plt.figure(figsize=(20, 5))
    plt.title(title)

    popularities = sorted(cnts, reverse=True)

    plt.xticks([])
    plt.xlabel(x_label)
    plt.yticks(ticks=np.arange(0, max(popularities) + y_interval, y_interval))
    plt.ylabel(y_label)

    plt.bar(x=problem_ids, height=popularities, width=1.0, color="blue")
    plt.tight_layout()
    plt.subplots_adjust(left=0.05)
    plt.show()
