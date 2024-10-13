import matplotlib.pyplot as plt
import numpy as np


def visualize(
    problem_ids: list[int],
    cnts: list[int],
    title: str,
    x_label: str,
    y_label: str,
    x_interval: int = 10,
    y_interval: int = 10,
) -> None:
    plt.figure(figsize=(20, 5))
    plt.xticks(ticks=problem_ids[::x_interval], labels=list(map(str, problem_ids[::x_interval])))
    plt.xlabel(x_label)
    plt.yticks(ticks=np.arange(0, max(cnts), y_interval))
    plt.ylabel(y_label)
    plt.title(title)

    plt.bar(problem_ids, cnts, width=1.0, color="blue")
    plt.show()
