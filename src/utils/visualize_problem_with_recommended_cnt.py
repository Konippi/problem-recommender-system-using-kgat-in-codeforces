import matplotlib.pyplot as plt
import numpy as np


def visualize_problem_with_recommended_cnt(problem_ids: list[int], recommended_cnts: list[int]) -> None:
    plt.figure(figsize=(20, 5))
    plt.xticks(ticks=problem_ids[::1000], labels=list(map(str, problem_ids[::1000])))
    plt.xlabel("Problem")
    plt.yticks(ticks=np.arange(0, max(recommended_cnts) + 10, 10))
    plt.ylabel("Recommended Count")
    plt.title("Recommended Count for Each Problem")

    plt.bar(problem_ids, recommended_cnts, width=1.0, color="blue")
    plt.show()
