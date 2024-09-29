from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils import json_loader

# TARGET_USER_NUM = None
TARGET_USER_NUM: int | None = 10

rng = np.random.default_rng()


def create_problem_maps(problems: list[dict]) -> dict[tuple[int, str], int]:
    """
    Create Problem Dictionary
    (key: Contest ID and Difficulty, value: Problem Index)
    -----------
    Parameters:
        problems: list[dict]
    --------
    Returns:
        problem_maps: dict[tuple[int, str], int]
    --------
    Example:
        {
            (1, "A"): 1,
            (1, "B"): 2,
            ...
        }
    """
    problem_maps = {}
    for idx, problem in enumerate(reversed(problems), 1):
        problem_maps[(problem["contest_id"], problem["index"])] = idx

    return problem_maps


def sort_each_submission_history(submission_history: list[dict]) -> None:
    """
    Sort Each Submission History
    -----------
    Parameters:
        submission_history: list[dict]
    --------
    Returns:
        sorted_submission_history: list[dict]
    --------
    """
    submission_history.sort(key=lambda x: x["created_at"])


def get_submission_ids(target_submission_history: list[dict]) -> list[dict]:
    """
    Get Submitted Problem Indexes
    -----------
    Parameters:
        target_submission_history: list[dict]
        problem_maps: dict[tuple[int, str], int]
    --------
    Returns:
        handle_with_submission_history: list[dict]
    --------
    Example:
        [
            {
                "handle": "tourist",
                "submission_ids": [1, 2, 3, ...]
            },
            ...
        ]
    """
    submission_maps = []
    for submission_history in target_submission_history:
        submission_map = {"handle": submission_history["user"]["handle"], "submission_ids": []}
        for submission in submission_history["submissions"]:
            problem_id = submission["problem"]["id"]
            if problem_id is not None:
                submission_map["submission_ids"].append(problem_id)
        submission_maps.append(submission_map)

    return submission_maps


def plt_submission_figure(all_submissions: list[dict]) -> None:
    """
    Plot Submission Figure
    -----------
    Parameters:
        all_submissions: list[dict]
    --------
    Returns:
        None
    --------
    """
    plt.figure(figsize=(15, 8))
    plt.xlabel("Submission Index")
    plt.ylabel("Problem Index")

    for submission in all_submissions[:TARGET_USER_NUM] if TARGET_USER_NUM is not None else all_submissions:
        submission_ids = submission["submission_ids"]
        plt.scatter(list(range(len(submission_ids))), submission_ids, s=1)

    plt.legend()
    plt.show()


def run() -> None:
    # Submission History
    all_experts_submission_history = json_loader.load_contents_of_json(
        path_from_root=Path("dataset/users-submission-history-sm.json")
    )
    for submission_history in all_experts_submission_history:
        sort_each_submission_history(submission_history=submission_history["submissions"])

    submission_history_num = len(all_experts_submission_history)
    target_submission_history_idx = (
        rng.choice(submission_history_num, size=TARGET_USER_NUM, replace=False)
        if TARGET_USER_NUM is not None
        else np.arange(submission_history_num)
    )
    target_submission_history = [all_experts_submission_history[idx] for idx in target_submission_history_idx]
    submissions_map = get_submission_ids(target_submission_history=target_submission_history)

    # plot
    plt_submission_figure(all_submissions=submissions_map)


if __name__ == "__main__":
    run()
