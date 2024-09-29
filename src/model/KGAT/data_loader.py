from argparse import Namespace
from pathlib import Path

from src.model.KGAT.dataset import Contest, Dataset, Problem, Relation, SubmissionHistory, User
from src.utils import json_loader


class DataLoader:
    def __init__(self, dataset_dir: Path) -> None:
        self._dataset_dir = dataset_dir

    def load_dataset(self, args: Namespace) -> Dataset:
        users = [
            User(**user)
            for user in json_loader.load_contents_of_json(
                path_from_root=Path.joinpath(self._dataset_dir, f"users{'-sm' if args.sm else ''}.json")
            )
        ]
        all_submission_history = [
            SubmissionHistory(**submission_history)
            for submission_history in json_loader.load_contents_of_json(
                path_from_root=Path.joinpath(
                    self._dataset_dir, f"users-submission-history{'-sm' if args.sm else ''}.json"
                )
            )
        ]
        contests = [
            Contest(**contest)
            for contest in json_loader.load_contents_of_json(
                path_from_root=Path.joinpath(self._dataset_dir, "contests.json")
            )
        ]
        problems = [
            Problem(**problem)
            for problem in json_loader.load_contents_of_json(
                path_from_root=Path.joinpath(self._dataset_dir, "problems.json")
            )
        ]
        relations = [
            Relation(**relation)
            for relation in json_loader.load_contents_of_json(
                path_from_root=Path.joinpath(self._dataset_dir, "relations.json")
            )
        ]
        return Dataset(
            users=users,
            all_submission_history=all_submission_history,
            contests=contests,
            problems=problems,
            relations=relations,
        )
