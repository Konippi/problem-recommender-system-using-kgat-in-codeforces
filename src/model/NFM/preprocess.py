from argparse import Namespace
from logging import getLogger
from typing import Literal

import numpy as np
import torch
from scipy import sparse as sp
from sklearn.model_selection import train_test_split

from src.constants import SEED
from src.type import Dataset, SplitSubmissionHistoryByUser, SubmissionHistory
from src.utils import kg_triplets_generator

logger = getLogger(__name__)
rng = np.random.default_rng()


class Preprocess:
    def __init__(
        self,
        args: Namespace,
        dataset: Dataset,
        train_batch_size: int,
        test_batch_size: int,
    ) -> None:
        self._args = args
        self._dataset = dataset
        self._train_batch_size = train_batch_size
        self._test_batch_size = test_batch_size

    def _split_submission_history(self) -> list[SplitSubmissionHistoryByUser]:
        """
        Split submission history into train, test, and validation.

        Returns
        -------
        split_submission_history_by_user: list[SplitSubmissionHistoryByUser]
            List of SplitSubmissionHistoryByUser.
        """
        all_split_submission_history_by_user: list[SplitSubmissionHistoryByUser] = []

        for user in self._dataset.users:
            submission_history = sorted(
                next(
                    filter(
                        lambda x: x.user.handle == user.handle,
                        self._dataset.all_submission_history,
                    )
                ).submissions,
                key=lambda x: x.created_at,
            )

            tmp_train, test = train_test_split(
                submission_history,
                train_size=0.8,
                test_size=0.2,
                shuffle=True,
                random_state=SEED,
            )
            train, validation = train_test_split(
                tmp_train,
                train_size=0.9,
                test_size=0.1,
                shuffle=True,
                random_state=SEED,
            )
            all_split_submission_history_by_user.append(
                SplitSubmissionHistoryByUser(
                    SubmissionHistory(user, train),
                    SubmissionHistory(user, test),
                    SubmissionHistory(user, validation),
                )
            )

        train_num = 0
        test_num = 0
        validation_num = 0
        for split_submission_history_by_user in all_split_submission_history_by_user:
            train_num += len(split_submission_history_by_user.train.submissions)
            test_num += len(split_submission_history_by_user.test.submissions)
            validation_num += len(split_submission_history_by_user.validation.submissions)

        logger.info("train submissions num (with duplicate submissions): %s", train_num)
        logger.info("test submissions num (with duplicate submissions): %s", test_num)
        logger.info("validation submissions num (with duplicate submissions): %s", validation_num)

        return all_split_submission_history_by_user

    def _get_interaction_matrix(
        self, all_submission_history: list[SubmissionHistory]
    ) -> tuple[np.ndarray, dict[int, list[int]]]:
        """
        Get interaction matrix.

        Parameters
        ----------
        all_submission_history: list[SubmissionHistory]
            List of submission history.

        Returns
        -------
        interaction_matrix: np.ndarray
            Interaction matrix.
        interaction_map: dict[int, list[int]]
            Interaction map.
        """
        interaction_map = {
            submission_history.user.id: list({submission.problem.id for submission in submission_history.submissions})
            for submission_history in all_submission_history
        }

        interaction_matrix = [
            [user_id, problem_id] for user_id, problem_ids in interaction_map.items() for problem_id in problem_ids
        ]

        return np.array(interaction_matrix), interaction_map

    def _get_statistics(self) -> None:
        """
        Get statistics of dataset.
        """
        self.user_num = len(self._dataset.users)
        self.item_num = len(self._dataset.problems)
        self.entity_num = len(self._entities)
        self.relation_num = len(self._relations)
        self.triplet_num = len(self._triplets)

        logger.info("users num: %s", self.user_num)
        logger.info("problems num: %s", self.item_num)
        logger.info("entities num: %s", self.entity_num)
        logger.info("relations num (without interaction relation): %s", self.relation_num)
        logger.info("triplets num: %s", self.triplet_num)

    def _get_user_and_feature_matrices(self) -> tuple[sp.csr_matrix, sp.csr_matrix]:
        """
        Get user and feature matrices.

        Returns
        -------
        user_matrix: sp.csr_matrix
            User matrix.
        feature_matrix: sp.csr_matrix
            Feature matrix.
        """
        ##################
        # User matrix
        ##################
        user_matrix = sp.identity(self.user_num).tocsr()

        ##################
        # Feature matrix
        ##################
        feature_rows = list(range(self.item_num))
        feature_cols = list(range(self.item_num))
        feature_data = [1] * self.item_num

        entity_map = {entity.id: entity for entity in self._entities}
        for triplet in self._triplets:
            if entity_map[triplet.head].target_type == "problem":
                feature_rows.append(triplet.head)
                feature_cols.append(triplet.tail)
                feature_data.append(1)

        feature_matrix = sp.coo_matrix(
            (feature_data, (feature_rows, feature_cols)), shape=(self.item_num, self.entity_num)
        ).tocsr()

        logger.info("user matrix shape: %s", user_matrix.shape)
        logger.info("feature matrix shape: %s", feature_matrix.shape)

        return user_matrix, feature_matrix

    def _sample_positive_problems(self, target_user_id: int, num: int) -> list[int]:
        """
        Sample positive problems.

        Parameters
        ----------
        interaction_matrix: np.ndarray
            Interaction matrix.
        target_user_id: int
            Target user id.

        Returns
        -------
        sampled_positive_problem_ids: list[int]
            List of sampled positive problem ids.
        """
        positive_problem_ids: list[int] = rng.choice(
            a=self.interaction_dict[target_user_id], size=num, replace=False
        ).tolist()

        return positive_problem_ids

    def _sample_negative_problems(self, target_user_id: int, num: int) -> list[int]:
        """
        Sample negative problems.

        Parameters
        ----------
        positive_problem_ids: list[int]
            List of positive problem ids.
        num: int
            Number of negative problems.

        Returns
        -------
        negative_problem_ids: list[int]
            List of negative problem ids.
        """
        positive_problem_ids = self.interaction_dict[target_user_id]
        negative_problem_ids: set[int] = set()

        while len(negative_problem_ids) < num:
            negative_problem_id = rng.integers(low=0, high=self.item_num)
            if negative_problem_id not in positive_problem_ids:
                negative_problem_ids.add(negative_problem_id)

        return list(negative_problem_ids)

    def _generate_cf_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate CF batch.

        Parameters
        ----------
        batch_size: int
            Batch size.

        Returns
        -------
        user_ids: torch.Tensor
            List of user ids.
        positive_problem_ids: torch.Tensor
            List of positive problem ids.
        negative_item_ids: torch.Tensor
            List of negative problem ids.
        """
        allow_duplicates = True
        if self._train_batch_size <= self.user_num:
            allow_duplicates = False
        user_ids = rng.choice(
            a=[user.id for user in self._dataset.users],
            size=self._train_batch_size,
            replace=allow_duplicates,
        )
        positive_problem_ids = []
        negative_item_ids = []

        for user_id in user_ids:
            positive_problem_ids.extend(self._sample_positive_problems(target_user_id=user_id, num=1))
            negative_item_ids.extend(self._sample_negative_problems(target_user_id=user_id, num=1))

        return torch.LongTensor(user_ids), torch.LongTensor(positive_problem_ids), torch.LongTensor(negative_item_ids)

    def _convert_to_tensor(
        self,
        coo_matrix: sp.coo_matrix,
    ) -> torch.Tensor:
        """
        Convert coo matrix to tensor.

        Parameters
        ----------
        coo_matrix: sp.coo_matrix
            COO matrix.

        Returns
        -------
        tensor: torch.Tensor
            Tensor.
        """
        values = coo_matrix.data
        indices = np.vstack((coo_matrix.row, coo_matrix.col))
        return torch.sparse_coo_tensor(
            indices=torch.LongTensor(indices),
            values=torch.FloatTensor(values),
            size=torch.Size(coo_matrix.shape),
        )

    def generate_train_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate train batch.

        Returns
        -------
        positive_feature_values: torch.Tensor
            Positive feature values.
        negative_feature_values: torch.Tensor
            Negative feature values.
        """
        batch_user_ids, batch_positive_problem_ids, batch_negative_problem_ids = self._generate_cf_batch()
        batch_user_matrix = self._user_matrix[batch_user_ids.numpy()]
        batch_positive_feature_matrix = self._feature_matrix[batch_positive_problem_ids.numpy()]
        batch_negative_feature_matrix = self._feature_matrix[batch_negative_problem_ids.numpy()]

        positive_feature_values = self._convert_to_tensor(
            coo_matrix=sp.hstack([batch_user_matrix, batch_positive_feature_matrix]).tocoo()
        )
        negative_feature_values = self._convert_to_tensor(
            coo_matrix=sp.hstack([batch_user_matrix, batch_negative_feature_matrix]).tocoo()
        )

        return positive_feature_values, negative_feature_values

    def generate_test_batch(self, batch_user_ids: torch.Tensor) -> torch.Tensor:
        """
        Generate test batch.

        Parameters
        ----------
        batch_user_ids: torch.Tensor
            List of user ids.

        Returns
        -------
        test_feature_values: torch.Tensor
            Test feature values.
        """
        row_num = len(batch_user_ids) * self.item_num
        user_rows = list(range(row_num))
        user_cols = np.repeat(batch_user_ids, self.item_num)
        user_data = [1] * row_num

        batch_user_matrix = sp.coo_matrix((user_data, (user_rows, user_cols)), shape=(row_num, self.user_num)).tocsr()
        batch_feature_matrix = sp.vstack([self._feature_matrix] * len(batch_user_ids))

        return self._convert_to_tensor(coo_matrix=sp.hstack([batch_user_matrix, batch_feature_matrix]).tocoo())

    def run(self, dataset_name: Literal["training", "test", "validation"]) -> None:
        # Split submission history into train, test, and validation.
        all_submission_history = self._split_submission_history()

        # Generate triplets for train.
        self._train_dataset = Dataset(
            users=self._dataset.users,
            all_submission_history=[submission_history.train for submission_history in all_submission_history],
            contests=self._dataset.contests,
            problems=self._dataset.problems,
            relations=self._dataset.relations,
        )

        # Generate triplets for test.
        self._test_dataset = Dataset(
            users=self._dataset.users,
            all_submission_history=[submission_history.test for submission_history in all_submission_history],
            contests=self._dataset.contests,
            problems=self._dataset.problems,
            relations=self._dataset.relations,
        )

        # Generate triplets for validation.
        self._validation_dataset = Dataset(
            users=self._dataset.users,
            all_submission_history=[submission_history.validation for submission_history in all_submission_history],
            contests=self._dataset.contests,
            problems=self._dataset.problems,
            relations=self._dataset.relations,
        )

        # Generate interaction matrices.
        self.train_interaction_matrix, self.train_interaction_dict = self._get_interaction_matrix(
            all_submission_history=self._train_dataset.all_submission_history
        )
        self.test_interaction_matrix, self.test_interaction_dict = self._get_interaction_matrix(
            all_submission_history=self._test_dataset.all_submission_history
        )
        self.validation_interaction_matrix, self.validation_interaction_dict = self._get_interaction_matrix(
            all_submission_history=self._validation_dataset.all_submission_history
        )

        # Set dataset.
        self._dataset = (
            self._train_dataset
            if dataset_name == "training"
            else self._test_dataset
            if dataset_name == "test"
            else self._validation_dataset
        )
        self.interaction_matrix = (
            self.train_interaction_matrix
            if dataset_name == "training"
            else self.test_interaction_matrix
            if dataset_name == "test"
            else self.validation_interaction_matrix
        )
        self.interaction_dict = (
            self.train_interaction_dict
            if dataset_name == "training"
            else self.test_interaction_dict
            if dataset_name == "test"
            else self.validation_interaction_dict
        )

        # Generate tripplets for Knowledge Graph (without interaction triplets).
        self._entities, self._relations, self._triplets = kg_triplets_generator.generate(
            args=self._args, dataset=self._dataset
        )

        # Get statistics of dataset.
        self._get_statistics()

        self._user_matrix, self._feature_matrix = self._get_user_and_feature_matrices()
