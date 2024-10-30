from argparse import Namespace
from collections import OrderedDict, defaultdict
from logging import getLogger
from typing import Literal

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp
import torch
from sklearn.model_selection import train_test_split

from src.constants import SEED
from src.type import Dataset, EntityID, RelationID, SplitSubmissionHistoryByUser, Submission, SubmissionHistory
from src.utils import kg_triplets_generator

logger = getLogger(__name__)
rng = np.random.default_rng()


class Preprocess:
    def __init__(
        self,
        args: Namespace,
        dataset: Dataset,
        cf_batch_size: int,
        kg_batch_size: int,
        device: torch.device,
    ) -> None:
        self._args = args
        self._dataset = dataset
        self._cf_batch_size = cf_batch_size
        self._kg_batch_size = kg_batch_size

        self.user_id_map = {user.id: user for user in self._dataset.users}
        self.problem_id_map = {entity.id: entity for entity in self._dataset.problems}
        self.relation_id_map = {relation.id: relation for relation in self._dataset.relations}
        self.device = device

    def _filter_submission_for_same_problem(self) -> None:
        """
        Filter submission for the same problem.
        """
        for submission_history in self._dataset.all_submission_history:
            submissions_by_user = submission_history.submissions
            unique_submissions: dict[int, Submission] = {}
            for submission in submissions_by_user:
                problem_id = submission.problem.id
                if (
                    problem_id not in unique_submissions
                    or submission.created_at < unique_submissions[problem_id].created_at
                ):
                    unique_submissions[problem_id] = submission
            submission_history.submissions = list(unique_submissions.values())

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
        self.entity_num = len(self.entities)
        self.relation_num = len(self._relations)
        self.triplet_num = len(self.triplets)

        logger.info("users num: %s", self.user_num)
        logger.info("problems num: %s", self.item_num)
        logger.info("entities num: %s", self.entity_num)
        logger.info("relations num (without interaction relation): %s", self.relation_num)
        logger.info("triplets num: %s", self.triplet_num)

    def _convert_to_sparse_matrix(
        self,
        matrix: npt.NDArray[np.int64],
        row_offset_v: int,
        col_offset_v: int,
    ) -> tuple[sp.coo_matrix, sp.coo_matrix]:
        all_num = self.user_num + self.entity_num
        x_rows = matrix[:, 0] + row_offset_v
        x_cols = matrix[:, 1] + col_offset_v
        x_vals = [1.0] * len(x_rows)

        y_rows = x_cols
        y_cols = x_rows
        y_vals = [1.0] * len(y_rows)

        x_adjacency_matrix = sp.coo_matrix((x_vals, (x_rows, x_cols)), shape=(all_num, all_num))
        y_adjacency_matrix = sp.coo_matrix((y_vals, (y_rows, y_cols)), shape=(all_num, all_num))

        return x_adjacency_matrix, y_adjacency_matrix

    def _get_adjacency_matrices(self) -> tuple[list[sp.coo_matrix], list[int]]:
        """
        Get adjacency matrices.

        Returns
        -------
        adjacency_matrices: list[sp.coo_matrix]
            List of adjacency matrices.
        adjacency_relations: list[int]]
            List of adjacency relations.
        """
        adjacency_matrices = []
        adjacency_relations = []

        #####################################
        # Relation between user and problem
        #####################################
        interaction_adjacency_matrix, interaction_adjacency_matrix_inv = self._convert_to_sparse_matrix(
            matrix=self.interaction_matrix,
            row_offset_v=0,
            col_offset_v=self.user_num,
        )
        adjacency_matrices.append(interaction_adjacency_matrix)
        adjacency_relations.append(0)
        adjacency_matrices.append(interaction_adjacency_matrix_inv)
        adjacency_relations.append(self.relation_num + 1)

        #############################
        # Ralation between entities
        #############################
        for relation in self._relations:
            entity_matrix = [
                (triplet.head, triplet.tail) for triplet in self.triplets if triplet.relation == relation.id
            ]
            entity_adjacency_matrix, entity_adjacency_matrix_inv = self._convert_to_sparse_matrix(
                matrix=np.array(entity_matrix),
                row_offset_v=self.user_num,
                col_offset_v=self.user_num,
            )
            adjacency_matrices.append(entity_adjacency_matrix)
            adjacency_relations.append(relation.id + 1)
            adjacency_matrices.append(entity_adjacency_matrix_inv)
            adjacency_relations.append(relation.id + 2 + self.relation_num)
        self.relation_num = len(adjacency_relations)

        return adjacency_matrices, adjacency_relations

    def _get_bi_norm_laplacian_matrices(self) -> list[sp.coo_matrix]:
        """
        Get Laplacian matrices.

        Returns
        -------
        laplacian_matrices: list[sp.coo_matrix]
            List of Laplacian matrices.
        """

        def _bi_normalize(matrix: sp.coo_matrix) -> sp.coo_matrix:
            row_sum = np.array(matrix.sum(axis=1))
            row_sum_inv_sqrt = np.power(row_sum, -0.5).flatten()
            row_sum_inv_sqrt[np.isinf(row_sum_inv_sqrt)] = 0.0
            diagonal_matrix_with_row_sum_inv_sqrt = sp.diags(row_sum_inv_sqrt)
            return (
                diagonal_matrix_with_row_sum_inv_sqrt.dot(matrix)
                .transpose()
                .dot(diagonal_matrix_with_row_sum_inv_sqrt)
                .tocoo()
            )

        return [_bi_normalize(matrix=adjacency_matrix) for adjacency_matrix in self._adjacency_matrices]

    def _get_kg_dict(self) -> dict[EntityID, list[tuple[RelationID, EntityID]]]:
        """
        Get Knowlege Graph Dictionary.

        Returns
        -------
        kg_dict: dict[EntityID, tuple[RelationID, EntityID]]
            Knowlege Graph Dictionary (k: head, v: (relation, tail)).
        """
        kg_dict = defaultdict(list)
        for laplacian_id, laplacian_matrix in enumerate(self.laplacian_matrices):
            rows = laplacian_matrix.row
            cols = laplacian_matrix.col
            for idx in range(len(rows)):
                head = rows[idx]
                relation = self.adjacency_relations[laplacian_id]
                tail = cols[idx]
                kg_dict[head].append((relation, tail))
        return dict(kg_dict)

    def _get_kg_data(self) -> tuple[list[EntityID], list[RelationID], list[EntityID], list[float]]:
        """
        Get Knowlege Graph Data.

        Returns
        -------
        all_heads: list[EntityID]
            List of heads.
        all_relations: list[RelationID]
            List of relations.
        all_tails: list[EntityID]
            List of tails.
        """
        all_heads: list[EntityID] = []
        all_relations: list[RelationID] = []
        all_tails: list[EntityID] = []
        all_values = []

        for laplacian_id, laplacian_matrix in enumerate(self.laplacian_matrices):
            rows = laplacian_matrix.row
            cols = laplacian_matrix.col
            values = laplacian_matrix.data
            all_heads.extend(rows)
            all_relations.extend([self.adjacency_relations[laplacian_id]] * len(rows))
            all_tails.extend(cols)
            all_values.extend(values)

        head_dict: dict[EntityID, tuple[list[RelationID], list[EntityID], list[float]]] = {}
        for idx, head in enumerate(all_heads):
            if head not in head_dict:
                head_dict[head] = ([], [], [])

            head_dict[head][0].append(all_relations[idx])
            head_dict[head][1].append(all_tails[idx])
            head_dict[head][2].append(all_values[idx])

        sorted_head_dict: dict[EntityID, tuple[list[RelationID], list[EntityID], list[float]]] = {}
        for head in head_dict:
            relations, tails, values = head_dict[head]
            order = np.argsort(tails)
            sorted_head_dict[head] = (
                np.array(relations, dtype=np.int64)[order].tolist(),
                np.array(tails, dtype=np.int64)[order].tolist(),
                np.array(values, dtype=np.float32)[order].tolist(),
            )

        ordered_head_dict = OrderedDict(sorted(sorted_head_dict.items()))
        new_all_heads: list[EntityID] = []
        new_all_relations: list[RelationID] = []
        new_all_tails: list[EntityID] = []
        new_all_values: list[float] = []

        for head, (relations, tails, values) in ordered_head_dict.items():
            new_all_heads.extend([head] * len(tails))
            new_all_relations.extend(relations)
            new_all_tails.extend(tails)
            new_all_values.extend(values)

        return new_all_heads, new_all_relations, new_all_tails, new_all_values

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
        positive_problem_ids = self.interaction_dict[target_user_id]
        sample_positive_problem_ids: set[int] = set()

        while len(sample_positive_problem_ids) < num:
            random_idx = rng.integers(low=0, high=len(positive_problem_ids), size=1)[0]
            positive_problem_id = positive_problem_ids[random_idx]
            sample_positive_problem_ids.add(positive_problem_id)

        return list(sample_positive_problem_ids)

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
            negative_item_id = rng.integers(low=0, high=self.item_num, size=1)[0]
            if negative_item_id not in positive_problem_ids:
                negative_problem_ids.add(negative_item_id)

        return list(negative_problem_ids)

    def generate_cf_batch(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
        negative_problem_ids: torch.Tensor
            List of negative problem ids.
        """
        exist_users = list(self.interaction_dict.keys())
        if self._cf_batch_size <= len(exist_users):
            batch_users = rng.choice(exist_users, size=self._cf_batch_size, replace=False)
        else:
            batch_users = rng.choice(exist_users, size=self._cf_batch_size, replace=True)

        positive_problem_ids = []
        negative_problem_ids = []

        for user_id in batch_users:
            positive_problem_ids.extend(self._sample_positive_problems(user_id, 1))
            negative_problem_ids.extend(self._sample_negative_problems(user_id, 1))

        return (
            torch.LongTensor(batch_users),
            torch.LongTensor(positive_problem_ids),
            torch.LongTensor(negative_problem_ids),
        )

    def _sample_positive_triplets_for_head(self, head: EntityID, num: int) -> tuple[list[RelationID], list[EntityID]]:
        """
        Sample positive triplets for head.

        Parameters
        ----------
        head: EntityID
            Head.
        num: int
            Number of positive triplets.

        Returns
        -------
        positive_relations: list[RelationID]
            List of sampled positive relations.
        positive_tails: list[EntityID]
            List of sampled positive tails.
        """
        positive_triplets = self._kg_dict[head]
        positive_relations: list[RelationID] = []
        positive_tails: list[EntityID] = []

        while True:
            if len(positive_relations) >= num:
                break

            triplet_id = rng.integers(low=0, high=len(positive_triplets))
            relation = positive_triplets[triplet_id][0]
            tail = positive_triplets[triplet_id][1]

            if relation not in positive_relations and tail not in positive_tails:
                positive_relations.append(relation)
                positive_tails.append(tail)

        return positive_relations, positive_tails

    def _sample_negative_triplets_for_head(self, head: EntityID, relation: RelationID, num: int) -> list[EntityID]:
        """
        Sample negative triplets for head.

        Parameters
        ----------
        head: EntityID
            Head.
        relation: RelationID
            Relation.
        num: int
            Number of negative triplets.

        Returns
        -------
        negative_tails: list[EntityID]
            List of sampled negative tails.
        """
        positive_triplets = self._kg_dict[head]
        negative_tails: list[EntityID] = []

        while True:
            if len(negative_tails) >= num:
                break

            tail = rng.integers(low=0, high=self.user_num + self.entity_num, size=1)[0]
            if (relation, tail) not in positive_triplets and tail not in negative_tails:
                negative_tails.append(tail)

        return negative_tails

    def generate_kg_batch(self) -> tuple[list[EntityID], list[RelationID], list[EntityID], list[EntityID]]:
        """
        Generate KG batch.

        Parameters
        ----------
        batch_size: int
            Batch size.

        Returns
        -------
        heads: list[EntityID]
            List of heads.
        positive_relation_batch: list[RelationID]
            List of positive relations.
        positive_tail_batch: list[EntityID]
            List of positive tails.
        negative_tail_batch: list[EntityID]
            List of negative tails.
        """
        exist_heads = list(self._kg_dict.keys())
        allow_duplicates = True
        if self._kg_batch_size <= len(exist_heads):
            allow_duplicates = False
        heads: list[EntityID] = rng.choice(
            a=exist_heads,
            size=self._kg_batch_size,
            replace=allow_duplicates,
        ).tolist()

        positive_relation_batch, positive_tail_batch, negative_tail_batch = [], [], []

        for head in heads:
            positive_relations, positive_tails = self._sample_positive_triplets_for_head(
                head=head,
                num=1,
            )
            positive_relation_batch.extend(positive_relations)
            positive_tail_batch.extend(positive_tails)
            negative_tails = self._sample_negative_triplets_for_head(
                head=head,
                relation=positive_relations[0],
                num=1,
            )
            negative_tail_batch.extend(negative_tails)

        return heads, positive_relation_batch, positive_tail_batch, negative_tail_batch

    def run(self, dataset_name: Literal["training", "test", "validation"]) -> None:
        # Filter submission for the same problem.
        self._filter_submission_for_same_problem()

        # Split submission history into train, test, and validation.
        all_submission_history = self._split_submission_history()

        # Train dataset.
        self._train_dataset = Dataset(
            users=self._dataset.users,
            all_submission_history=[submission_history.train for submission_history in all_submission_history],
            contests=self._dataset.contests,
            problems=self._dataset.problems,
            relations=self._dataset.relations,
        )

        # Test dataset.
        self._test_dataset = Dataset(
            users=self._dataset.users,
            all_submission_history=[submission_history.test for submission_history in all_submission_history],
            contests=self._dataset.contests,
            problems=self._dataset.problems,
            relations=self._dataset.relations,
        )

        # Validation dataset.
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
        self.entities, self._relations, self.triplets = kg_triplets_generator.generate(
            args=self._args, dataset=self._dataset
        )

        # Get statistics of dataset.
        self._get_statistics()

        # Get adjacency matrix.
        self._adjacency_matrices, self.adjacency_relations = self._get_adjacency_matrices()

        # Get Laplacian matrix.
        self.laplacian_matrices = self._get_bi_norm_laplacian_matrices()

        # Get Knowlege Graph Dictionary (k: head, v: (relation, tail)).
        self._kg_dict = self._get_kg_dict()

        # Get Knowlege Graph Data.
        self.all_heads, self.all_relation_indices, self.all_tails, self.all_values = self._get_kg_data()

        # Sum up all laplacian matrices
        total_laplacian_matrix = sum(self.laplacian_matrices).tocoo()
        self.attentive_matrix = torch.sparse_coo_tensor(
            indices=torch.LongTensor(np.vstack((total_laplacian_matrix.row, total_laplacian_matrix.col))),
            values=torch.FloatTensor(total_laplacian_matrix.data),
            size=torch.Size(total_laplacian_matrix.shape),
        )
