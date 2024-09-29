from dataclasses import dataclass, field
from enum import Enum
from typing import Literal


@dataclass
class User:
    id: int
    handle: str
    rating: int
    max_rating: int


@dataclass
class Tag:
    id: int
    name: str


@dataclass
class Rating:
    id: int
    value: int


@dataclass
class Problem:
    id: int
    contest_id: int
    index: str
    name: str
    type: str
    tags: list[Tag]
    rating: Rating | None = field(default=None)
    points: float | None = field(default=None)

    def __post_init__(self) -> None:
        self.tags = [Tag(**tag) if isinstance(tag, dict) else tag for tag in self.tags]
        if self.rating is not None and isinstance(self.rating, dict):
            self.rating = Rating(**self.rating)


@dataclass
class Submission:
    id: int
    problem: Problem
    created_at: str
    result: str | None = field(default=None)

    def __post_init__(self) -> None:
        self.problem = Problem(**self.problem) if isinstance(self.problem, dict) else self.problem


@dataclass
class SubmissionHistory:
    user: User
    submissions: list[Submission]

    def __post_init__(self) -> None:
        self.user = User(**self.user) if isinstance(self.user, dict) else self.user
        self.submissions = [
            Submission(**submission) if isinstance(submission, dict) else submission for submission in self.submissions
        ]


@dataclass
class SplitSubmissionHistoryByUser:
    train: SubmissionHistory
    test: SubmissionHistory
    validation: SubmissionHistory


class RelationType(Enum):
    TAGGED = 0
    HAS_DIFFICULTY = 1


EntityID = int
RelationID = int
Weight = float


@dataclass
class Entity:
    id: EntityID
    target_type: Literal["user", "problem", "tag", "rating"]
    target_id: int


@dataclass
class Relation:
    id: RelationID
    name: str


@dataclass
class Triplet:
    head: EntityID
    relation: RelationID
    tail: EntityID


@dataclass
class Dataset:
    users: list[User]
    all_submission_history: list[SubmissionHistory]
    problems: list[Problem]
    relations: list[Relation]


class SIZE(Enum):
    SM = 1
    DEFAULT = 2
