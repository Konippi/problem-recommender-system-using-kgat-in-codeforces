import dataclasses
from argparse import Namespace
from pathlib import Path
from typing import Literal

from src.model.KGAT.data_loader import DataLoader
from src.model.KGAT.dataset import Dataset, Entity, EntityID, Relation, RelationType, Triplet
from src.utils import json_writer

EntityDict = dict[tuple[Literal["user", "problem", "contest_division", "tag", "rating"], int], EntityID]


def load_dataset(args: Namespace) -> Dataset:
    data_loader = DataLoader(dataset_dir=Path("dataset"))
    return data_loader.load_dataset(args=args)


# def get_users_entities(dataset: Dataset, entity_dict: EntityDict) -> list[Entity]:
#     entity_id = list(entity_dict.values())[-1] + 1 if entity_dict else 0
#     entities = []

#     for user in dataset.users:
#         if ("user", user.id) not in entity_dict:
#             entity_dict[("user", user.id)] = entity_id
#             entities.append(Entity(id=entity_id, target_type="user", target_id=user.id))

#     return entities


def create_triplets_problem_with_contest_division(
    dataset: Dataset,
    entity_dict: EntityDict,
) -> tuple[list[Entity], list[Triplet]]:
    entity_id = list(entity_dict.values())[-1] + 1 if entity_dict else 0
    entities = []
    relation = RelationType.IN_CONTEST_DIVISION
    triplets = set()
    contest_map = {contest.id: contest for contest in dataset.contests}

    for problem in dataset.problems:
        contest = contest_map[problem.contest_id]

        if ("problem", problem.id) not in entity_dict:
            head_entity_id = entity_id
            entity_dict[("problem", problem.id)] = entity_id
            entities.append(Entity(id=entity_id, target_type="problem", target_id=problem.id))
            entity_id += 1
        else:
            head_entity_id = entity_dict[("problem", problem.id)]

        if contest.division_id is not None:
            if ("contest_division", contest.division_id) not in entity_dict:
                tail_entity_id = entity_id
                entity_dict[("contest_division", contest.division_id)] = entity_id
                entities.append(Entity(id=entity_id, target_type="contest_division", target_id=contest.division_id))
                entity_id += 1
            else:
                tail_entity_id = entity_dict[("contest_division", contest.division_id)]

            triplets.add(Triplet(head=head_entity_id, relation=relation.value, tail=tail_entity_id))

    return entities, list(triplets)


def create_triplets_problem_with_tag(
    dataset: Dataset,
    entity_dict: EntityDict,
) -> tuple[list[Entity], list[Triplet]]:
    entity_id = list(entity_dict.values())[-1] + 1 if entity_dict else 0
    entities = []
    relation = RelationType.TAGGED
    triplets = set()

    for problem in dataset.problems:
        if ("problem", problem.id) not in entity_dict:
            head_entity_id = entity_id
            entity_dict[("problem", problem.id)] = entity_id
            entities.append(Entity(id=entity_id, target_type="problem", target_id=problem.id))
            entity_id += 1
        else:
            head_entity_id = entity_dict[("problem", problem.id)]
        for tag in problem.tags:
            if ("tag", tag.id) not in entity_dict:
                tail_entity_id = entity_id
                entity_dict[("tag", tag.id)] = entity_id
                entities.append(Entity(id=entity_id, target_type="tag", target_id=tag.id))
                entity_id += 1
            else:
                tail_entity_id = entity_dict[("tag", tag.id)]

            triplets.add(Triplet(head=head_entity_id, relation=relation.value, tail=tail_entity_id))
    return entities, list(triplets)


def create_triplets_problem_with_difficulty(
    dataset: Dataset,
    entity_dict: EntityDict,
) -> tuple[list[Entity], list[Triplet]]:
    entity_id = list(entity_dict.values())[-1] + 1 if entity_dict else 0
    entities = []
    relation = RelationType.HAS_DIFFICULTY
    triplets = set()

    for problem in dataset.problems:
        if problem.rating is None:
            continue

        if ("problem", problem.id) not in entity_dict:
            head_entity_id = entity_id
            entity_dict[("problem", problem.id)] = entity_id
            entities.append(Entity(id=entity_id, target_type="problem", target_id=problem.id))
            entity_id += 1
        else:
            head_entity_id = entity_dict[("problem", problem.id)]

        if ("rating", problem.rating.id) not in entity_dict:
            tail_entity_id = entity_id
            entity_dict[("rating", problem.rating.id)] = entity_id
            entities.append(Entity(id=entity_id, target_type="rating", target_id=problem.rating.id))
            entity_id += 1
        else:
            tail_entity_id = entity_dict[("rating", problem.rating.id)]

        triplets.add(Triplet(head=head_entity_id, relation=relation.value, tail=tail_entity_id))

    return entities, list(triplets)


def generate(args: Namespace, dataset: Dataset) -> tuple[list[Entity], list[Relation], list[Triplet]]:
    entities = []
    relations = [Relation(id=relation.value, name=relation.name) for relation in RelationType]
    all_triplets = []

    entity_dict: EntityDict = {}

    # # Add users to entities
    # user_entities = get_users_entities(
    #     dataset=dataset,
    #     entity_dict=entity_dict,
    # )
    # entities.extend(user_entities)

    # Create triplets for problem with contest division
    contest_division_entities, contest_division_triplets = create_triplets_problem_with_contest_division(
        dataset=dataset,
        entity_dict=entity_dict,
    )
    entities.extend(contest_division_entities)
    all_triplets.extend(contest_division_triplets)

    # Create triplets for problem with tag
    tagged_entities, tagged_triplets = create_triplets_problem_with_tag(
        dataset=dataset,
        entity_dict=entity_dict,
    )
    entities.extend(tagged_entities)
    all_triplets.extend(tagged_triplets)

    # Create triplets for problem with difficulty
    has_difficulty_entities, has_difficulty_triplets = create_triplets_problem_with_difficulty(
        dataset=dataset,
        entity_dict=entity_dict,
    )
    entities.extend(has_difficulty_entities)
    all_triplets.extend(has_difficulty_triplets)

    # Generate json files
    json_writer.generate_json_by_dict(
        contents=[dataclasses.asdict(entity) for entity in entities],
        path_from_root=Path(f"../dataset/entities{'-sm' if args.sm else ''}.json"),
    )
    json_writer.generate_json_by_dict(
        contents=[dataclasses.asdict(triplet) for triplet in all_triplets],
        path_from_root=Path(f"../dataset/triplets{'-sm' if args.sm else ''}.json"),
    )

    return entities, relations, all_triplets
