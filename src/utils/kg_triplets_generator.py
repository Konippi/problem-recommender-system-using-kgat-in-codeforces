import dataclasses
from argparse import Namespace
from pathlib import Path
from typing import Literal

from src.type import Dataset, Entity, EntityID, Relation, RelationType, Triplet
from src.utils import json_writer

EntityDict = dict[tuple[Literal["problem", "contest", "contest_division", "tag", "rating"], int], EntityID]


def _add_problems_to_entities(dataset: Dataset, entity_dict: EntityDict, start_entity_id: int) -> list[Entity]:
    entity_id = start_entity_id
    entities = []

    for problem in dataset.problems:
        if ("problem", problem.id) not in entity_dict:
            entity_dict[("problem", problem.id)] = entity_id
            entities.append(Entity(id=entity_id, target_type="problem", target_id=problem.id))
            entity_id += 1

    return entities


def _create_triplets_problem_with_contest(
    dataset: Dataset,
    entity_dict: EntityDict,
    start_entity_id: int,
) -> tuple[list[Entity], list[Triplet]]:
    entity_id = start_entity_id
    entities = []
    relation = RelationType.IN_CONTEST
    triplets = set()
    contest_map = {contest.id: contest for contest in dataset.contests}

    for problem in dataset.problems:
        head_entity_id = entity_dict[("problem", problem.id)]

        contest = contest_map[problem.contest_id]
        if ("contest", contest.id) not in entity_dict:
            tail_entity_id = entity_id
            entity_dict[("contest", contest.id)] = entity_id
            entities.append(Entity(id=entity_id, target_type="contest", target_id=contest.id))
            entity_id += 1
        else:
            tail_entity_id = entity_dict[("contest", contest.id)]

        triplets.add(Triplet(head=head_entity_id, relation=relation.value, tail=tail_entity_id))

    return entities, list(triplets)


def _create_triplets_contest_with_contest_division(
    dataset: Dataset,
    entity_dict: EntityDict,
    start_entity_id: int,
) -> tuple[list[Entity], list[Triplet]]:
    entity_id = start_entity_id
    entities = []
    relation = RelationType.HAS_CONTEST_DIVISION
    triplets = set()
    contest_map = {contest.id: contest for contest in dataset.contests}

    for problem in dataset.problems:
        contest = contest_map[problem.contest_id]
        head_entity_id = entity_dict[("contest", problem.contest_id)]
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


def _create_triplets_problem_with_tag(
    dataset: Dataset,
    entity_dict: EntityDict,
    start_entity_id: int,
) -> tuple[list[Entity], list[Triplet]]:
    entity_id = start_entity_id
    entities = []
    relation = RelationType.TAGGED
    triplets = set()

    for problem in dataset.problems:
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


def _create_triplets_problem_with_rating(
    dataset: Dataset,
    entity_dict: EntityDict,
    start_entity_id: int,
) -> tuple[list[Entity], list[Triplet]]:
    entity_id = start_entity_id
    entities = []
    relation = RelationType.HAS_DIFFICULTY
    triplets = set()

    for problem in dataset.problems:
        if problem.rating is None:
            continue

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
    entities: list[Entity] = []
    relations = [Relation(id=relation.value, name=relation.name) for relation in RelationType]
    all_triplets: list[Triplet] = []

    entity_dict: EntityDict = {}

    # Add all problems to entities
    problem_entities = _add_problems_to_entities(
        dataset=dataset,
        entity_dict=entity_dict,
        start_entity_id=0,
    )
    entities.extend(problem_entities)

    # Create triplets for problem with contest
    contest_entities, contest_triplets = _create_triplets_problem_with_contest(
        dataset=dataset,
        entity_dict=entity_dict,
        start_entity_id=len(entities),
    )
    entities.extend(contest_entities)
    all_triplets.extend(contest_triplets)

    # Create triplets for problem with contest division
    contest_division_entities, contest_division_triplets = _create_triplets_contest_with_contest_division(
        dataset=dataset,
        entity_dict=entity_dict,
        start_entity_id=len(entities),
    )
    entities.extend(contest_division_entities)
    all_triplets.extend(contest_division_triplets)

    # Create triplets for problem with tag
    tagged_entities, tagged_triplets = _create_triplets_problem_with_tag(
        dataset=dataset,
        entity_dict=entity_dict,
        start_entity_id=len(entities),
    )
    entities.extend(tagged_entities)
    all_triplets.extend(tagged_triplets)

    # Create triplets for problem with difficulty
    has_difficulty_entities, has_difficulty_triplets = _create_triplets_problem_with_rating(
        dataset=dataset,
        entity_dict=entity_dict,
        start_entity_id=len(entities),
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
