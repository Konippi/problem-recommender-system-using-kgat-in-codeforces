import sys

sys.path.append("../..")

import dataclasses
import time
from logging import INFO, basicConfig, getLogger
from pathlib import Path
from typing import Any

import requests
from requests import Session
from tenacity import retry, stop_after_attempt, wait_fixed

from src.constants import HTTP_USER_AGENT
from src.model.KGAT.dataset import Contest, Division, Problem, Rating, Tag
from src.utils import json_writer, retry_settings

basicConfig(level=INFO)
logger = getLogger(__name__)


@retry(stop=stop_after_attempt(3), wait=wait_fixed(60))
def get_all_problems(session: Session, contest_ids: list[int]) -> list[dict[str, Any]]:
    """
    Get All Problems in Codeforces
    -----------
    Parameters:
        None
    --------
    Returns:
        problems: list[dict[str, Any]]
    --------
    Example:
        [
            {
                "contestId": 1956,
                "index": F,
                "name": "Nene and the Passing Game",
                "type": "PROGRAMMING",
                "points": 2500.0,
                "rating": 2000,
                "tags": [
                    "constructive algorithms",
                    "data structures",
                    "dsu",
                    "graphs",
                    "sortings"
                ]
            },
            ...
        ]
    """
    problems = []
    for contest_id in contest_ids:
        url = f"https://codeforces.com/api/contest.standings?contestId={contest_id}&count=1"
        headers = {"Content-Type": "application/json", "User-Agent": HTTP_USER_AGENT}

        logger.info("GET: %s", url)

        try:
            response = session.get(url=url, headers=headers, timeout=(120, 60))
            response.raise_for_status()
        except requests.HTTPError:
            if str(response.status_code).startswith("4"):  # 4xx Client Error
                logger.warning("Not Found Contest Id: %s", contest_id)
                time.sleep(0.2)
                continue

            logger.exception("HTTP Request Error: %s", response.status_code)

        problems.extend(response.json()["result"]["problems"])
        time.sleep(0.5)

    return problems  # type: ignore[no-any-return]


def get_all_problems_from_problemsets(session: Session) -> list[dict[str, Any]]:
    """
    Get All Problems from Problemsets in Codeforces
    -----------
    Parameters:
        None
    --------
    Returns:
        problems: list[dict[str, Any]]
    --------
    Example:
        [
            {
                "contestId": 1956,
                "index": F,
                "name": "Nene and the Passing Game",
                "type": "PROGRAMMING",
                "points": 2500.0,
                "rating": 2000,
                "tags": [
                    "constructive algorithms",
                    "data structures",
                    "dsu",
                    "graphs",
                    "sortings"
                ]
            },
            ...
        ]
    """
    url = "https://codeforces.com/api/problemset.problems"
    headers = {"Content-Type": "application/json", "User-Agent": HTTP_USER_AGENT}

    logger.info("GET: %s", url)

    try:
        response = session.get(url=url, headers=headers, timeout=60)
        response.raise_for_status()
    except requests.HTTPError:
        logger.exception("HTTP Request Error: %s", response.status_code)

    return response.json()["result"]["problems"]  # type: ignore[no-any-return]


def get_all_contests(session: Session) -> list[dict[str, Any]]:
    """
    Get All Contest Ids in Codeforces
    -----------
    Parameters:
        None
    --------
    Returns:
        contests: list[dict]
    """
    url = "https://codeforces.com/api/contest.list"
    headers = {"Content-Type": "application/json", "User-Agent": HTTP_USER_AGENT}

    logger.info("GET: %s", url)

    try:
        response = session.get(url=url, headers=headers, timeout=30)
        response.raise_for_status()
    except requests.HTTPError:
        logger.warning("HTTP Request Error: %s", response.status_code)

    return [contest for contest in response.json()["result"] if contest["phase"] == "FINISHED"]


def get_tag_with_ids(problems: list[dict[str, Any]]) -> list[Tag]:
    """
    Assign index to each tag
    -----------
    Parameters:
        problems: list[dict[str, Any]]
    --------
    Returns:
        tag_with_ids: list[Tag]
    --------
    """
    tags = {tag for problem in problems for tag in problem["tags"]}
    return [Tag(id=idx, name=tag) for idx, tag in enumerate(tags)]


def get_tag_by_name(name: str, tags: list[Tag]) -> Tag:
    return next(filter(lambda tag: tag.name == name, tags))


def get_rating_with_ids(problems: list[dict[str, Any]]) -> list[Rating]:
    """
    Assign index to each rating
    -----------
    Parameters:
        problems: list[dict[str, Any]]
    --------
    Returns:
        rating_with_ids: list[Rating]
    --------
    """
    ratings = sorted({problem["rating"] for problem in problems if problem["rating"] is not None})
    return [Rating(id=idx, value=rating) for idx, rating in enumerate(ratings)]


def get_rating_by_value(value: int | None, ratings: list[Rating]) -> Rating:
    return next(filter(lambda rating: rating.value == value, ratings))


def run() -> None:
    # Retry Settings
    session = retry_settings.setup()

    # contests
    contests = get_all_contests(session=session)

    all_contests: list[Contest] = []
    for contest in contests:
        name = contest["name"]
        division = None

        if "Div. 1 + Div. 2" in name:
            division = Division.DIV1AND2
        elif "Div. 1" in name:
            division = Division.DIV1
        elif "Div. 2" in name:
            division = Division.DIV2
        elif "Div. 3" in name:
            division = Division.DIV3
        elif "Div. 4" in name:
            division = Division.DIV4

        all_contests.append(
            Contest(
                id=contest["id"],
                name=name,
                type=contest["type"],
                division_id=division.value if division is not None else None,
            )
        )

    all_contests = sorted(all_contests, key=lambda contest: contest.id)
    json_writer.generate_json_by_dict(
        contents=[dataclasses.asdict(contest) for contest in all_contests],
        path_from_root=Path("dataset/contests.json"),
    )

    contest_ids = [contest.id for contest in all_contests]

    # divisions
    divisions = [{"id": division.value, "name": division.name.lower()} for division in Division]
    json_writer.generate_json_by_dict(
        contents=divisions,
        path_from_root=Path("dataset/contest-divisions.json"),
    )

    # problems
    problems = get_all_problems(session=session, contest_ids=contest_ids)
    problems = sorted(
        [
            {
                "contest_id": problem["contestId"],
                "index": problem["index"],
                "name": problem["name"],
                "type": problem["type"],
                "tags": problem["tags"],
                "points": problem.get("points", None),
                "rating": problem.get("rating", None),
            }
            for problem in problems
        ],
        key=lambda problem: (problem["contest_id"], problem["index"]),
    )

    # indices
    indices = {problem["index"] for problem in problems}
    indices_with_ids = [{"idx": idx, "index": index} for idx, index in enumerate(indices)]
    json_writer.generate_json_by_dict(contents=indices_with_ids, path_from_root=Path("dataset/problem-indices.json"))

    # tags
    tags = get_tag_with_ids(problems=problems)
    json_writer.generate_json_by_dict(
        contents=[dataclasses.asdict(tag) for tag in tags], path_from_root=Path("dataset/problem-tags.json")
    )

    # ratings
    ratings = get_rating_with_ids(problems=problems)
    json_writer.generate_json_by_dict(
        contents=[dataclasses.asdict(rating) for rating in ratings],
        path_from_root=Path("dataset/problem-ratings.json"),
    )

    # problems
    completed_problems = [
        Problem(
            id=idx,
            contest_id=problem["contest_id"],
            index=problem["index"],
            name=problem["name"],
            type=problem["type"],
            tags=[get_tag_by_name(name=tag, tags=tags) for tag in problem["tags"]],
            points=problem["points"],
            rating=get_rating_by_value(value=problem["rating"], ratings=ratings)
            if problem.get("rating") is not None
            else None,
        )
        for idx, problem in enumerate(problems)
    ]
    json_writer.generate_json_by_dict(
        contents=[dataclasses.asdict(problem) for problem in completed_problems],
        path_from_root=Path("dataset/problems.json"),
    )


if __name__ == "__main__":
    run()
