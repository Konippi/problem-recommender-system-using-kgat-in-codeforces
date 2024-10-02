import sys

sys.path.append("../..")

import argparse
import dataclasses
import time
from argparse import Namespace
from datetime import UTC, datetime
from http import HTTPStatus
from logging import INFO, basicConfig, getLogger
from pathlib import Path

import requests
from requests import Session
from tenacity import retry, stop_after_attempt, wait_fixed

from src.constants import HTTP_USER_AGENT
from src.type import Problem, Submission, SubmissionHistory, User
from src.utils import json_loader, json_writer, retry_settings

SM_USER_NUM = 100
RATE_LIMIT_INTERVAL = 1.0
MINIMUM_SUBMISSION_NUM = 10

basicConfig(level=INFO)
logger = getLogger(__name__)


def get_all_users(session: Session) -> list[User]:
    """
    Get All Activated Users in Codeforces

    Codeforces API:
    - API Helop: https://codeforces.com/apiHelp/methods#user.ratedList
    - Endpoint: https://codeforces.com/api/user.ratedList
    -----------
    Parameters:
        None
    --------
    Returns:
        users: list[User]
    --------
    Example:
        [
            {
                "handle": "Radewoosh",
                "rating": 3759,
                "maxRating": 3759
            },
            ...
        ]
    """
    url = "https://codeforces.com/api/user.ratedList"
    params = {"activeOnly": "true", "includeRetired": "false"}
    headers = {"Content-Type": "application/json", "User-Agent": HTTP_USER_AGENT}

    logger.info("GET: %s", url)

    try:
        response = session.get(url=url, params=params, headers=headers, timeout=120)
        response.raise_for_status()
    except requests.HTTPError:
        logger.warning("HTTP Request Error: %d", response.status_code)

    return [
        User(id=idx, handle=user["handle"], rating=user["rating"], max_rating=user["maxRating"])
        for idx, user in enumerate(response.json()["result"])
    ]


@retry(stop=stop_after_attempt(5), wait=wait_fixed(300))
def get_submission_history_by_handle(
    session: Session, handle: str, problem_map: dict[tuple[int, str], Problem]
) -> list[Submission]:
    """
    Get the Submission History by Handle
    -----------
    Parameters:
        handle: str
        problem_map: dict[tuple[int, str], Problem]
    --------
    Returns:
        submission_history: list[Submission]
    --------
    Example:
        [
            {
                "handle": "ecnerwala",
                "submissions": [
                    {
                        "id": 1,
                        "problem": {
                            "id": 1,
                            "contest_id": 1392,
                            "index": "I",
                            "name": "Kevin and Grid",
                            "type": "PROGRAMMING",
                            "points": 4000.0,
                            "rating": {
                                "id": 1,
                                "value": 2000
                            },
                            "tags": [
                                {
                                    "id": 1,
                                    "name": "fft"
                                },
                                {
                                    "id": 2,
                                    "name": "graphs"
                                },
                                {
                                    "id": 3,
                                    "name": "math"
                                },
                            ]
                        },
                        ...
                    },
                    "result": "OK",
                    "created_at": "2024-01-01",
                ]
            },
            ...
        ]

    """
    url = f"https://codeforces.com/api/user.status?handle={handle}"
    params = {
        "handle": handle,
    }
    headers = {"Content-Type": "application/json", "User-Agent": HTTP_USER_AGENT}

    logger.info("GET: %s", url)

    try:
        response = session.get(url=url, params=params, headers=headers, timeout=(180, 180))
        if response.status_code != HTTPStatus.OK:
            logger.warning("HTTP Request Error: %d", response.status_code)
        response.raise_for_status()
    except requests.HTTPError:
        logger.warning("HTTP Request Error: %d", response.status_code)
        if response.status_code == HTTPStatus.BAD_REQUEST:
            logger.warning("User Not Found: %s", handle)
            return []

    try:
        res_json = response.json()
    except requests.exceptions.JSONDecodeError:
        logger.exception("Failed to decode JSON. Response content: %s", response.text)

    return [
        Submission(
            id=idx,
            problem=problem_map[(res["problem"]["contestId"], res["problem"]["index"])],
            result=res["verdict"] if res.get("verdict") is not None else None,
            created_at=str(datetime.fromtimestamp(res["creationTimeSeconds"], tz=UTC)),
        )
        for idx, res in enumerate(res_json["result"])
        if problem_map.get((res["problem"].get("contestId"), res["problem"].get("index"))) is not None
    ]


def run(args: Namespace) -> None:
    # Retry Settings
    session = retry_settings.setup()

    # All Activated Users
    users = get_all_users(session=session)

    # Problems
    problems = [
        Problem(**problem)
        for problem in json_loader.load_contents_of_json(path_from_root=Path("dataset/problems.json"))
    ]
    problem_map = {(problem.contest_id, problem.index): problem for problem in problems}

    all_users = []
    all_submission_history = []
    for user in users:
        submissions = get_submission_history_by_handle(session=session, handle=user.handle, problem_map=problem_map)

        # Filter users who have at least ${MINIMUM_SUBMISSION_NUM}  submissions
        if len(submissions) < MINIMUM_SUBMISSION_NUM:
            continue

        all_users.append(user)
        all_submission_history.append(SubmissionHistory(user=user, submissions=submissions))

        # For Small Dataset
        if len(all_users) == SM_USER_NUM:
            break

        time.sleep(RATE_LIMIT_INTERVAL)

    # Write to JSON
    json_writer.generate_json_by_dict(
        contents=[dataclasses.asdict(user) for user in all_users],
        path_from_root=Path(f"dataset/users{'-sm' if args.sm else ''}.json"),
    )
    json_writer.generate_json_by_dict(
        contents=[dataclasses.asdict(submission_history) for submission_history in all_submission_history],
        path_from_root=Path(f"dataset/users-submission-history{'-sm' if args.sm else ''}.json"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sm", help="for creating small dataset", action="store_true")
    args = parser.parse_args()

    run(args=args)
