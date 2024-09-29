import sys

sys.path.append("../..")

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def generate_json_by_dict(contents: list[dict[str, Any]], path_from_root: Path) -> None:
    """
    Generate the json by dictionary
    -----------
    Parameters:
        contents: list[dict][str, Any]
            The contents to be written into the json file
        path_from_root: str
            The path from the root directory
    --------
    Returns:
        None
    --------
    Example:
        [json]
        {
            "date": "2024-01-01",
            "contents": [
                {
                    "handle": "Radewoosh",
                    "rating": 3759,
                    "maxRating": 3759
                },
                ...
            ]
        }
    """
    today = datetime.now(tz=UTC).date().strftime("%Y-%m-%d")
    contents = [{k: v for k, v in content.items() if v is not None} for content in contents]
    root_path = Path.joinpath(Path.cwd(), "../..")

    with Path(Path.joinpath(root_path, path_from_root)).open("w") as f:
        json.dump({"date": today, "contents": contents}, fp=f, allow_nan=False, indent=2)
