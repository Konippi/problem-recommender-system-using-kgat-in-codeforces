import sys

sys.path.append("../..")

import json
from pathlib import Path
from typing import Any


def load_contents_of_json(path_from_root: Path) -> list[dict[str, Any]]:
    """
    Load the contents of json
    -----------
    Parameters:
        path_from_root: str
            The path from the root directory
    --------
    Returns:
        contents: list[dict]
            The contents of the json file
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
    root_path = Path.joinpath(Path.cwd(), "../..")

    with Path(Path.joinpath(root_path, path_from_root)).open("r") as f:
        return json.load(f)["contents"]  # type: ignore[no-any-return]
