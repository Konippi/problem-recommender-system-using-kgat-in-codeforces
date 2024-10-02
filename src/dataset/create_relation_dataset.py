import sys

sys.path.append("../..")

from pathlib import Path

from src.type import RelationType
from src.utils import json_writer


def run() -> None:
    relations = [{"id": relation.value, "name": relation.name.lower()} for relation in RelationType]
    json_writer.generate_json_by_dict(
        contents=relations,
        path_from_root=Path(
            "dataset/relations.json",
        ),
    )


if __name__ == "__main__":
    run()
