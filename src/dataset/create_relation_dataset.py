import sys

sys.path.append("../..")

from pathlib import Path

from src.model.KGAT.dataset import RelationType
from src.utils import json_writer


def run() -> None:
    relations = [{"id": relation.value, "name": relation.name.lower()} for relation in RelationType]
    json_writer.generate_json_by_dict(relations, Path("dataset/relations.json"))


if __name__ == "__main__":
    run()
