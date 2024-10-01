import random

import matplotlib.pyplot as plt
import networkx as nx

from src.constants import SEED
from src.model.KGAT.dataset import Entity, Triplet


def visualize_kg(triplets: list[Triplet], entities: list[Entity]) -> None:
    triplets = random.sample(triplets, min(len(triplets), 300))

    g = nx.Graph()

    entity_map = {entity.id: entity for entity in entities}
    entity_type_label = {
        "problem": "P",
        "contest_division": "D",
        "tag": "T",
        "rating": "R",
    }

    for triplet in triplets:
        # relation_name = next(
        #     (relation.name for relation in RelationType if relation.value == triplet.relation), "Unknown"
        # )
        h = entity_map[triplet.head]
        t = entity_map[triplet.tail]

        g.add_edge(
            f"{entity_type_label.get(h.target_type, "Unknown")}{h.target_id}",
            f"{entity_type_label.get(t.target_type, "Unknown")}{t.target_id}",
            label=triplet.relation,
        )

    plt.figure(figsize=(9, 9))

    edge_labels = nx.get_edge_attributes(g, "label")
    pos = nx.spring_layout(g, seed=SEED, k=0.5)
    nx.draw(
        g,
        pos,
        with_labels=True,
        node_size=250,
        font_size=4,
        node_color="skyblue",
        edge_color="gray",
        width=0.1,
        alpha=0.8,
    )
    nx.draw_networkx_edge_labels(
        g,
        pos,
        edge_labels=edge_labels,
        font_size=4,
        alpha=0.5,
    )

    plt.title("Knowledge Graph")
    plt.show()
