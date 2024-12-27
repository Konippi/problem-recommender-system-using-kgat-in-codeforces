import random

import matplotlib.pyplot as plt
import networkx as nx

from src.constants import SEED
from src.type import Entity, Triplet


def visualize(
    triplets: list[Triplet],
    entities: list[Entity],
    triplet_num: int | None = None,
    highlight_nodes: list[str] | None = None,
) -> None:
    if triplet_num is not None:
        triplets = random.sample(triplets, min(len(triplets), triplet_num))

    g = nx.Graph()

    entity_map = {entity.id: entity for entity in entities}
    entity_type_label = {
        "problem": "P",
        "contest": "C",
        "contest_division": "D",
        "tag": "T",
        "rating": "R",
    }

    for triplet in triplets:
        h = entity_map[triplet.head]
        t = entity_map[triplet.tail]

        g.add_edge(
            f"{entity_type_label.get(h.target_type, 'Unknown')}{h.target_id}",
            f"{entity_type_label.get(t.target_type, 'Unknown')}{t.target_id}",
            label=triplet.relation,
        )

    plt.figure(figsize=(9, 9))

    edge_labels = nx.get_edge_attributes(g, "label")
    pos = nx.spring_layout(g, seed=SEED, k=0.5)

    # Draw normal nodes
    normal_nodes = [node for node in g.nodes() if highlight_nodes is None or node not in highlight_nodes]
    nx.draw(
        g,
        pos,
        with_labels=True,
        nodelist=normal_nodes,
        node_size=300,
        font_size=4,
        node_color="skyblue",
        edge_color="gray",
        width=0.1,
        alpha=0.3,
    )

    # Draw highlight nodes
    if highlight_nodes:
        nx.draw(
            g,
            pos,
            with_labels=True,
            nodelist=highlight_nodes,
            node_size=500,
            font_size=4,
            node_color="orange",
            edge_color="gray",
            width=0.5,
            alpha=0.6,
        )

    nx.draw_networkx_edge_labels(
        g,
        pos,
        edge_labels=edge_labels,
        font_size=4,
        alpha=0.5,
        bbox={"facecolor": "none", "edgecolor": "none"},
    )

    plt.title("Knowledge Graph")
    plt.show()
