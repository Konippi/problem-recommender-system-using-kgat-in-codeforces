import matplotlib.pyplot as plt
import networkx as nx

from src.model.KGAT.model import KGAT
from src.model.KGAT.preprocess import Preprocess


def visualize_attention_scores(
    model: KGAT,
    preprocess: Preprocess,
    user_idx: int,
    problem_indices: list[int],
) -> None:
    G = nx.DiGraph()  # noqa: N806

    G.add_node(f"u{user_idx}", color="red", node_type="user")

    for problem_idx in problem_indices:
        weight = model.attentive_matrix[user_idx, problem_idx].item()
        G.add_node(f"p{problem_idx}", color="blue", node_type="problem")
        G.add_edge(f"u{user_idx}", f"p{problem_idx}", weight=weight)

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, node_color=[node[1]["color"] for node in G.nodes(data=True)])
    nx.draw_networkx_labels(G, pos)

    edges = G.edges()
    weights = [G[u][v]["weight"] for u, v in edges]
    nx.draw_networkx_edges(G, pos, width=weights, edge_color="gray")

    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    for problem_idx in problem_indices:
        relation = preprocess.relation_id_map.get(problem_idx)
        relation_label = relation.name if relation is not None else "Unknown"
        x1, y1 = pos[f"u{user_idx}"]
        x2, y2 = pos[f"p{problem_idx}"]
        plt.text((x1 + x2) / 2, (y1 + y2) / 2, relation_label, fontsize=8, color="green")

    plt.axis("off")
    plt.show()
