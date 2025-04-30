import json
import networkx as nx


def build_graph_from_terminals(component_file, terminal_net_file):
    with open(component_file) as f:
        components = json.load(f)

    with open(terminal_net_file) as f:
        terminal_map = json.load(f)

    G = nx.Graph()

    # Add components as nodes
    for i, comp in enumerate(components):
        comp_id = f"{comp['label'][0]}{i+1}"
        G.add_node(comp_id, type=comp['label'])

    # Map nets and build edges
    nets = {}
    for comp_id, pins in terminal_map.items():
        for term, net in pins.items():
            nets.setdefault(net, []).append((comp_id, term))

    for net, connected in nets.items():
        for i in range(len(connected)):
            for j in range(i + 1, len(connected)):
                src, _ = connected[i]
                dst, _ = connected[j]
                if src != dst:
                    G.add_edge(src, dst, net=net)

    return G

import matplotlib.pyplot as plt
def draw_graph(G, title="Circuit Graph"):
    plt.figure(figsize=(8, 6))
    pos = nx.spring_layout(G, seed=42)
    node_labels = {n: f"{n}\n{G.nodes[n]['type']}" for n in G.nodes}
    edge_labels = nx.get_edge_attributes(G, 'net')

    nx.draw(G, pos, with_labels=True, labels=node_labels,
            node_size=2000, node_color='skyblue', font_size=8)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7)
    plt.title(title)
    plt.tight_layout()
    plt.show()

