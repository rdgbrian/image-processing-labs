import matplotlib.pyplot as plt
import networkx as nx
import itertools

def create_edge_name(node1_label, node2_label):
    """Generate a custom edge label."""
    return f"{node1_label}-{node2_label}"

def custom_edge_rule(layer1, layer2):
    """Define how nodes from layer1 connect to layer2. This connects every node in layer1 to all nodes in layer2."""
    edges = []
    for node1 in layer1:
        for node2 in layer2:
            edge_name = create_edge_name(node1, node2)
            edges.append((node1, node2, edge_name))
    return edges

def plot_nn_graph(layers):
    """Plot a neural network-style graph given layers of nodes."""
    G = nx.DiGraph()
    pos = {}

    # Build nodes and positions
    x_spacing = 10
    y_spacing = 10
    for i, layer in enumerate(layers):
        y_positions = range(len(layer))
        for j, label in enumerate(layer):
            pos[label] = (i * x_spacing, -j * y_spacing)
            G.add_node(label)

    # Build edges using the custom rule
    for i in range(len(layers) - 1):
        edges = custom_edge_rule(layers[i], layers[i + 1])
        for node1, node2, edge_name in edges:
            G.add_edge(node1, node2, label=edge_name)

    # Plot graph
    fig, ax = plt.subplots(figsize=(10, 6))
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color="lightblue", font_size=10, font_weight="bold", ax=ax)

    # Add edge labels
    edge_labels = {(u, v): d['label'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8, ax=ax)

    plt.title("Neural Network Graph Visualization")
    plt.show()

# Example usage
layers = [
    ["Input1", "Input2", "Input3"],
    ["Hidden1", "Hidden2", "Hidden3"],
    ["Output1", "Output2","Output3"],
    ["a1", "a2","a3"],
    ["b1", "b2","b3"],

]

plot_nn_graph(layers)
