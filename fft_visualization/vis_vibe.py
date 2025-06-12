import itertools
from typing import List, Tuple
from flat_vis import order_flat_nd_fft
import numpy as np

def generate_fft_layers(n: int, d: int) -> Tuple[List[List[str]], List[np.ndarray]]:
    """
    Generates labeled layers and their corresponding index maps for FFT visualization.
    
    Parameters:
    - n: FFT depth (log2 of dimension size)
    - d: number of dimensions

    Returns:
    - layers: list of lists of node labels (e.g., F0, F1, ...)
    - orderings: list of ndarray containing the positions and flattened indices
    """
    layers = []
    orderings = []

    for i in range(1,n + 1):  # From level 0 (leaves) to level n (final FFT)
        order = order_flat_nd_fft(i, d)
        flat_indices = order.flatten()
        labels = [f"F{i}_{idx}" for idx in flat_indices]
        layers.append(labels)
        orderings.append(order)

    return layers, orderings

def fft_edge_rule_from_orderings(orderings: List[np.ndarray]) -> callable:
    """
    Returns an edge_rule function that connects 2^d children to their parent node,
    based on the recursive FFT structure.

    Parameters:
    - orderings: list of ndarrays returned by generate_fft_layers()

    Returns:
    - edge_rule: function used in PyVis that maps (src_id, tgt_id) to bool
    """
    # Create a reverse map: flat_id -> coordinate for each layer
    coord_maps = [{v: idx for idx, v in np.ndenumerate(order)} for order in orderings]

    def edge_rule(src: int, tgt: int) -> bool:
        # Determine which layer each node is in
        for i in range(len(orderings) - 1):
            if src in coord_maps[i] and tgt in coord_maps[i + 1]:
                src_coord = coord_maps[i][src]
                tgt_coord = coord_maps[i + 1][tgt]

                # Check if src is one of the 2^d children of tgt
                match = True
                for s, t in zip(src_coord, tgt_coord):
                    if s != t * 2 and s != t * 2 + 1:
                        match = False
                        break
                return match
        return False

    return edge_rule

# Now generate the layers and edge rule
layers, orderings = generate_fft_layers(n=2, d=2)
edge_rule = fft_edge_rule_from_orderings(orderings)


print(layers)
print(orderings)


# Pass these into PyVis visualization
import pyvis
from pyvis.network import Network

def create_visualization_fft(layers, edge_rule, spacing=800):
    net = Network(height="900px", width="1400px", directed=True)
    total_nodes = 0
    layer_nodes = []

    max_layer_size = max(len(layer) for layer in layers)
    num_layers = len(layers)
    layer_spacing = int(1.5 * (spacing // num_layers))
    num_nodes = len(layers[0])
    node_spacing = int(1 * (spacing // num_nodes))

    node_positions = {}  # Keep track of node ID positions for layout

    for layer_index, layer in enumerate(layers):
        current_layer_ids = []
        layer_x = layer_index * layer_spacing
        layer_y_offset = (max_layer_size - len(layer)) * node_spacing // 2

        for node_index, node_label in enumerate(layer):
            node_y = node_index * node_spacing + layer_y_offset
            net.add_node(total_nodes, label=node_label, x=layer_x, y=node_y,
                         physics=False, font={"size": 12}, size=5)
            node_positions[node_label] = total_nodes
            current_layer_ids.append(total_nodes)
            total_nodes += 1

        layer_nodes.append(current_layer_ids)

    for i in range(len(layer_nodes) - 1):
        current_layer = layer_nodes[i]
        next_layer = layer_nodes[i + 1]
        for src in current_layer:
            for tgt in next_layer:
                if edge_rule(src, tgt):
                    net.add_edge(src, tgt, color={"color": "rgba(0,0,0,0.15)"}, width=1)

    net.set_options('''
    var options = {
      "physics": { "enabled": false },
      "edges": {
        "font": { "align": "top" },
        "arrows": { "to": { "enabled": true } }
      }
    }''')

    return net

# Create and write the final visual
net_fft = create_visualization_fft(layers, edge_rule)
net_fft.write_html("fft_butterfly_visualization.html")
