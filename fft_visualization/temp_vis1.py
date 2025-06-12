from pyvis.network import Network
import random
from typing import List, Tuple
from flat_vis import order_flat_nd_fft
import numpy as np


def create_visualization(layers, edge_rule=None, spacing=500):
    """
    Visualizes FFT layers using PyVis.
    Each node is labeled with its full tuple: (parity, coord, order).
    Adds butterfly edges based on the given edge_rule and layer stage.
    """
    from pyvis.network import Network

    # Initialize the PyVis Network
    net = Network(height="800px", width="1200px", directed=True)

    total_nodes = 0
    layer_nodes = []
    id_to_node = {}

    max_layer_size = max(len(layer) for layer in layers)
    num_layers = len(layers)
    layer_spacing = int(1.5 * (spacing // num_layers))
    node_spacing = int(1 * (spacing // max_layer_size))

    # Create and label nodes
    for layer_index, layer in enumerate(layers):
        current_layer_ids = []
        layer_x = layer_index * layer_spacing
        layer_y_offset = (max_layer_size - len(layer)) * node_spacing // 2

        for node_index, node in enumerate(layer):
            node_y = node_index * node_spacing + layer_y_offset
            label = str(node)
            net.add_node(
                total_nodes,
                label=label,
                x=layer_x,
                y=node_y,
                physics=False,
                font={"size": 10},
                size=3
            )
            id_to_node[total_nodes] = node
            current_layer_ids.append(total_nodes)
            total_nodes += 1

        layer_nodes.append(current_layer_ids)

    # Add edges between layers
    if edge_rule:
        fft_lengths = [2**(i) for i in range(1,len(layers)+1)]

        for L in range(len(layer_nodes) - 1):
            src_ids = layer_nodes[L]
            tgt_ids = layer_nodes[L + 1]
            fft_length = fft_lengths[L]

            for src_id in src_ids:
                for tgt_id in tgt_ids:
                    src_node = id_to_node[src_id]
                    tgt_node = id_to_node[tgt_id]
                    if edge_rule(src_node, tgt_node, fft_length):
                        net.add_edge(
                            src_id, tgt_id,
                            color={"color": "rgba(0, 0, 0, 0.2)"},
                            arrows={"to": {"enabled": False}},
                            width=1
                        )

    # Final visual tuning
    net.set_options('''
    var options = {
      "physics": {
        "enabled": false
      },
      "edges": {
        "font": {
          "align": "top"
        },
        "arrows": {
          "to": {
            "enabled": true
          }
        }
      }
    }
    ''')

    return net



# Example edge rule: Connect a node to another if the sum of their IDs is even
def fft_edge_rule(src_node, tgt_node, fft_length):
    """
    src_node: (parity_src, coord_src, order_src)
    tgt_node: (parity_tgt, coord_tgt, order_tgt)
    fft_length: scalar length of FFT in any dimension for layer L
    """
    parity_src, coord_src, _ = src_node
    parity_tgt, coord_tgt, _ = tgt_node

    # 1. Parity match: src.parity[1:] == tgt.parity
    if any(p_src[1:] != p_tgt for p_src, p_tgt in zip(parity_src, parity_tgt)):
        return False

    # 2. Coord match mod fft_length
    for c_src, c_tgt in zip(coord_src, coord_tgt):
        if c_src != (c_tgt % fft_length):
            return False

    return True

def create_temp_layers(L, N):
    """
    Create L layers with N nodes each.
    Args:
        L (int): Number of layers.
        N (int): Number of nodes per layer.

    Returns:
        list of lists: Layers with node labels.
    """
    layers = []
    for i in range(L):
        layer = [f"{chr(65 + i)}{j + 1}" for j in range(N)]
        layers.append(layer)
    return layers



def generate_parity_strings(length):
    """
    Generates all combinations of 'e' and 'o' strings of a given length.

    Parameters:
        length (int): Desired length of each string

    Returns:
        List[str]: All combinations like 'eeo', 'eoe', etc.
    """
    from itertools import product
    return [''.join(bits) for bits in product('eo', repeat=length)]

def create_layers(n, d):
    """
    Generates layers of coordinates and order values for an n-level FFT in d dimensions.

    Each layer consists of tuples:
        (parity_str, coordinate, order)

    Layers are sorted by:
        - order value (ascending)
    """
    layers = []

    for i, num_bits in enumerate(reversed(range(1, n + 1))):  # i = 0 to n-1
        order = order_flat_nd_fft(num_bits, d)
        base_layer = [(coord, val) for coord, val in np.ndenumerate(order)]
        base_layer.sort(key=lambda x: x[1])  # sort by order


        if i > 0:
            parity_lists = [generate_parity_strings(i) for _ in range(d)]
            from itertools import product
            parity_combos = list(product(*parity_lists))

            # Right-to-left lex sort on parity_combos
            def right_to_left_parity_sort_key(parity_tuple):
                bit_matrix = [list(p) for p in parity_tuple]
                zipped_by_bit = list(zip(*bit_matrix))
                return zipped_by_bit[::-1]  # reversed for right-to-left

            parity_combos.sort(key=right_to_left_parity_sort_key)
        else:
            parity_combos = [('',) * d]

        # Cartesian product of parity tuples and base layer
        full_layer = [
            (parity_tuple, coord, val)
            for parity_tuple in parity_combos
            for coord, val in base_layer
        ]

        layers.append(full_layer)

    return layers[::-1]  # Reverse the list of layers before returning

# layers = create_layers(3, 2)
# for i, layer in enumerate(layers):
#     print(f"Layer {i+1}")
#     for entry in layer:
#         print(entry)

layers = create_layers(4, 2)


# Create the visualization
net = create_visualization(layers, edge_rule=fft_edge_rule,spacing=1000)
# net = create_visualization(layers, edge_rule=None,spacing=1000)




# Save and display the visualization
# net.show("graph_visualization.html")
net.write_html("graph_visualization.html")

