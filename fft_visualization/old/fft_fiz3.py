import plotly.graph_objs as go
import networkx as nx
from lab4_fft.fast_fourier_transform import fast_fourier_transform
from lab3_dft.discrete_fourier_transform import create_white_square
import numpy as np

n = 4
square_size = 4
image = create_white_square(n, square_size)

temp_fft, info = fast_fourier_transform(image, save_info=True)

def es_os_to_binary(s):
    if s == '':
        return 0
    binary_str = s.replace('e', '0').replace('o', '1')
    decimal_value = int(binary_str, 2)
    return decimal_value

def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):
            flat_list.extend(flatten_list(item))
        else:
            flat_list.append(item)
    return flat_list

def parition_to_cord(partition_str, xr, yr, N, M):
    partition_string_x = partition_str[::2]
    partition_string_y = partition_str[1::2]
    decimal_value_x = es_os_to_binary(partition_string_x)
    decimal_value_y = es_os_to_binary(partition_string_y)
    num_y_part = 2**(len(partition_string_y))
    num_x_part = 2**(len(partition_string_x))

    x_cord = decimal_value_x * M // num_x_part + xr
    y_cord = decimal_value_y * N // num_y_part + yr
    z_cord = 3 * len(partition_str) // 2

    return (x_cord, z_cord, y_cord)

temp = list(info.partitions.values())
temp = flatten_list(temp)

nodes_coords = {}
frag_to_idx = {}

N = image.shape[0]
M = image.shape[1]

for i, partition_info in enumerate(temp):
    frag_to_idx[partition_info] = i
    nodes_coords[i] = parition_to_cord(partition_info.partition_string,
                                       partition_info.relative_cord[1],
                                       partition_info.relative_cord[0],
                                       N, M)

G = nx.Graph()
for i, partition_info in enumerate(temp):
    if partition_info.dependants is not None:
        for dep in partition_info.dependants:
            G.add_edge(i, frag_to_idx[dep])

# Create edge traces
edge_trace = go.Scatter3d(
    x=[], y=[], z=[],
    line=dict(width=2, color='blue'),
    hoverinfo='none',
    mode='lines')

# Add coordinates for edges
for edge in G.edges():
    x0, y0, z0 = nodes_coords[edge[0]]
    x1, y1, z1 = nodes_coords[edge[1]]
    
    # Add edge coordinates
    edge_trace['x'] += (x0, x1, None)
    edge_trace['y'] += (y0, y1, None)
    edge_trace['z'] += (z0, z1, None)

# Create node traces
node_trace = go.Scatter3d(
    x=[pos[0] for pos in nodes_coords.values()],
    y=[pos[1] for pos in nodes_coords.values()],
    z=[pos[2] for pos in nodes_coords.values()],
    mode='markers',
    marker=dict(size=10, color='red'),
    text=list(nodes_coords.keys()),
    hoverinfo='text'
)

# Create layout
layout = go.Layout(
    showlegend=True,
    hovermode='closest',
    margin=dict(l=0, r=0, b=0, t=0)
)

# Add hover effect to show connected nodes and edges
def get_connected_nodes_edges(node_idx):
    # Get the nodes and edges connected to the hovered node
    sub_nodes = set([node_idx])
    sub_edges = set()
    
    # Recursively find all connected nodes and edges
    def explore(node):
        neighbors = list(G.neighbors(node))
        sub_nodes.update(neighbors)
        for neighbor in neighbors:
            sub_edges.add((node, neighbor))
            explore(neighbor)

    explore(node_idx)
    return sub_nodes, sub_edges

# Generate update for hover functionality
def update_on_hover(trace, points, state):
    hover_node_idx = points.point_inds[0]
    
    sub_nodes, sub_edges = get_connected_nodes_edges(hover_node_idx)
    
    # Update edge and node visibility based on hover
    with fig.batch_update():
        # Update edges
        edge_trace.x = []
        edge_trace.y = []
        edge_trace.z = []
        
        for edge in sub_edges:
            x0, y0, z0 = nodes_coords[edge[0]]
            x1, y1, z1 = nodes_coords[edge[1]]
            edge_trace.x += (x0, x1, None)
            edge_trace.y += (y0, y1, None)
            edge_trace.z += (z0, z1, None)
        
        # Update nodes
        node_trace.x = [nodes_coords[i][0] for i in sub_nodes]
        node_trace.y = [nodes_coords[i][1] for i in sub_nodes]
        node_trace.z = [nodes_coords[i][2] for i in sub_nodes]

# Assign the hover event callback
fig = go.Figure(data=[edge_trace, node_trace], layout=layout)
fig.data[1].on_hover(update_on_hover)

# Reset when not hovering over any node
def reset_on_unhover(trace, points, state):
    with fig.batch_update():
        # Reset edges
        edge_trace.x = []
        edge_trace.y = []
        edge_trace.z = []
        
        for edge in G.edges():
            x0, y0, z0 = nodes_coords[edge[0]]
            x1, y1, z1 = nodes_coords[edge[1]]
            edge_trace.x += (x0, x1, None)
            edge_trace.y += (y0, y1, None)
            edge_trace.z += (z0, z1, None)

        # Reset nodes
        node_trace.x = [pos[0] for pos in nodes_coords.values()]
        node_trace.y = [pos[1] for pos in nodes_coords.values()]
        node_trace.z = [pos[2] for pos in nodes_coords.values()]

fig.data[1].on_unhover(reset_on_unhover)

# Show figure
fig.show()
