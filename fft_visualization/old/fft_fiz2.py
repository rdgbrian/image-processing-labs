import plotly.graph_objs as go
import networkx as nx
from src.fast_fourier_transform import fft2d
from src.discrete_fourier_transform import create_white_square
import numpy as np


n = 8
square_size = 4
image = create_white_square(n, square_size)

temp_fft, info = fft2d(image,save_info=True)



def es_os_to_binary(s):
    # Replace 'e' with '0' and 'o' with '1'
    if s == '':
        return 0
    binary_str = s.replace('e', '0').replace('o', '1')
    
    # Convert the binary string to an integer
    decimal_value = int(binary_str, 2)
    
    return decimal_value

def flatten_list(nested_list):
    flat_list = []
    for item in nested_list:
        if isinstance(item, list):  # Check if the item is a list
            flat_list.extend(flatten_list(item))  # Recursively flatten the sublist
        else:
            flat_list.append(item)  # Append non-list items directly
    return flat_list

def parition_to_cord(partition_str,xr,yr,N,M):
    partition_string_x = partition_str[::2]
    partition_string_y = partition_str[1::2]
    decimal_value_x = es_os_to_binary(partition_string_x)
    decimal_value_y = es_os_to_binary(partition_string_y)
    num_y_part = 2**(len(partition_string_y))
    num_x_part = 2**(len(partition_string_x))

    x_cord = decimal_value_x * M//num_x_part + xr
    y_cord = decimal_value_y * N//num_y_part + yr
    z_cord = 3 * len(partition_str)//2

    return (x_cord,z_cord,y_cord)


temp = list(info.partitions.values())
temp = flatten_list(temp)

nodes_coords = {}
frag_to_idx = {}

N = image.shape[0]
M = image.shape[1]



for i, partition_info in enumerate(temp):

    # print(partition_info.value)

    frag_to_idx[partition_info] = i

    nodes_coords[i] = parition_to_cord(partition_info.partition_string,
                                       partition_info.relative_cord[1],
                                       partition_info.relative_cord[0],
                                       N,M)
    
G = nx.Graph()
for i, partition_info in enumerate(temp):
    if partition_info.dependants != None:
        for dep in partition_info.dependants:
            G.add_edge(i, frag_to_idx[dep])


# # Add edges (aligned in a straight line for 0 -> 1 -> 2)
# G.add_edge(0, 1)
# G.add_edge(1, 2)
# G.add_edge(2, 3)
# G.add_edge(3, 4)

# Create edge traces (lines connecting nodes)
edge_trace = go.Scatter3d(
    x=[], y=[], z=[],
    line=dict(width=2, color='blue'),
    hoverinfo='none',
    mode='lines')

# Create edge label trace (to show edge labels)
edge_label_trace = go.Scatter3d(
    x=[], y=[], z=[],
    mode='text',
    text=[],
    textposition='middle center',
    hoverinfo='none'
)

# Arbitrary labels for each edge
edge_labels = {
    (0, 1): "Edge A",
    (1, 2): "Edge B",
    (2, 3): "Edge C",
    (3, 4): "Edge D"
}

# Add coordinates for edges and their labels
for edge in G.edges():
    x0, y0, z0 = nodes_coords[edge[0]]
    x1, y1, z1 = nodes_coords[edge[1]]
    
    # Add edge coordinates
    edge_trace['x'] += (x0, x1, None)
    edge_trace['y'] += (y0, y1, None)
    edge_trace['z'] += (z0, z1, None)
    
    # Calculate midpoint for edge label placement
    mid_x = (x0 + x1) / 2
    mid_y = (y0 + y1) / 2
    mid_z = (z0 + z1) / 2
    
    # # Add edge label at midpoint
    # edge_label_trace['x'] += (mid_x,)
    # edge_label_trace['y'] += (mid_y,)
    # edge_label_trace['z'] += (mid_z,)
    # edge_label_trace['text'] += (edge_labels[edge],)

# Create node traces (nodes on the graph)
node_trace = go.Scatter3d(
    x=[pos[0] for pos in nodes_coords.values()],
    y=[pos[1] for pos in nodes_coords.values()],
    z=[pos[2] for pos in nodes_coords.values()],
    mode='markers',
    marker=dict(size=10, color='red'),
    text=list(nodes_coords.keys()),
    hoverinfo='text'
)

# Create layout and plot (without axis numbers)
layout = go.Layout(
    showlegend=True,
    # scene=dict(
    #     xaxis=dict(showbackground=False, showticklabels=False, tickvals=[]),  # Hide x-axis numbers
    #     yaxis=dict(showbackground=False, showticklabels=False, tickvals=[]),  # Hide y-axis numbers
    #     zaxis=dict(showbackground=False, showticklabels=False, tickvals=[])   # Hide z-axis numbers
    # ),
    margin=dict(l=0, r=0, b=0, t=0)
)

# Create figure with node, edge, and edge label traces
fig = go.Figure(data=[edge_trace, node_trace, edge_label_trace], layout=layout)
fig.show()
