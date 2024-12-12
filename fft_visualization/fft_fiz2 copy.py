import plotly.graph_objs as go
import networkx as nx

# Create a graph
G = nx.Graph()

# Manually define node positions to control angles
nodes_coords = {
    0: (0, 0, 0),   # Node 0
    1: (1, 0, 0),   # Node 1 (aligned in a straight line with Node 0)
    2: (2, 0, 0),   # Node 2 (aligned in a straight line with Node 1)
    3: (3, 1, 1),   # Node 3
    4: (4, 2, 1)    # Node 4
}

# Add edges (aligned in a straight line for 0 -> 1 -> 2)
G.add_edge(0, 1)
G.add_edge(1, 2)
G.add_edge(2, 3)
G.add_edge(3, 4)

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
    
    # Add edge label at midpoint
    edge_label_trace['x'] += (mid_x,)
    edge_label_trace['y'] += (mid_y,)
    edge_label_trace['z'] += (mid_z,)
    edge_label_trace['text'] += (edge_labels[edge],)

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
