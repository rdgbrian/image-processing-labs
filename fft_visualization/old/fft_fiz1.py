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

# Create edge traces
edge_trace = go.Scatter3d(
    x=[], y=[], z=[],
    line=dict(width=2, color='blue'),
    hoverinfo='none',
    mode='lines'
)

# Create edge label traces
edge_label_trace = go.Scatter3d(
    x=[], y=[], z=[],
    mode='text',
    text=[],
    textposition='middle center',
    hoverinfo='none',
    showlegend=False
)

# Loop through the edges to add to the trace and create labels
for i, edge in enumerate(G.edges()):
    x0, y0, z0 = nodes_coords[edge[0]]
    x1, y1, z1 = nodes_coords[edge[1]]
    edge_trace['x'] += (x0, x1, None)
    edge_trace['y'] += (y0, y1, None)
    edge_trace['z'] += (z0, z1, None)
    
    # Calculate the midpoint of each edge for the label
    mid_x = (x0 + x1) / 2
    mid_y = (y0 + y1) / 2
    mid_z = (z0 + z1) / 2
    edge_label_trace['x'] += (mid_x,)
    edge_label_trace['y'] += (mid_y,)
    edge_label_trace['z'] += (mid_z,)
    
    # Add the label (using the edge index or any label you'd like)
    edge_label_trace['text'] += (f'Edge {edge[0]}-{edge[1]}',)

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

# Create layout and plot
layout = go.Layout(
    showlegend=False,
    scene=dict(xaxis=dict(showbackground=False),
               yaxis=dict(showbackground=False),
               zaxis=dict(showbackground=False)),
    margin=dict(l=0, r=0, b=0, t=0)
)

fig = go.Figure(data=[edge_trace, node_trace, edge_label_trace], layout=layout)
fig.show()
