import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import networkx as nx
import numpy as np

# Function to add edges along straight lines in 3D
def add_straight_line_edges(graph, nodes_coords):
    for i in range(len(nodes_coords) - 1):
        graph.add_edge(i, i + 1)  # Connect consecutive nodes

# Define a 3D plot
def plot_graph_3d(graph, pos):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Draw nodes
    x_vals, y_vals, z_vals = zip(*pos.values())
    ax.scatter(x_vals, y_vals, z_vals, c='r', s=100)

    # Draw edges
    for edge in graph.edges():
        x = [pos[edge[0]][0], pos[edge[1]][0]]
        y = [pos[edge[0]][1], pos[edge[1]][1]]
        z = [pos[edge[0]][2], pos[edge[1]][2]]
        ax.plot(x, y, z, c='b')

    plt.show()

# Create a graph
G = nx.Graph()

# Manually specify node coordinates (aligned in a straight line for certain edges)
nodes_coords = {
    0: (0, 0, 0),     # Node 0
    1: (1, 0, 0),     # Node 1 (aligned in x-axis with Node 0)
    2: (2, 0, 0),     # Node 2 (aligned in x-axis with Node 1)
    3: (3, 1, 1),     # Node 3 (different coordinates)
    4: (4, 2, 1)      # Node 4 (connected in a non-straight line)
}

# Add edges in a straight line (0 -> 1 -> 2)
add_straight_line_edges(G, nodes_coords)

# Add other edges
G.add_edge(2, 3)  # Edge from node 2 to 3
G.add_edge(3, 4)  # Edge from node 3 to 4

# Plot the graph
plot_graph_3d(G, nodes_coords)
