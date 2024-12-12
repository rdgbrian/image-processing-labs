import plotly.graph_objs as go
import networkx as nx
from lab4_fft.fast_fourier_transform import fast_fourier_transform
from lab3_dft.discrete_fourier_transform import create_white_square


n = 8
square_size = 4
image = create_white_square(n, square_size)


temp_fft, info = fast_fourier_transform(image,save_info=True)

def create_node_cord(info):
    parition_string = ""
    parition = info[""]
    
    # nodes_coords = {
    #     0: (0, 0, 0),   # Node 0
    #     1: (1, 0, 0),   # Node 1 (aligned in a straight line with Node 0)
    #     2: (2, 0, 0),   # Node 2 (aligned in a straight line with Node 1)
    #     3: (3, 1, 1),   # Node 3
    #     4: (4, 2, 1)    # Node 4
    # }
    nodes_info = {}
    nodes_coords = {}
    for row in parition:
        for value_info in row:
            nodes_coords[0] = (0,value_info.parition_cord)
            nodes_info[0] = value_info



