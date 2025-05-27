import random
import networkx as nx
from scipy.spatial import Delaunay
import math

def create_unit_disk_graph(num_nodes, initial_radius=0.1):
    """
    Creates a connected unit disk graph with position attributes.
    """
    radius = initial_radius
    max_attempts = 100
    attempt = 0
    while attempt < max_attempts:
        initial_positions = {i: (random.random(), random.random()) for i in range(num_nodes)}
        G = nx.Graph()
        for i, pos in initial_positions.items():
            G.add_node(i, pos=pos)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                dist = math.sqrt((initial_positions[i][0] - initial_positions[j][0]) ** 2 +
                                 (initial_positions[i][1] - initial_positions[j][1]) ** 2)
                if dist <= radius:
                    G.add_edge(i, j)
        if nx.is_connected(G):
            return G
        radius += 0.01
        attempt += 1
    raise ValueError("No connected graph found.")

def apply_delaunay_triangulation(G):
    """
    Applies Delaunay triangulation to the stored node positions.
    """
    positions = {i: G.nodes[i]['pos'] for i in G.nodes}
    points = list(positions.values()) 
    tri = Delaunay(points)
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                node1, node2 = int(simplex[i]), int(simplex[j])
                H.add_edge(node1, node2)
    return H

def apply_gabriel_graph(G):
    """
    Applies the Gabriel graph to the stored node positions.
    """
    positions = {i: G.nodes[i]['pos'] for i in G.nodes}
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for i in G.nodes:
        for j in G.nodes:
            if i < j:
                dist = math.sqrt((positions[i][0] - positions[j][0]) ** 2 + (positions[i][1] - positions[j][1]) ** 2)
                midpoint = ((positions[i][0] + positions[j][0]) / 2, (positions[i][1] + positions[j][1]) / 2)
                if dist <= 1.0 and all(
                    math.sqrt((positions[k][0] - midpoint[0]) ** 2 + (positions[k][1] - midpoint[1]) ** 2) >= dist / 2
                    for k in G.nodes if k != i and k != j
                ):
                    H.add_edge(i, j)
                
    return H
