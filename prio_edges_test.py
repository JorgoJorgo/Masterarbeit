import networkx as nx
import math
import matplotlib.pyplot as plt
import random

# Helper function to calculate the angle between two coordinates
def calculate_angle(pos1, pos2):
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    return math.atan2(dy, dx)  # Returns angle in radians

# Helper function to prioritize edges based on the previous edge
def prioritize_edges(edges, previous_edge, tree):
    previous_angle = calculate_angle(tree.nodes[previous_edge[0]]['pos'], tree.nodes[previous_edge[1]]['pos'])

    def angle_difference(angle):
        diff = angle - previous_angle
        return (diff + math.pi) % (2 * math.pi) - math.pi  # Normalize to [-pi, pi]

    # Sort edges based on the smallest angular difference
    sorted_edges = [edge for edge in edges if not (
        edge[0][0] == previous_edge[1] and edge[0][1] == previous_edge[0]
    )]  # Exclude reverse previous edge from sorting
    sorted_edges.sort(key=lambda x: angle_difference(x[1]))

    # Add the reverse of the previous edge with the lowest priority
    reverse_edge = (previous_edge[1], previous_edge[0])
    if reverse_edge in tree.edges:
        reverse_angle = calculate_angle(tree.nodes[reverse_edge[0]]['pos'], tree.nodes[reverse_edge[1]]['pos'])
        sorted_edges.append(((reverse_edge[0], reverse_edge[1]), reverse_angle))

    return sorted_edges

# Helper function to get edges sorted in clockwise order
def get_sorted_edges(node, tree, fails, previous_edge):
    edges = []
    node_pos = tree.nodes[node]['pos']

    for neighbor in tree.neighbors(node):
        edge = (node, neighbor) if node < neighbor else (neighbor, node)  # Ensure undirected edge representation
        if edge not in fails:  # Exclude edges in fails
            neighbor_pos = tree.nodes[neighbor]['pos']
            angle = calculate_angle(node_pos, neighbor_pos)
            edges.append((edge, angle))

    if previous_edge is not None:
        edges = prioritize_edges(edges, previous_edge, tree)

    return [e[0] for e in edges]

# Function to draw the graph with highlighted edges
def draw_graph(graph, previous_edge, next_edge):
    pos = nx.get_node_attributes(graph, 'pos')
    plt.figure(figsize=(10, 8))
    nx.draw(graph, pos, with_labels=True, node_color='lightblue', node_size=500, font_size=10, font_weight='bold')

    # Separate all edges excluding highlighted ones
    all_edges = set(graph.edges())
    highlighted_edges = {previous_edge, next_edge}
    normal_edges = list(all_edges - highlighted_edges)

    # Draw normal edges in black
    nx.draw_networkx_edges(graph, pos, edgelist=normal_edges, edge_color='black', width=1)

    # Highlight the previous edge in yellow
    if previous_edge in graph.edges:
        nx.draw_networkx_edges(graph, pos, edgelist=[previous_edge], edge_color='yellow', width=2)

    # Highlight the next edge in green
    if next_edge in graph.edges:
        nx.draw_networkx_edges(graph, pos, edgelist=[next_edge], edge_color='green', width=2)

    plt.show()

# Manually create a star graph with 8 nodes (1 center and 7 outer nodes)
def generate_star_graph():
    graph = nx.Graph()
    center = 0
    outer_nodes = range(1, 8)

    # Add edges from the center to all outer nodes
    for node in outer_nodes:
        graph.add_edge(center, node)

    # Assign manual positions for visualization
    pos = {
        0: (0, 0),
        1: (1, 0),
        2: (0.7, 0.7),
        3: (0, 1),
        4: (-0.7, 0.7),
        5: (-1, 0),
        6: (-0.7, -0.7),
        7: (0, -1),
    }
    nx.set_node_attributes(graph, pos, "pos")
    return graph

# Generate the manually created star graph
graph = generate_star_graph()

# Get all edges and set the first edge as the starting edge
all_edges = list(graph.edges)
previous_edge_index = 0

# Define fails set (edges to exclude)
fails = {(0, 5), (0, 7)}  # Example failed edges

# Loop to plot and move to the next edge
for i in range(len(all_edges)):
    previous_edge = (i + 1, 0)

    # Get the sorted edges and identify the next edge
    sorted_edges = get_sorted_edges(previous_edge[1], graph, fails, previous_edge)
    next_edge = sorted_edges[0] if sorted_edges else None

    print("Current edge:", previous_edge)
    print("Sorted edges (excluding fails):", sorted_edges)

    # Draw the graph with highlighted edges
    draw_graph(graph, previous_edge, next_edge)

    # Move to the next edge in the list
    previous_edge_index = (previous_edge_index + 1) % len(all_edges)
