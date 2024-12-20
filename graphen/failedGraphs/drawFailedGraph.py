import networkx as nx
import matplotlib.pyplot as plt

def visualize_graph(graph_file, fails):
    """
    Reads a DOT file to create the graph and visualizes it with highlighted fail edges.

    Args:
        graph_file (str): The path to the DOT file containing the graph.
        fails (dict): Dictionary of fail edges. Keys are tuples representing edges, values are weights or any indicator.

    Returns:
        None
    """
    # Load the graph from the DOT file
    G = nx.DiGraph(nx.nx_agraph.read_dot(graph_file))

    for edge in fails:
        G.add_edge(edge[0],edge[1])

    # Extract positions (if provided in the DOT file)
    pos = {}
    # for node in G.nodes(data=True):
    #     if 'pos' in node[1]:
    #         x, y = map(float, node[1]['pos'].strip('()').split(','))
    #         pos[node[0]] = (x, y)

    # Assign default positions if not present in the DOT file
    if not pos:
        pos = nx.spring_layout(G)

    # Prepare edge colors and weights
    edge_colors = []
    for edge in G.edges():
        # Ensure fails can be matched as (u, v) or (v, u)
        if edge in fails or (edge[1], edge[0]) in fails:
            edge_colors.append('red')  # Highlight failed edges in red
        else:
            edge_colors.append('black')  # Normal edges in black

    # Draw the graph
    plt.figure(figsize=(12, 8))
    nx.draw(
        G,
        pos,
        with_labels=True,
        node_color='lightblue',
        edge_color=edge_colors,
        node_size=500,
        font_size=8,
        arrows=True,
    )

    # Add legend for fails
    legend_labels = ['Fail Edges', 'Normal Edges']
    legend_colors = ['red', 'black']
    for color, label in zip(legend_colors, legend_labels):
        plt.plot([], [], color=color, label=label)

    plt.legend(loc='upper right')
    plt.title("Graph Visualization with Fail Edges Highlighted")
    plt.show()


if __name__ == "__main__":
    # Input DOT file name
    graph_file = "graph"  # Replace with your graph file name

    # Example fails dictionary
    fails = {
        (35, 33): -1, (11, 79): -1, (14, 55): -1, (60, 16): -1, (65, 20): -1, (65, 49): -1, (19, 60): -1, (3, 12): -1, (69, 72): -1, (1, 32): -1, (56, 12): -1, (28, 25): -1, (72, 60): -1, (36, 48): -1, (79, 60): -1, (74, 66): -1, (74, 23): -1, (4, 1): -1, (25, 5): -1, (66, 74): -1, (73, 0): -1, (40, 74): -1, (39, 58): -1, (19, 59): -1, (6, 38): -1, (10, 39): -1, (70, 2): -1, (41, 66): -1, (65, 77): -1, (13, 3): -1, (64, 14): -1, (76, 78): -1, (42, 74): -1, (7, 23): -1, (66, 62): -1, (76, 66): -1, (71, 37): -1, (24, 64): -1, (9, 43): -1, (22, 68): -1, (75, 25): -1, (24, 15): -1, (21, 7): -1, (50, 73): -1, (33, 69): -1, (43, 63): -1, (63, 75): -1, (9, 71): -1, (0, 45): -1, (40, 41): -1, (60, 72): -1, (34, 58): -1, (72, 33): -1, (60, 11): -1, (75, 17): -1, (11, 22): -1, (63, 2): -1, (2, 63): -1, (17, 10): -1, (72, 19): -1, (67, 51): -1, (14, 64): -1, (77, 20): -1, (20, 27): -1, (4, 78): -1, (15, 25): -1, (54, 37): -1, (16, 55): -1, (15, 75): -1, (76, 62): -1, (9, 57): -1, (23, 46): -1, (79, 11): -1, (35, 27): -1, (64, 52): -1, (61, 3): -1, (36, 38): -1, (47, 21): -1, (77, 38): -1, (10, 17): -1, (1, 4): -1, (56, 45): -1, (62, 29): -1, (47, 10): -1, (3, 32): -1, (63, 43): -1, (59, 11): -1, (18, 21): -1, (18, 1): -1, (10, 15): -1, (31, 46): -1, (2, 54): -1, (21, 47): -1, (26, 20): -1, (51, 67): -1, (8, 54): -1, (47, 18): -1, (16, 60): -1, (75, 10): -1, (41, 1): -1, (22, 38): -1, (75, 63): -1, (13, 56): -1, (5, 67): -1, (47, 39): -1, (41, 40): -1, (79, 30): -1, (17, 47): -1, (39, 20): -1, (25, 51): -1, (74, 29): -1, (6, 36): -1, (56, 36): -1, (14, 33): -1, (27, 34): -1, (35, 65): -1, (59, 72): -1, (20, 39): -1, (45, 0): -1, (46, 31): -1, (43, 28): -1, (18, 78): -1, (17, 63): -1, (12, 3): -1, (10, 47): -1, (0, 48): -1, (71, 51): -1, (10, 52): -1, (32, 3): -1, (3, 45): -1, (9, 70): -1, (32, 40): -1, (45, 13): -1, (52, 15): -1, (60, 19): -1, (30, 51): -1, (46, 23): -1, (45, 56): -1, (43, 70): -1, (25, 75): -1, (61, 12): -1, (73, 39): -1, (32, 45): -1, (11, 68): -1, (51, 53): -1, (24, 5): -1, (67, 30): -1, (9, 37): -1, (51, 28): -1, (37, 9): -1, (40, 32): -1, (59, 69): -1, (78, 66): -1, (78, 4): -1, (71, 28): -1, (29, 74): -1, (1, 78): -1, (77, 26): -1, (78, 1): -1, (21, 76): -1, (6, 56): -1, (6, 12): -1, (16, 24): -1, (58, 39): -1, (30, 79): -1, (42, 41): -1, (49, 69): -1, (12, 61): -1, (20, 26): -1, (22, 65): -1, (18, 47): -1, (74, 40): -1, (52, 64): -1, (68, 59): -1, (65, 69): -1, (3, 40): -1, (75, 15): -1, (40, 1): -1, (75, 43): -1, (23, 29): -1, (24, 30): -1, (13, 61): -1, (44, 18): -1, (34, 14): -1, (5, 30): -1, (76, 7): -1, (70, 63): -1, (12, 13): -1, (20, 0): -1, (67, 5): -1, (33, 60): -1, (76, 21): -1, (65, 35): -1, (18, 44): -1, (7, 46): -1, (22, 6): -1, (30, 24): -1, (68, 65): -1, (49, 65): -1, (34, 52): -1, (70, 9): -1, (78, 44): -1, (52, 24): -1, (71, 53): -1, (65, 38): -1, (43, 9): -1, (7, 76): -1, (18, 50): -1, (15, 52): -1, (40, 3): -1, (21, 18): -1, (66, 41): -1, (19, 72): -1, (68, 22): -1, (66, 76): -1, (31, 54): -1, (63, 17): -1, (8, 63): -1, (69, 49): -1, (10, 75): -1, (48, 56): -1, (55, 33): -1, (55, 14): -1, (56, 48): -1, (33, 49): -1, (66, 4): -1, (73, 45): -1, (34, 39): -1, (47, 63): -1, (79, 24): -1, (79, 16): -1, (38, 22): -1, (64, 55): -1, (8, 21): -1, (33, 14): -1, (54, 46): -1, (63, 8): -1, (57, 2): -1, (55, 16): -1, (26, 0): -1, (45, 3): -1, (3, 61): -1, (13, 12): -1, (24, 55): -1, (37, 71): -1, (28, 43): -1, (66, 42): -1, (69, 59): -1, (53, 71): -1, (76, 44): -1, (54, 57): -1, (78, 76): -1, (56, 13): -1, (35, 20): -1, (26, 38): -1, (33, 35): -1, (20, 65): -1, (72, 69): -1, (32, 1): -1, (12, 56): -1, (25, 28): -1, (48, 36): -1, (60, 79): -1, (23, 74): -1, (5, 25): -1, (0, 73): -1, (59, 19): -1, (38, 6): -1, (39, 10): -1, (2, 70): -1, (77, 65): -1, (3, 13): -1, (74, 42): -1, (23, 7): -1, (62, 66): -1, (64, 24): -1, (15, 24): -1, (7, 21): -1, (73, 50): -1, (69, 33): -1, (71, 9): -1, (58, 34): -1, (33, 72): -1, (11, 60): -1, (17, 75): -1, (22, 11): -1, (20, 77): -1, (27, 20): -1, (25, 15): -1, (37, 54): -1, (62, 76): -1, (57, 9): -1, (27, 35): -1, (38, 36): -1, (38, 77): -1, (29, 62): -1, (11, 59): -1, (1, 18): -1, (15, 10): -1, (54, 2): -1, (54, 8): -1, (1, 41): -1, (39, 47): -1, (47, 17): -1, (51, 25): -1, (36, 6): -1, (36, 56): -1, (34, 27): -1, (72, 59): -1, (78, 18): -1, (48, 0): -1, (51, 71): -1, (52, 10): -1, (13, 45): -1, (51, 30): -1, (70, 43): -1, (39, 73): -1, (45, 32): -1, (68, 11): -1, (53, 51): -1, (5, 24): -1, (30, 67): -1, (28, 51): -1, (66, 78): -1, (28, 71): -1, (26, 77): -1, (56, 6): -1, (12, 6): -1, (24, 16): -1, (41, 42): -1, (65, 22): -1, (59, 68): -1, (69, 65): -1, (1, 40): -1, (43, 75): -1, (29, 23): -1, (61, 13): -1, (14, 34): -1, (30, 5): -1, (63, 70): -1, (0, 20): -1, (60, 33): -1, (46, 7): -1, (6, 22): -1, (65, 68): -1, (52, 34): -1, (44, 78): -1, (24, 52): -1, (38, 65): -1, (50, 18): -1, (54, 31): -1, (33, 55): -1, (49, 33): -1, (4, 66): -1, (45, 73): -1, (39, 34): -1, (63, 47): -1, (24, 79): -1, (16, 79): -1, (55, 64): -1, (21, 8): -1, (46, 54): -1, (2, 57): -1, (0, 26): -1, (55, 24): -1, (42, 66): -1, (44, 76): -1, (57, 54): -1, (20, 35): -1, (38, 26): -1}
    


    # Visualize the graph
    visualize_graph(graph_file, fails)



