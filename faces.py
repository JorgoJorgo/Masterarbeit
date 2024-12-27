import matplotlib.pyplot as plt
import networkx as nx

import math

def convert_to_undirected(tree):
    """
    Converts the given tree to an undirected graph.

    Parameters:
    - tree: NetworkX graph object (directed or undirected)

    Returns:
    - An undirected NetworkX graph object
    """
    return tree.to_undirected()

def route(s, d, tree, fails):
    tree = convert_to_undirected(tree)
    print("Converted tree to undirected graph.")
    print("Type of Tree :", type(tree))

    visited_edges = set()  # Set to keep track of visited edges
    current_node = s
    path = [current_node]  # Path traversed
    previous_edge = None  # Last edge used to reach the current node

    hops = 0  # Count of hops (edges traversed)
    switches = 0  # Count of node switches
    detour_edges = []  # List of detour edges taken due to failures

    while current_node != d:
        print(f"Current node: {current_node}")
        print(f"Path so far: {path}")
        print(f"Visited edges: {visited_edges}")

        edges = get_sorted_edges(current_node, tree, fails, previous_edge)  # Sort edges by clockwise order
        print(f"Sorted edges: {edges}")

        if not edges:  # No available edges to proceed
            print("No edges available. Backtracking...")
            if len(path) > 1:
                # Go back to the previous node
                previous_node = path[-2]
                path.pop()
                current_node = previous_node
                switches += 1
                previous_edge = (current_node, path[-1])
                #draw_tree_with_highlights(tree, nodes=[s, d], fails=fails, current_edge=previous_edge)
                print(f"Backtracked to {current_node}")
            else:
                print("Routing failed. No way to proceed.")
                return (True, hops, switches, detour_edges)  # No way to proceed

        edge_taken = False
        reverse_edge = (previous_edge[1], previous_edge[0]) if previous_edge else None

        for edge in edges:
            print(f"Checking edge {edge}")
            if edge == reverse_edge:
                print(f"Skipping reverse edge {edge} temporarily.")
                continue
            if edge not in visited_edges:
                visited_edges.add(edge)
                previous_edge = edge
                current_node = edge[1] if edge[0] == current_node else edge[0]
                path.append(current_node)
                hops += 1
                if edge in detour_edges:
                    detour_edges.append(edge)
                edge_taken = True
                #draw_tree_with_highlights(tree, nodes=[s, d], fails=fails, current_edge=edge)
                print(f"Edge {edge} taken. Moving to node {current_node}")
                break

        if not edge_taken and reverse_edge and reverse_edge not in visited_edges:
            print(f"No other options. Taking reverse edge {reverse_edge}.")
            visited_edges.add(reverse_edge)
            previous_edge = reverse_edge
            current_node = reverse_edge[1] if reverse_edge[0] == current_node else reverse_edge[0]
            path.append(current_node)
            hops += 1
            edge_taken = True
            #draw_tree_with_highlights(tree, nodes=[s, d], fails=fails, current_edge=reverse_edge)

        if not edge_taken:
            print("Cycle detected or all edges revisited. Routing failed.")
            return (True, hops, switches, detour_edges)  # All edges revisited, cycle found
        print("-----")

    print("Routing successful.")
    print(f"Final path: {path}")
    print(f"Total hops: {hops}, switches: {switches}, detour edges: {detour_edges}")
    return (False, hops, switches, detour_edges)  # Path successfully found to destination


# Helper function to get edges sorted in clockwise order
def get_sorted_edges(node, tree, fails, previous_edge):
    edges = []
    node_pos = tree.nodes[node]['pos']

    for neighbor in tree.neighbors(node):
        edge = (node, neighbor)  # Behalte die originale Richtung bei
        if edge not in fails and (neighbor, node) not in fails:  # Prüfe beide Richtungen in fails
            neighbor_pos = tree.nodes[neighbor]['pos']
            angle = calculate_angle(node_pos, neighbor_pos)
            edges.append((edge, angle))

    edges.sort(key=lambda x: x[1])  # Sort edges based on angles
    if previous_edge is not None:
        edges = prioritize_edges(edges, previous_edge, tree)

    return [e[0] for e in edges]


# Helper function to calculate the angle between two coordinates
def calculate_angle(pos1, pos2):
    dx = pos2[0] - pos1[0]
    dy = pos2[1] - pos1[1]
    return math.atan2(dy, dx)  # Returns angle in radians


# Helper function to prioritize edges based on the previous edge
def prioritize_edges(edges, previous_edge, tree):
    previous_angle = calculate_angle(tree.nodes[previous_edge[0]]['pos'], tree.nodes[previous_edge[1]]['pos'])
    edges.sort(key=lambda x: (x[1] - previous_angle) % (2 * math.pi))
    return edges


def draw_tree_with_highlights(tree, nodes=None, fails=None, current_edge=None):
    """
    Zeichnet einen Baum-Graphen und hebt bestimmte Knoten, fehlerhafte Kanten und die aktuelle Kante hervor.

    Parameter:
    - tree: NetworkX-Graph-Objekt, das den Baum darstellt.
    - nodes: Liste von Knoten, die hervorgehoben werden sollen (optional).
    - fails: Liste von fehlerhaften Kanten, die hervorgehoben werden sollen (optional).
    - current_edge: Aktuelle Kante, die hervorgehoben werden soll (optional).
    """
    pos = {node: tree.nodes[node]['pos'] for node in tree.nodes}  # Positionen der Knoten

    plt.figure(figsize=(10, 8))

    # Zeichne alle Kanten in Grau
    nx.draw_networkx_edges(tree, pos, edge_color='gray')

    # Zeichne fehlerhafte Kanten in Rot, falls vorhanden
    if fails:
        failed_edges = [(u, v) for u, v in fails if tree.has_edge(u, v)]
        nx.draw_networkx_edges(tree, pos, edgelist=failed_edges, edge_color='red', width=2)
        #print(f"Hervorgehobene Kanten (Fails): {fails}")

    # Highlight aktuelle Kante in Blau, falls vorhanden
    if current_edge:
        if tree.has_edge(*current_edge):
            nx.draw_networkx_edges(tree, pos, edgelist=[current_edge], edge_color='blue', width=2)
            #print(f"Aktuelle Kante hervorgehoben: {current_edge}")

    # Zeichne alle Knoten
    nx.draw_networkx_nodes(tree, pos, node_color='lightgray', node_size=500)
    nx.draw_networkx_labels(tree, pos)

    # Hervorheben spezieller Knoten in Orange, falls vorhanden
    if nodes:
        nx.draw_networkx_nodes(tree, pos, nodelist=nodes, node_color="orange", node_size=700)
        #print(f"Hervorgehobene Knoten: {nodes}")

    #plt.title("Baum mit hervorgehobenen Knoten, Kanten und aktueller Kante")
    plt.show()
