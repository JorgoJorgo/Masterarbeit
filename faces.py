import matplotlib.pyplot as plt
import networkx as nx

import math

from cut_algorithms import print_cut_structure

def convert_to_undirected(tree):
    """
    Converts the given tree to an undirected graph.

    Parameters:
    - tree: NetworkX graph object (directed or undirected)

    Returns:Update directory name in print_results.py
    - An undirected NetworkX graph object
    """
    return tree.to_undirected()

def route(s, d, fails, tree):
    speacial_nodes = [] #wenn man nix zeichnen will
    #speacial_nodes = [9,47] #wenn man bestimmte nodes zeichnen will
    #speacial_nodes = [s,d] #wenn man alles zeichnen will
    
    print("Routing from", s, "to", d)
    print("Fails:", fails)
    print("Tree:", tree)

    print("Tree nodes:", tree.nodes)
    tree = convert_to_undirected(tree)

    
    visited_edges = set()  # Set to keep track of visited edges
    current_node = s
    path = [current_node]  # Path traversed
    previous_edge = None  # Last edge used to reach the current node

    hops = 0  # Count of hops (edges traversed)
    switches = 0  # Count of node switches
    detour_edges = []  # List of detour edges taken due to failures

    while current_node != d:

        edges = get_sorted_edges(current_node, tree, fails, previous_edge,s=s,d=d)  # Sort edges by clockwise order

        if not edges:  # No available edges to proceed

            if len(path) > 1:
                # Go back to the previous node
                previous_node = path[-2]
                path.pop()
                current_node = previous_node
                switches += 1
                previous_edge = (current_node, path[-1])
                if s in speacial_nodes and d in speacial_nodes:
                    draw_tree_with_highlights(tree, nodes=[s, d], fails=fails, current_edge=previous_edge)
            else:
                print("Routing failed. No way to proceed.")
                print("[route] detour_edges:",detour_edges)
                return (True, hops, switches, detour_edges)  # No way to proceed

        edge_taken = False
        reverse_edge = (previous_edge[1], previous_edge[0]) if previous_edge else None

        for edge in edges:
            #print(f"Checking edge {edge}")
            if edge == reverse_edge:
                continue
            if edge not in visited_edges:
                visited_edges.add(edge)
                previous_edge = edge
                current_node = edge[1] if edge[0] == current_node else edge[0]
                path.append(current_node)
                hops += 1
                if edge in visited_edges:
                    detour_edges.append(edge)
                edge_taken = True
                if s in speacial_nodes and d in speacial_nodes:
                    draw_tree_with_highlights(tree, nodes=[s, d], fails=fails, current_edge=edge)
                break

        if not edge_taken and reverse_edge and reverse_edge not in visited_edges:
            visited_edges.add(reverse_edge)
            previous_edge = reverse_edge
            current_node = reverse_edge[1] if reverse_edge[0] == current_node else reverse_edge[0]
            path.append(current_node)
            hops += 1
            edge_taken = True
            if s in speacial_nodes and d in speacial_nodes:
                draw_tree_with_highlights(tree, nodes=[s, d], fails=fails, current_edge=reverse_edge)

        if not edge_taken:
            print("Cycle detected or all edges revisited. Routing failed.")
            return (True, hops, switches, detour_edges)  # All edges revisited, cycle found
        print("-----")

    print("Routing successful.")
    return (False, hops, switches, detour_edges)  # Path successfully found to destination


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

# # Helper function to get edges sorted in clockwise order
# def get_sorted_edges(node, tree, fails, previous_edge):
#     edges = []
#     node_pos = tree.nodes[node]['pos']

#     for neighbor in tree.neighbors(node):
#         edge = (node, neighbor) if node < neighbor else (neighbor, node)  # Ensure undirected edge representation
#         if edge not in fails and (edge[1],edge[0]) not in fails:  # Exclude edges in fails
#             neighbor_pos = tree.nodes[neighbor]['pos']
#             angle = calculate_angle(node_pos, neighbor_pos)
#             edges.append((edge, angle))

#     if previous_edge is not None:
#         edges = prioritize_edges(edges, previous_edge, tree)

#     return [e[0] for e in edges]

import math
import uuid

def calculate_angle(vec1, vec2):
        dot_product = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        magnitude1 = math.sqrt(vec1[0]**2 + vec1[1]**2)
        magnitude2 = math.sqrt(vec2[0]**2 + vec2[1]**2)
        if magnitude1 == 0 or magnitude2 == 0:
            return 0
        # Clamp the value to avoid math domain errors
        cos_theta = max(-1, min(1, dot_product / (magnitude1 * magnitude2)))
        angle = math.acos(cos_theta)
        # Determine the orientation (clockwise or counterclockwise)
        cross_product = vec1[0] * vec2[1] - vec1[1] * vec2[0]
        return angle if cross_product >= 0 else -angle

def get_sorted_edges(node, tree, fails, previous_edge,s,d):
    sonderfall = False
    # Get position of the current node
    node_pos = tree.nodes[node]['pos']

    # Default values for previous edge
    previous_vector = (1, 0)  # Default vector if no previous edge is given
    previous_neighbor = None

    if previous_edge is not None:
        # Extract the previous edge's source and target
        previous_source, previous_target = previous_edge

        # Calculate the vector of the previous edge
        previous_neighbor = previous_source if previous_target == node else previous_target
        previous_pos = tree.nodes[previous_neighbor]['pos']
        previous_vector = (node_pos[0] - previous_pos[0], node_pos[1] - previous_pos[1])
    
    else: #if there is no previous edge, then the previous edge is the imaginary edge between s and d
        sonderfall = True
        # Extract the previous edge's source and target
        previous_source = s
        previous_target = d

        # Calculate the vector of the previous edge
        previous_neighbor = previous_source if previous_target == node else previous_target
        previous_pos = tree.nodes[previous_neighbor]['pos']
        previous_vector = (node_pos[0] - previous_pos[0], node_pos[1] - previous_pos[1])


    # List to store edges and their angles
    edges_and_angles = []

    for neighbor in tree.neighbors(node):
        # Skip the previous edge
        if (node, neighbor) in fails or (neighbor, node) in fails:
            continue

        # Get the position of the neighbor
        neighbor_pos = tree.nodes[neighbor]['pos']

        # Calculate the vector from the current node to the neighbor
        vector = (neighbor_pos[0] - node_pos[0], neighbor_pos[1] - node_pos[1])

        # Calculate the angle between the previous edge vector and this edge vector
        angle = calculate_angle(previous_vector, vector)

        # Store the edge and the angle
        edges_and_angles.append(((node, neighbor), angle))

    # Sort the edges by angle in ascending order (clockwise)
    edges_and_angles.sort(key=lambda x: x[1])

    # Ensure the previous edge's reverse direction is at the end, if it exists
    if previous_neighbor is not None and sonderfall == False:
        edges_and_angles.append(((previous_neighbor, node), float('inf')))

    return [edge for edge, _ in edges_and_angles]


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


def route_faces_with_paths(s, d, fails, paths):
    
    speacial_nodes = [] #wenn man nix zeichnen will
    #speacial_nodes = [49,47] #wenn man bestimmte nodes zeichnen will
    #speacial_nodes = [s,d] #wenn man alles zeichnen will
    
    tree = paths[s][d]['structure']
    cut_edges = paths[s][d]['cut_edges']
    cut_nodes = paths[s][d]['cut_nodes']
    tree = convert_to_undirected(tree)

    
    visited_edges = set()  # Set to keep track of visited edges
    current_node = s
    path = [current_node]  # Path traversed
    previous_edge = None  # Last edge used to reach the current node

    hops = 0  # Count of hops (edges traversed)
    switches = 0  # Count of node switches
    detour_edges = []  # List of detour edges taken due to failures

    while current_node != d:

        edges = get_sorted_edges(current_node, tree, fails, previous_edge,s=s,d=d)  # Sort edges by clockwise order

        if not edges:  # No available edges to proceed

            if len(path) > 1:
                # Go back to the previous node
                previous_node = path[-2]
                path.pop()
                current_node = previous_node
                switches += 1
                previous_edge = (current_node, path[-1])
                if s in speacial_nodes and d in speacial_nodes:
                    print_cut_structure(cut_nodes, cut_edges, tree, s, d,current_edge=previous_edge, fails=fails)
            else:
                print("Routing failed. No way to proceed.")
                print("[route] detour_edges:",detour_edges)
                unique_filename = f"failedgraphs/graph_{uuid.uuid4().hex}.png"
                print_cut_structure(cut_nodes, cut_edges, tree, s, d, fails=fails, filename=unique_filename,save_plot=True)
                return (True, hops, switches, detour_edges)  # No way to proceed

        edge_taken = False
        reverse_edge = (previous_edge[1], previous_edge[0]) if previous_edge else None

        for edge in edges:
            #print(f"Checking edge {edge}")
            if edge == reverse_edge:
                continue
            if edge not in visited_edges:
                visited_edges.add(edge)
                previous_edge = edge
                current_node = edge[1] if edge[0] == current_node else edge[0]
                path.append(current_node)
                hops += 1
                if edge in visited_edges:
                    detour_edges.append(edge)
                edge_taken = True
                if s in speacial_nodes and d in speacial_nodes:
                    print_cut_structure(cut_nodes, cut_edges, tree, s, d,current_edge=previous_edge, fails=fails)
                break

        if not edge_taken and reverse_edge and reverse_edge not in visited_edges:
            visited_edges.add(reverse_edge)
            previous_edge = reverse_edge
            current_node = reverse_edge[1] if reverse_edge[0] == current_node else reverse_edge[0]
            path.append(current_node)
            hops += 1
            edge_taken = True
            if s in speacial_nodes and d in speacial_nodes:
                    print_cut_structure(cut_nodes, cut_edges, tree, s, d,current_edge=previous_edge, fails=fails)

        if not edge_taken:
            print("Cycle detected or all edges revisited. Routing failed.")
            unique_filename = f"failedgraphs/graph_{uuid.uuid4().hex}.png"
            print_cut_structure(cut_nodes, cut_edges, tree, s, d, fails=fails, filename=unique_filename,save_plot=True)
            return (True, hops, switches, detour_edges)  # All edges revisited, cycle found
        print("-----")

    print("Routing successful.")
    return (False, hops, switches, detour_edges)  # Path successfully found to destination

