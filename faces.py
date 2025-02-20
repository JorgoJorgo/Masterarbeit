import matplotlib.pyplot as plt
import networkx as nx
import math
import uuid

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

def route_faces_firstFace(s, d, tree, fails):
    limit = len(tree.nodes)*len(tree.nodes)
    debug = False
    #if(d == 12): debug = True

    #print_cut_structure([], [], tree, s, d, fails=fails, filename=" ", save_plot=False)
    hops_faces = 0
    switches = 0
    detour_edges = []

    #print(f"[route Faces] Routing from {s} to {d}")
    #print(f"Edges: {tree.edges}")
    #print(f"Expected Edges: {[(s, d)]} or {[(d, s)]}")
    
    edges_list = list(tree.edges)
    #print(f"Edges List: {edges_list}")
    #print(f"Comparison 1: {edges_list == [(s, d)]}")
    #print(f"Comparison 2: {edges_list == [(d, s)]}")

    if edges_list == [(s, d)] or edges_list == [(d, s)]:
        if (s, d) in fails or (d, s) in fails:
            print("[route Faces] Routing failed with Start Face")
            return (True, hops_faces, switches, detour_edges)
        else:
            print("[route Faces] Routing success with Start Face")
            return (False, hops_faces, switches, detour_edges)

    faces_with_s_and_d = find_faces_pre(tree, source=s, destination=d)
    #print(f"[route Faces] Faces before sorting: {faces_with_s_and_d}")
    faces_with_s_and_d.sort(key=len)
    #print(f"[route Faces] Faces after sorting: {faces_with_s_and_d}")

    if not faces_with_s_and_d:
        #print("[route Faces] No faces found containing both s and d")
        return (True, hops_faces, switches, detour_edges)

    smallest_face = faces_with_s_and_d[0]
    #print(f"[route Faces] Smallest face: {smallest_face}")

    current_node = s
    currentIndex = 0
    previous_node = s

    while current_node != d:
        if currentIndex + 1 >= len(smallest_face):
            #print("[route Faces] Index out of bounds for smallest face")
            break

        next_node = smallest_face[currentIndex + 1]
        #print(f"[route Faces] Trying edge ({current_node}, {next_node})")

        if (current_node, next_node) in fails or (next_node, current_node) in fails:
            #print(f"[route Faces] Edge ({current_node}, {next_node}) is a failure")
            break

        previous_node = current_node
        current_node = next_node
        currentIndex += 1
        hops_faces += 1
        detour_edges.append((previous_node, current_node))
        #if(debug):
        #    print_cut_structure([current_node], [(previous_node, current_node)], tree, s, d, fails=fails, filename=" ", save_plot=False)
        #print(f"[route Faces] Moved to {current_node}")

    if current_node == d:
        print("[route Faces] Routing success via Start Face")
        return (False, hops_faces, switches, detour_edges)

    switches += 1
    source_edges = [(s, smallest_face[1])]
    print(f"[route Faces] Source edges initialized: {source_edges}")

    if current_node == s:
        #print("[route Faces] Still at source, trying clockwise routing")
        neighbors = sorted_neighbors_for_face_routing(tree, s, None, fails)
        #print(f"[route Faces] Neighbors of {s}: {neighbors}")
        
        for neighbor in neighbors:
            if (s, neighbor) in fails or (neighbor, s) in fails:
                #print(f"[route Faces] Neighbor {neighbor} is a failure")
                continue
            
            source_edges.append((s, neighbor))
            previous_node = s
            current_node = neighbor
            hops_faces += 1
            detour_edges.append((previous_node, current_node))
            #print(f"[route Faces] Taking edge ({previous_node}, {current_node})")
            break
    
    if(debug):
        print_cut_structure([current_node], [(previous_node, current_node)], tree, s, d, fails=fails, filename=" ", save_plot=False)

    while current_node != d:
        print("Source Edges: ", source_edges)
        neighbors = sorted_neighbors_for_face_routing(tree, current_node, previous_node, fails)
        print(f"[route Faces] Current Node: {current_node}")
        print(f"[route Faces] Previous Node: {previous_node}")
        print(f"[route Faces] D: {d}")
        edge_taken = False

        if d in neighbors:

            if (current_node, d) in fails or (d, current_node) in fails:
                print("[route Faces] Edge ({current_node}, {d}) is a failure")
            else:
                previous_node = current_node
                current_node = d
                hops_faces += 1
                detour_edges.append((previous_node, current_node))
                edge_taken = True
                #print(f"[route Faces] Taking edge ({previous_node}, {current_node})")
                break
        
        for neighbor in neighbors:
            if (current_node, neighbor) in fails or (neighbor, current_node) in fails:
                continue

            # Verhindern, dass wir eine Kante doppelt in source_edges speichern
            if (current_node, neighbor) in source_edges:
                break

            previous_node = current_node
            current_node = neighbor
            hops_faces += 1
            detour_edges.append((previous_node, current_node))
            edge_taken = True
            
            # Falls die Kante von der Quelle s ausgeht, füge sie in source_edges hinzu
            if previous_node == s:
                source_edges.append((previous_node, current_node))

            break

        if not edge_taken:
            print("[route Faces] No valid edge found, routing failed with Clockwise Face Routing")
            return (True, hops_faces, switches, detour_edges)

        if(debug):
            print_cut_structure([current_node], [(previous_node, current_node)], tree, s, d, fails=fails, filename=" ", save_plot=False)


    #print("[route Faces] Routing success with Clockwise Face Routing")
    return (False, hops_faces, switches, detour_edges)


import math


def sorted_neighbors_for_face_routing(graph, node, coming_from=None, fails=[]):
    """Sortiert die Nachbarn eines Knotens im Uhrzeigersinn für Face Routing."""
    
    # Hole die Koordinaten des aktuellen Knotens
    node_x, node_y = graph.nodes[node]['pos']
    
    all_neighbors = list(graph.neighbors(node))
    print(f"\n[DEBUG] Current Node: {node}, Coming From: {coming_from}")
    print(f"[DEBUG] All Neighbors before filtering: {all_neighbors}")

    # Fail-Kanten sauber filtern
    valid_neighbors = [neighbor for neighbor in all_neighbors if (node, neighbor) not in fails and (neighbor, node) not in fails]
    print(f"[DEBUG] Valid Neighbors after filtering fails: {valid_neighbors}")
    
    if not valid_neighbors:
        return []
    
    # Falls coming_from existiert, Winkel relativ zu diesem berechnen
    if coming_from and coming_from in graph.nodes:
        from_x, from_y = graph.nodes[coming_from]['pos']
        reference_angle = math.atan2(from_y - node_y, from_x - node_x)
    else:
        reference_angle = math.atan2(0, 1)  # Standardmäßig positive X-Achse (0°)
    
    # Funktion zur Berechnung des Winkels
    def angle(neighbor):
        nx, ny = graph.nodes[neighbor]['pos']
        return math.atan2(ny - node_y, nx - node_x)
    
    # Winkel relativ zur Referenz normalisieren
    def normalized_angle(neighbor):
        diff = angle(neighbor) - reference_angle
        return (diff + 2 * math.pi) % (2 * math.pi)  # Wert in [0, 2π] halten
    
    # Winkel berechnen und ausgeben
    neighbor_angles = {neighbor: round(normalized_angle(neighbor), 5) for neighbor in valid_neighbors}
    print(f"[DEBUG] Normalized Angles (before sorting): {neighbor_angles}")
    
    # Nachbarn im Uhrzeigersinn sortieren
    valid_neighbors.sort(key=lambda neighbor: neighbor_angles[neighbor], reverse=True)
    print(f"[DEBUG] Sorted Neighbors (Clockwise Order): {valid_neighbors}")
    
    # Falls coming_from existiert, sortiere es korrekt ein
    if coming_from is not None and coming_from in graph.nodes:
        if coming_from in valid_neighbors:
            index = valid_neighbors.index(coming_from)
            valid_neighbors = valid_neighbors[index+1:] + valid_neighbors[:index+1]
    
    print(f"[DEBUG] Final Sorted Neighbors (coming_from at end if exists): {valid_neighbors}\n")
    return valid_neighbors


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
    print("----------------------------------------------------------------")
    print("[route_faces_with_paths] Routing from", s, "to", d)
    #print_cut_structure([], [], paths[s][d]['structure'], s, d, fails=fails, filename=" ", save_plot=False)
    return route_faces_firstFace(s, d, paths[s][d]['structure'], fails)


def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def route_greedy_perimeter(s, d, fails, paths):
    speacial_nodes = []  # wenn man nix zeichnen will
    #speacial_nodes = [s,d] #wenn man alles zeichnen will
    count_all_nodes= len(paths)

    print("[route_greedy_perimeter] Routing from", s, "to", d)  
    tree = paths[s][d]['structure']
    cut_edges = paths[s][d]['cut_edges']
    cut_nodes = paths[s][d]['cut_nodes']
    tree = convert_to_undirected(tree)

    visited_edges = set()
    visited_nodes = set()
    current_node = s
    path = [current_node]
    previous_edge = None
    
    hops = 0
    switches = 0
    detour_edges = []
    greedy_mode = True  # Start in Greedy Mode
    
    while current_node != d:
        visited_nodes.add(current_node)

        if greedy_mode:
            # Greedy Forwarding: Wähle den Nachbarn mit der kleinsten Distanz zu D, der nicht in fails ist
            neighbors = [n for n in tree[current_node] if (current_node, n) not in fails and (n, current_node) not in fails]
            neighbors = [n for n in neighbors if n not in visited_nodes]  # Vermeidung von Zyklen
            if not neighbors:
                greedy_mode = False  # Wechsel zu Perimeter Routing
                continue

            best_neighbor = min(neighbors, key=lambda n: euclidean_distance(tree.nodes[n]['pos'], tree.nodes[d]['pos']))
            next_edge = (current_node, best_neighbor)
        else:
            # Perimeter Routing: Fallback für Sackgassen
            edges = get_sorted_edges(current_node, tree, fails, previous_edge, s=s, d=d)
            edges = [e for e in edges if e not in fails and (e[1], e[0]) not in fails]
            edges = [e for e in edges if e[1] not in visited_nodes]  # Vermeidung von unendlichen Loops
            if not edges:
                if len(path) > 1:
                    previous_node = path[-2]
                    path.pop()
                    current_node = previous_node
                    switches += 1
                    previous_edge = (current_node, path[-1])
                    continue  # Rücksprung zur Schleife, um neuen Versuch zu starten
                else:
                    print("Routing failed. No way to proceed.")
                    unique_filename = f"failedgraphs/routeGreedyPerimeter_graph_{uuid.uuid4().hex}.png"
                    print_cut_structure(cut_nodes, cut_edges, tree, s, d, fails=fails, filename=unique_filename,save_plot=True)
                    print("[route_greedy_perimeter] count_visited_nodes:",len(visited_nodes))
                    print("[route_greedy_perimeter] nodes: ", count_all_nodes)
                    print("[route_greedy_perimeter] len(visited_nodes) < log(nodes):", len(visited_nodes) < math.log(count_all_nodes))
                    return (True, hops, switches, detour_edges)
            
            next_edge = edges[0] if edges else None
            if next_edge is None:
                print("Perimeter Routing failed: No available edges.")
                unique_filename = f"failedgraphs/routeGreedyPerimeter_graph_{uuid.uuid4().hex}.png"
                print_cut_structure(cut_nodes, cut_edges, tree, s, d, fails=fails, filename=unique_filename,save_plot=True)
                print("[route_greedy_perimeter] count_visited_nodes:",len(visited_nodes))
                print("[route_greedy_perimeter] nodes: ", count_all_nodes)
                print("[route_greedy_perimeter] len(visited_nodes) < log(nodes):", len(visited_nodes) < math.log(count_all_nodes))
                return (True, hops, switches, detour_edges)
        
        if next_edge in visited_edges:
            detour_edges.append(next_edge)

        visited_edges.add(next_edge)
        previous_edge = next_edge
        current_node = next_edge[1] if next_edge[0] == current_node else next_edge[0]
        path.append(current_node)
        hops += 1

        if s in speacial_nodes and d in speacial_nodes:
            print_cut_structure(cut_nodes, cut_edges, tree, s, d, current_edge=previous_edge, fails=fails)

        print("-----")
    
    print("Routing successful.")
    print("[route_greedy_perimeter] count_visited_nodes:",len(visited_nodes))
    print("[route_greedy_perimeter] nodes: ", count_all_nodes)
    print("[route_greedy_perimeter] len(visited_nodes) < log(nodes):", len(visited_nodes) < math.log(count_all_nodes))
    return (False, hops, switches, detour_edges)

def angle_between(p1, p2):
    """Berechnet den Winkel zwischen zwei Punkten relativ zur x-Achse."""
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

def sorted_neighbors(graph, node, coming_from):
    """Sortiere die Nachbarn eines Knotens basierend auf ihrem Winkel relativ zur vorherigen Kante."""
    pos = nx.get_node_attributes(graph, 'pos')
    neighbors = list(nx.neighbors(graph, node))
    if coming_from is not None:
        base_angle = angle_between(pos[node], pos[coming_from])
    else:
        base_angle = 0  # Falls kein vorheriger Knoten vorhanden ist
    
    neighbors.sort(key=lambda n: (angle_between(pos[node], pos[n]) - base_angle) % (2 * math.pi))
    return neighbors

def trace_face(graph, start, first_neighbor, clockwise=True):
    """Verfolge ein Face vom Startknoten aus, basierend auf der Sortierung der Kanten."""
    pos = nx.get_node_attributes(graph, 'pos')
    face = [start]
    current = first_neighbor
    previous = start
    visited_edges = set()
    
    while True:
        face.append(current)
        visited_edges.add((previous, current))
        
        neighbors = sorted_neighbors(graph, current, previous)
        if not clockwise:
            neighbors.reverse()
        
        found_next = False
        for next_node in neighbors:
            if (current, next_node) not in visited_edges or next_node == start:  # Erlaubt Zurückkehren zum Start
                previous = current
                current = next_node
                found_next = True
                break
        
        if current == start:
            return face  # Wenn wir zurück am Start sind, ist das Face vollständig
        
        if not found_next:
            break  # Falls keine gültige Fortsetzung gefunden wird, brechen wir ab
    
    return face if len(face) > 2 else []  # Stelle sicher, dass das Face mehr als nur Start und ein Nachbar enthält

def find_faces_pre(graph, source, destination):

    """Findet alle Faces um den Quellknoten, die die Destination beinhalten."""
    faces = []
    neighbors = list(nx.neighbors(graph, source))
    #print("Neighbors: ", neighbors)
    for neighbor in neighbors:
        #print("Current Neighbor: ", neighbor)
        face_cw = trace_face(graph, source, neighbor, clockwise=True)
        if face_cw and face_cw not in faces and destination in face_cw:
            faces.append(face_cw)
        
        face_ccw = trace_face(graph, source, neighbor, clockwise=False)
        if face_ccw and face_ccw not in faces and destination in face_ccw:
            faces.append(face_ccw)

    #draw_graph_with_colored_faces(graph, faces, source, destination)

    return faces


def draw_graph_with_highlighted_edge(graph, source, destination, edge):
    """Zeichnet den Graphen mit einer hervorgehobenen Kante in Blau und hebt Source und Destination hervor."""
    pos = nx.get_node_attributes(graph, 'pos')
    plt.figure(figsize=(8, 6))
    
    # Zeichne den Graphen
    nx.draw(graph, pos, with_labels=True, edge_color='black', node_color='lightgray', node_size=500, font_size=10)
    
    # Hebe die Kante hervor
    if edge in graph.edges:
        nx.draw_networkx_edges(graph, pos, edgelist=[edge], edge_color='blue', width=2.5)
    
    # Hebe Source und Destination hervor
    nx.draw_networkx_nodes(graph, pos, nodelist=[source], node_color='red', node_size=700)
    nx.draw_networkx_nodes(graph, pos, nodelist=[destination], node_color='green', node_size=700)
    
    plt.show()

