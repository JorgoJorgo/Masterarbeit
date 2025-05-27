import matplotlib.pyplot as plt
import networkx as nx
import math
import uuid
import math

from scipy.spatial.distance import euclidean

from scipy.spatial.distance import euclidean

def convert_to_undirected(tree):
    """
    Converts the given tree to an undirected graph.

    Parameters:
    - tree: NetworkX graph object (directed or undirected)

    Returns:Update directory name in print_results.py
    - An undirected NetworkX graph object
    """
    return tree.to_undirected()



#route from s to d using face routing with the smallest face
#if routing fails via the smallest face, then route using clockwise face routing
#routing fails if cycle is detected at the source
def route_faces_firstFace(s, d, tree, fails):
    debug = False
    #if(d == 12): debug = True

    #print_cut_structure([], [], tree, s, d, fails=fails, filename=" ", save_plot=False)
    hops_faces = 0
    switches = 0
    detour_edges = []
    if(s==d):
        return (False, hops_faces, switches, detour_edges)
    edges_list = list(tree.edges)

    if edges_list == [(s, d)] or edges_list == [(d, s)]:
        if (s, d) in fails or (d, s) in fails:
            print("[route Faces] Routing failed with Start Face")
            return (True, hops_faces, switches, detour_edges)
        else:
            print("[route Faces] Routing success with Start Face")
            return (False, hops_faces, switches, detour_edges)

    faces_with_s_and_d = find_faces_pre(tree, source=s, destination=d)
    faces_with_s_and_d.sort(key=len)

    if not faces_with_s_and_d:
        return (True, hops_faces, switches, detour_edges)

    smallest_face = faces_with_s_and_d[0]

    current_node = s
    currentIndex = 0
    previous_node = s

    while current_node != d:
        if currentIndex + 1 >= len(smallest_face):
            break

        next_node = smallest_face[currentIndex + 1]

        if (current_node, next_node) in fails or (next_node, current_node) in fails:
            break

        previous_node = current_node
        current_node = next_node
        currentIndex += 1
        hops_faces += 1
        detour_edges.append((previous_node, current_node))
        if(debug):
            print_cut_structure([current_node], [(previous_node, current_node)], tree, s, d, fails=fails, filename=" ", save_plot=False)
        #print(f"[route Faces] Moved to {current_node}")

    if current_node == d:
        print("[route Faces] Routing success via Start Face")
        return (False, hops_faces, switches, detour_edges)

    switches += 1
    source_edges = [(s, smallest_face[1])]
    #print(f"[route Faces] Source edges initialized: {source_edges}")

    if current_node == s:
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
        #print("Source Edges: ", source_edges)
        neighbors = sorted_neighbors_for_face_routing(tree, current_node, previous_node, fails)
        #print(f"[route Faces] Current Node: {current_node}")
        #print(f"[route Faces] Previous Node: {previous_node}")
        #print(f"[route Faces] D: {d}")
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

            # Prevent storing an edge twice in source_edges
            if (current_node, neighbor) in source_edges:
                break

            previous_node = current_node
            current_node = neighbor
            hops_faces += 1
            detour_edges.append((previous_node, current_node))
            edge_taken = True
            
            # If the edge originates from the source s, add it to source_edges
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


def sorted_neighbors_for_face_routing(graph, node, coming_from=None, fails=[]):
    """Sorts the neighbors of a node clockwise for face routing."""
    
    # Get the coordinates of the current node
    node_x, node_y = graph.nodes[node]['pos']
    
    all_neighbors = list(graph.neighbors(node))
    #print(f"\n[DEBUG] Current Node: {node}, Coming From: {coming_from}")
    #print(f"[DEBUG] All Neighbors before filtering: {all_neighbors}")

    # Cleanly filter fail edges
    valid_neighbors = [neighbor for neighbor in all_neighbors if (node, neighbor) not in fails and (neighbor, node) not in fails]
    #print(f"[DEBUG] Valid Neighbors after filtering fails: {valid_neighbors}")
    
    if not valid_neighbors:
        return []
    
    # If coming_from exists, calculate angles relative to it
    if coming_from and coming_from in graph.nodes:
        from_x, from_y = graph.nodes[coming_from]['pos']
        reference_angle = math.atan2(from_y - node_y, from_x - node_x)
    else:
        reference_angle = math.atan2(0, 1)  # Default to positive X-axis (0°)
    
    # Function to calculate the angle
    def angle(neighbor):
        nx, ny = graph.nodes[neighbor]['pos']
        return math.atan2(ny - node_y, nx - node_x)
    
    # Normalize angles relative to the reference
    def normalized_angle(neighbor):
        diff = angle(neighbor) - reference_angle
        return (diff + 2 * math.pi) % (2 * math.pi)  # Keep value in [0, 2π]
    
    # Calculate and output angles
    neighbor_angles = {neighbor: round(normalized_angle(neighbor), 5) for neighbor in valid_neighbors}
    #print(f"[DEBUG] Normalized Angles (before sorting): {neighbor_angles}")
    
    # Sort neighbors clockwise
    valid_neighbors.sort(key=lambda neighbor: neighbor_angles[neighbor], reverse=True)
    #print(f"[DEBUG] Sorted Neighbors (Clockwise Order): {valid_neighbors}")
    
    # If coming_from exists, sort it correctly
    if coming_from is not None and coming_from in graph.nodes:
        if coming_from in valid_neighbors:
            index = valid_neighbors.index(coming_from)
            valid_neighbors = valid_neighbors[index+1:] + valid_neighbors[:index+1]
    
    #print(f"[DEBUG] Final Sorted Neighbors (coming_from at end if exists): {valid_neighbors}\n")
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
    hops= 0
    switches = 0 
    detour_edges = []
    
    routing_failure, hops, switches, detour_edges = route_faces_firstFace(s, d, paths[s][d]['structure'], fails)

    print(f"[MultipleTrees Cuts Extended / MultipleTrees Faces Extended] sizes : {len(paths[s][d]['structure'].edges())}")

    if routing_failure == False:
        print("[route_faces_with_paths] Routing success")

    else:
        print("[route_faces_with_paths] Routing failed")

    return (routing_failure, hops, switches, detour_edges)


def route_greedy_faces_with_paths(s,d,fails,paths):
    print("----------------------------------------------------------------")
    print("[route_greedy_faces_with_paths] Routing from", s, "to", d)
    hops= 0
    switches = 0 
    detour_edges = []
    if(s==d):
        return (False, hops, switches, detour_edges)
    routing_failure, hops, switches, detour_edges = route_greedy_perimeter(s, d, paths[s][d]['structure'], fails)



    if routing_failure == False:
        print("[route_greedy_faces_with_paths] Routing success")

    else:
        print("[route_greedy_faces_with_paths] Routing failed")

    return (routing_failure, hops, switches, detour_edges)

def euclidean_distance(a, b):
    return math.sqrt((a[0] - b[0])**2 + (a[1] - b[1])**2)

def route_greedy_perimeter(s, d, tree, fails):
    hops = 0
    switches = 0
    detour_edges = []
    current_node = s
    source_edges = []
    debug = False
    previous_node = s
    max_hops = len(tree.nodes)*len(tree.nodes)
    if(s==d):
        return (False, hops, switches, detour_edges)

    while current_node != d:

        if hops > max_hops:
            print(f"[route Greedy] Routing failed, max_hops ({max_hops}) reached")
            return (True, hops, switches, detour_edges)

        #sort current neighbors based on the distance to d
        neighbors = sort_neighbors_for_greedy_routing(tree, current_node,previous_node, d, fails)

        if not neighbors:
            print("[route Greedy] Routing failed, no way to proceed")
            return (True, hops, switches, detour_edges)
        
        if d in neighbors:
            if not (current_node, d) in fails and not (d, current_node) in fails:
                print("[route Greedy] Routing success")
                return (False, hops + 1, switches, detour_edges + [(current_node, d)])
        
        next_node = neighbors[0]
        previous_node = current_node
        current_node = next_node

        if (previous_node, current_node) in source_edges:
            print("[route Greedy] Routing failed, cycle detected at source")
            return (True, hops, switches, detour_edges)

        if previous_node == s:
            source_edges.append((s, current_node))

        hops += 1
        detour_edges.append((previous_node, current_node))



        if debug:
            print_cut_structure([current_node], [(previous_node, current_node)], tree, s, d, fails=fails, filename=" ", save_plot=False)



    return (False, hops, switches, detour_edges)

def angle_between(p1, p2):
    """Calculates the angle between two points relative to the x-axis."""
    return math.atan2(p2[1] - p1[1], p2[0] - p1[0])

def sorted_neighbors(graph, node, coming_from):
    """Sorts the neighbors of a node based on their angle relative to the previous edge."""
    pos = nx.get_node_attributes(graph, 'pos')
    neighbors = list(nx.neighbors(graph, node))
    if coming_from is not None:
        base_angle = angle_between(pos[node], pos[coming_from])
    else:
        base_angle = 0  # If no previous node is present
    
    neighbors.sort(key=lambda n: (angle_between(pos[node], pos[n]) - base_angle) % (2 * math.pi))
    return neighbors

def trace_face(graph, start, first_neighbor, clockwise=True):
    """Traces a face from the start node based on the sorting of edges."""
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
            if (current, next_node) not in visited_edges or next_node == start:  # Allows returning to the start
                previous = current
                current = next_node
                found_next = True
                break
        
        if current == start:
            return face  # If we are back at the start, the face is complete
        
        if not found_next:
            break  # If no valid continuation is found, break
    
    return face if len(face) > 2 else []  # Ensure the face contains more than just the start and one neighbor

def find_faces_pre(graph, source, destination):
    """Finds all faces around the source node that include the destination."""
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
    """Draws the graph with a highlighted edge in blue and highlights source and destination."""
    pos = nx.get_node_attributes(graph, 'pos')
    plt.figure(figsize=(8, 6))
    
    # Draw the graph
    nx.draw(graph, pos, with_labels=True, edge_color='black', node_color='lightgray', node_size=500, font_size=10)
    
    # Highlight the edge
    if edge in graph.edges:
        nx.draw_networkx_edges(graph, pos, edgelist=[edge], edge_color='blue', width=2.5)
    
    # Highlight source and destination
    nx.draw_networkx_nodes(graph, pos, nodelist=[source], node_color='red', node_size=700)
    nx.draw_networkx_nodes(graph, pos, nodelist=[destination], node_color='green', node_size=700)
    
    plt.show()



def sort_neighbors_for_greedy_routing(graph, current_node, previous_node, destination, fails=[]):
    neighbors = list(graph.neighbors(current_node))  # Convert neighbors to a list
    destination_pos = graph.nodes[destination]['pos']
    
    # Filter neighbors whose edge is in `fails`
    valid_neighbors = [
        neighbor for neighbor in neighbors 
        if (current_node, neighbor) not in fails and (neighbor, current_node) not in fails
    ]
    
    # Sort remaining neighbors by Euclidean distance to the destination
    sorted_neighbors = sorted(
        valid_neighbors, 
        key=lambda neighbor: euclidean(graph.nodes[neighbor]['pos'], destination_pos)
    )
    
    # If previous_node is in the list, move it to the end
    if previous_node in sorted_neighbors:
        sorted_neighbors.remove(previous_node)
        sorted_neighbors.append(previous_node)

    #print(f"Sorted neighbors by greedy routing: {sorted_neighbors}")
    
    return sorted_neighbors


def print_cut_structure(highlighted_nodes, cut_edges, structure, source, destination, fails=[], current_edge=None, save_plot=False, filename="graphen/graph.png"):
    pos = nx.get_node_attributes(structure, 'pos')
    
    plt.figure(figsize=(10, 10))
    
    # Draw the entire graph with normal edges in black
    nx.draw(structure, pos, with_labels=True, node_color='lightblue', edge_color='black', node_size=500, font_size=10)
    
    # Highlight specific nodes
    nx.draw_networkx_nodes(structure, pos, nodelist=highlighted_nodes, node_color='red')
    
    # Highlight cut edges in green
    nx.draw_networkx_edges(structure, pos, edgelist=cut_edges, edge_color='green', width=2)
    
    # Highlight source and destination nodes
    nx.draw_networkx_nodes(structure, pos, nodelist=[source], node_color='green')
    nx.draw_networkx_nodes(structure, pos, nodelist=[destination], node_color='yellow')
    
    # Highlight the current edge, if present
    if current_edge:
        nx.draw_networkx_edges(structure, pos, edgelist=[current_edge], edge_color='blue', width=2)
    
    # **Fix: Check fail edges in both directions**
    valid_fails = [(u, v) for (u, v) in structure.edges if (u, v) in fails or (v, u) in fails]
    
    if valid_fails:
        nx.draw_networkx_edges(structure, pos, edgelist=valid_fails, edge_color='red', width=2)
    
    if save_plot:
        os.makedirs("failedgraphs", exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"failedgraphs/graph_{source}_{destination}_{current_time}.png"
        plt.savefig(filename)
    else:
        plt.show()

    plt.close()
