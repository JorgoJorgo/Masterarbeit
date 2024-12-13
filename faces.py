import matplotlib.pyplot as plt
import networkx as nx

def get_face_containing_edge(u, v, faces):
    """Returns the face that contains the edge (u, v)."""
    for face in faces:
        if (u, v) in face.edges or (v, u) in face.edges:
            return face
    return None

def next_edge_on_face(face, current_edge):
    """Finds the next edge on the given face using the right-hand rule."""
    u, v = current_edge
    edges = list(face.edges)
    for i, edge in enumerate(edges):
        if edge == (u, v) or edge == (v, u):
            next_edge = edges[(i + 1) % len(edges)]  # Move to the next edge
            return next_edge
    raise Exception("Edge not found on face.")

def FaceRouting(s, d, fails, tree, faces):
    """
    Routes a packet from node s to node d using the right-hand rule on a planar embedding
    of a tree while avoiding failed edges.

    Parameters:
        s (int): The source node.
        d (int): The destination node.
        fails (list): List of failed edges [(u, v), ...].
        tree: Graph representation of the tree.
        faces (list): List of faces, each face is a planar embedding face object.

    Returns:
        tuple: (cycle_detected, hops, switches, detour_edges, path)
            cycle_detected (bool): True if a cycle was detected, False otherwise.
            hops (int): Total number of hops made.
            switches (int): Total number of switches due to failed edges.
            detour_edges (list): List of detour edges taken.
            path (list): The path taken by the packet from s to d.
    """
    # Initialize variables
    current_node = s
    path = [current_node]
    visited_edges = set()
    visited_nodes = set([current_node])
    current_edge = None

    # Statistics tracking
    detour_edges = []
    hops = 0
    switches = 0
    n = len(tree.nodes())
    k = 3  # Example constant for cycle detection

    while current_node != d:
        print(f"[RouteFaces] currentNode: {current_node}")
        neighbors = list(tree.neighbors(current_node))
        print(f"Untersuche die Nachbarn: {neighbors}")

        failed_neighbors = [v for v in neighbors if (current_node, v) in fails or (v, current_node) in fails]
        print(f"Gefailte Kanten zu Nachbarn: {failed_neighbors}")

        neighbors = [v for v in neighbors if (current_node, v) not in fails and (v, current_node) not in fails]
        print(f"Übrige Nachbarn: {neighbors}")

        if not neighbors:
            draw_tree_with_fails(tree, fails)
            raise Exception("No valid neighbors available due to failed edges.")

        if current_edge is None:
            next_node = neighbors[0]  # Start with any valid neighbor
            current_edge = (current_node, next_node)

        print(f"[RouteFaces] Aktuelle Kante: {current_edge}")
        current_face = get_face_containing_edge(*current_edge, faces)
        if not current_face:
            print(f"[RouteFaces] Keine Face gefunden für Kante: {current_edge}")
            raise Exception("No face found containing the current edge.")

        print(f"[RouteFaces] Aktuelles Face: {list(current_face.edges)}")

        # Follow the face using the right-hand rule
        while current_edge in visited_edges or (current_edge[1], current_edge[0]) in visited_edges or current_edge in fails:
            current_edge = next_edge_on_face(current_face, current_edge)
            print(f"[RouteFaces] Nächste Kante auf Face: {current_edge}")

        # Update visited edges and move to the next node
        visited_edges.add(current_edge)
        current_node = current_edge[1]

        # Cycle detection based on visited nodes
        if current_node in visited_nodes:
            print(f"[RouteFaces] Cycle detected at node: {current_node}")
            return (True, hops, switches, detour_edges, path)

        visited_nodes.add(current_node)
        path.append(current_node)

        # Update statistics
        hops += 1
        if current_edge in fails:
            switches += 1
            detour_edges.append(current_edge)

        print(f"[RouteFaces] Pfad bisher: {path}")
        print(f"[RouteFaces] Besuchte Kanten: {visited_edges}")

        # Cycle detection based on excessive hops or switches
        if hops > 3 * n or switches > k * n:
            print("[RouteFaces] Cycle detected due to excessive hops or switches.")
            return (True, hops, switches, detour_edges, path)

        # If the current node is not the destination, prepare for the next iteration
        if current_node != d:
            neighbors = list(tree.neighbors(current_node))
            print(f"[RouteFaces] Wechsel zu neuem currentNode: {current_node}")
            neighbors = [v for v in neighbors if (current_node, v) not in fails and (v, current_node) not in fails]
            if not neighbors:
                raise Exception("No valid neighbors available due to failed edges.")
            next_node = neighbors[0]  # Start with any valid neighbor
            current_edge = (current_node, next_node)

    print(f"[RouteFaces] Ziel erreicht: {path}")
    return (False, hops, switches, detour_edges, path)

def draw_tree_with_fails(tree, fails):
    """Draws the tree with edges in blue and failed edges in red, using node positions."""
    pos = {node: tree.nodes[node]['pos'] for node in tree.nodes}  # Extract positions from nodes

    plt.figure(figsize=(10, 8))

    # Draw all edges in blue
    nx.draw_networkx_edges(tree, pos, edge_color='blue')

    # Draw failed edges in red
    failed_edges = [(u, v) for u, v in fails if tree.has_edge(u, v)]
    nx.draw_networkx_edges(tree, pos, edgelist=failed_edges, edge_color='red')

    # Draw nodes and labels
    nx.draw_networkx_nodes(tree, pos, node_color='lightgray', node_size=500)
    nx.draw_networkx_labels(tree, pos)

    plt.title("Tree with Failed Edges")
    plt.show()
