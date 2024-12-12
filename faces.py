def find_direction_to_intersection(current_node, target_node, face, fails):
    """
    Bestimme die Richtung innerhalb eines Faces, um den Zielknoten zu erreichen,
    und überspringe fehlerhafte Kanten.
    """
    neighbors = list(face.neighbors(current_node))
    for neighbor in neighbors:
        # Überspringe fehlerhafte Kanten
        if (current_node, neighbor) in fails or (neighbor, current_node) in fails:
            continue
        # Wenn der Nachbar der Zielknoten ist, wähle ihn
        if neighbor == target_node:
            return neighbor
    # Fallback: Wenn kein direkter Weg gefunden wird (z. B. alle Kanten fehlerhaft)
    return None

def FaceRouting(s, d, fails, faces):
    # Routing-Prozess initialisieren
    print("Routing gestartet für:", s, "->", d)

    # Initialisiere Listen und Zähler
    detour_edges = []  # Liste für Umleitungskanten
    hops = 0  # Anzahl der Hops
    switches = 0  # Anzahl der Wechsel zwischen Faces

    # Prüfe, ob überhaupt Faces vorhanden sind
    if len(faces) == 0:
        print("Fehler: Keine Faces vorhanden. Routing nicht möglich.")
        return False, hops, switches, detour_edges

    # Extrahiere den Hauptgraphen (outer graph)
    main_graph = faces[len(faces)-1]

    # Berechne die Position der imaginären Kante (s -> d)
    try:
        pos_imaginary_edge = (main_graph.nodes[s]['pos'], main_graph.nodes[d]['pos'])
    except KeyError:
        print("Fehler: Positionsinformationen fehlen. Routing nicht möglich.")
        return False, hops, switches, detour_edges

    current_node = s
    last_node = None

    while current_node != d:
        # Finde das aktuelle Face basierend auf dem Knoten
        current_face = None
        for face in faces[:-1]:  # Schließe den Hauptgraphen aus
            if current_node in face:
                current_face = face
                break

        if current_face is None:
            print("Kein gültiges Face gefunden. Routing fehlgeschlagen.")
            return False, hops, switches, detour_edges

        # Iteriere über die Kanten im aktuellen Face und prüfe Schnittpunkte mit der imaginären Kante
        best_intersection = None
        best_edge = None

        for edge in current_face.edges():
            if edge in fails or (edge[1], edge[0]) in fails:
                continue  # Überspringe fehlerhafte Kanten

            pos_edge = (main_graph.nodes[edge[0]]['pos'], main_graph.nodes[edge[1]]['pos'])
            intersection = intersection_point(pos_edge, pos_imaginary_edge)

            if intersection:
                # Aktualisiere den besten Schnittpunkt basierend auf der Nähe zur Zielposition
                if best_intersection is None or euclidean_distance(intersection, pos_imaginary_edge[1]) < euclidean_distance(best_intersection, pos_imaginary_edge[1]):
                    best_intersection = intersection
                    best_edge = edge

        if best_intersection is None:
            print("Keine Schnittpunkte gefunden. Routing fehlgeschlagen.")
            return False, hops, switches, detour_edges

        # Wähle die nächste Richtung basierend auf der Kante des besten Schnittpunkts
        next_node = None
        for node in best_edge:
            if node != current_node:
                next_node = node
                break

        if next_node is None:
            print("Keine gültige Richtung gefunden. Routing fehlgeschlagen.")
            return False, hops, switches, detour_edges

        # Aktualisiere die Routing-Informationen
        detour_edges.append((current_node, next_node))
        hops += 1
        last_node = current_node
        current_node = next_node

        # Prüfe, ob ein Wechsel des Faces stattfindet
        if next_node not in current_face:
            switches += 1

    # Ziel erreicht
    print("Routing abgeschlossen")
    return True, hops, switches, detour_edges



# Find the distance between 2 points
# Used to find the closer point
def euclidean_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

# Helper function to get the intersection point of 2 edges using the position parameters
def intersection_point(pos_edge1, pos_edge2):
    x1, y1 = pos_edge1[0]
    x2, y2 = pos_edge1[1]
    x3, y3 = pos_edge2[0]
    x4, y4 = pos_edge2[1]

    # Calculate the parameters for the line equations of the two edges
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = x2 * y1 - x1 * y2

    a2 = y4 - y3
    b2 = x3 - x4
    c2 = x4 * y3 - x3 * y4

    # Calculate the intersection point
    det = a1 * b2 - a2 * b1

    if det == 0:
        # The edges are parallel, there's no unique intersection point
        return None
    else:
        x = (b1 * c2 - b2 * c1) / det
        y = (a2 * c1 - a1 * c2) / det

        # Check if the intersection point lies within the bounded segment
        if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2) and \
           min(x3, x4) <= x <= max(x3, x4) and min(y3, y4) <= y <= max(y3, y4):
            return x, y
        else:
            return None
        