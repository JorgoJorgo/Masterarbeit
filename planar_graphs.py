import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

def create_unit_disk_graph(num_nodes, initial_radius):
    """
    Erzeugt einen zusammenhängenden und planaren Unit-Disk-Graphen (UDG).
    
    Parameter:
    - num_nodes: Anzahl der Knoten im Graphen
    - initial_radius: Startwert für den Radius, um Kanten zwischen Knoten zu erstellen
    
    Rückgabe:
    - G: Der erzeugte planare und zusammenhängende Unit-Disk-Graph
    - positions: Die Positionen der Knoten im 2D-Raum
    """
    radius = initial_radius
    max_attempts = 100  # Maximale Anzahl an Versuchen, um einen planaren und zusammenhängenden Graphen zu erzeugen
    attempt = 0
    
    while attempt < max_attempts:
        # Zufällige Positionen der Knoten im Einheitsquadrat erzeugen (x, y-Werte zwischen 0 und 1)
        initial_positions = {i: (np.random.rand(), np.random.rand()) for i in range(num_nodes)}
        
        # Leeren Graphen erstellen
        G = nx.Graph()
        
        # Knoten mit Positionen hinzufügen
        for i, pos in initial_positions.items():
            G.add_node(i, pos=pos)
        
        # Kanten hinzufügen, wenn der Abstand zwischen zwei Knoten kleiner oder gleich dem Radius ist
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.linalg.norm(np.array(initial_positions[i]) - np.array(initial_positions[j])) <= radius:
                    G.add_edge(i, j)
        
        # Prüfen, ob der Graph zusammenhängend ist
        if nx.is_connected(G):
            print(f"Erforderlicher Radius für einen zusammenhängenden Graphen: {radius:.2f}")
            
            # Positionen der Knoten in einem Dictionary speichern
            positions = {i: data['pos'] for i, data in G.nodes(data=True)}
            return G, positions
        
        # Radius um einen kleinen Betrag erhöhen, falls der Graph nicht zusammenhängend ist
        radius += 0.01
        attempt += 1
    
    # Fehler, falls nach allen Versuchen kein zusammenhängender Graph erstellt werden konnte
    raise ValueError("Kein zusammenhängender Graph gefunden.")

def apply_delaunay_triangulation(G, positions):
    """
    Wendet die Delaunay-Triangulation auf die Knotenpositionen an, um eine planare Struktur zu erzeugen.
    
    Parameter:
    - G: Der ursprüngliche Unit-Disk-Graph
    - positions: Die Positionen der Knoten
    
    Rückgabe:
    - H: Der planare Graph basierend auf der Delaunay-Triangulation
    """
    # Knotenpositionen in ein NumPy-Array konvertieren
    points = np.array(list(positions.values()))
    
    # Delaunay-Triangulation auf die Knotenpositionen anwenden
    tri = Delaunay(points)
    
    # Neuer Graph H mit denselben Knoten wie G
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    
    # Hinzufügen der Kanten aus der Delaunay-Triangulation
    for simplex in tri.simplices:
        # Jedes Simplex ist ein Dreieck, wir verbinden alle Paare von Knoten im Dreieck
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                node1, node2 = simplex[i], simplex[j]
                H.add_edge(node1, node2)
    
    return H

def apply_gabriel_graph(G, positions):
    """
    Wendet den Gabriel-Graphen auf die Knotenpositionen an, um eine planare Struktur zu erzeugen.
    
    Parameter:
    - G: Der ursprüngliche Unit-Disk-Graph
    - positions: Die Positionen der Knoten
    
    Rückgabe:
    - H: Der planare Graph basierend auf dem Gabriel-Graphen
    """
    # Neuer Graph H mit denselben Knoten wie G
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    
    # Gabriel-Kanten hinzufügen: zwei Knoten sind verbunden, wenn der Kreismittelpunkt keine weiteren Knoten enthält
    for i in G.nodes:
        for j in G.nodes:
            if i < j:
                # Abstand zwischen den Knoten i und j berechnen
                dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
                
                # Berechne den Mittelpunkt der Kante (i, j)
                midpoint = (np.array(positions[i]) + np.array(positions[j])) / 2
                
                # Bedingung für die Gabriel-Verbindung: Kein anderer Knoten liegt näher am Mittelpunkt
                if dist <= 1.0 and all(
                    np.linalg.norm(np.array(positions[k]) - midpoint) >= dist / 2
                    for k in G.nodes if k != i and k != j
                ):
                    H.add_edge(i, j)
    
    return H

def draw_graph(G, positions):
    """
    Zeichnet den Graphen G mit den angegebenen Positionen.
    
    Parameter:
    - G: Der darzustellende Graph
    - positions: Dictionary mit den Positionen der Knoten
    """
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos=positions, with_labels=True, node_color="lightblue", node_size=500)
    plt.show()

def main():
    # Parameter für den Unit-Disk-Graphen
    num_nodes = 50       # Anzahl an Knoten
    initial_radius = 0.1 # Startwert für den Radius
    
    # Abfrage der gewünschten Methode (Delaunay oder Gabriel)
    method = input("Bitte wählen Sie die Methode ('DT' für Delaunay-Triangulation, 'GG' für Gabriel-Graph): ").strip().upper()
    
    try:
        # Erzeuge einen zusammenhängenden Unit-Disk-Graphen und bestimme Knotenpositionen
        G, positions = create_unit_disk_graph(num_nodes, initial_radius)
        
        # Auswahl der Planarisierungsmethode
        if method == 'DT':
            # Wendet Delaunay-Triangulation an und gibt einen planaren Graph zurück
            H = apply_delaunay_triangulation(G, positions)
        elif method == 'GG':
            # Wendet den Gabriel-Graph an und gibt einen planaren Graph zurück
            H = apply_gabriel_graph(G, positions)
        else:
            print("Beide Methoden werden ausgeführt")
            # Delaunay-Triangulation und Gabriel-Graph separat erstellen und visualisieren
            F = apply_delaunay_triangulation(G, positions)
            U = apply_gabriel_graph(G, positions)
            draw_graph(F, positions)
            draw_graph(U, positions)
            return
        
        # Planaren Graphen basierend auf der gewählten Methode zeichnen
        draw_graph(H, positions)
        print("Positionen der Knoten:", positions)
        
    except ValueError as e:
        # Fehlerbehandlung, falls kein zusammenhängender Graph gefunden wird
        print(e)

if __name__ == "__main__":
    main()
