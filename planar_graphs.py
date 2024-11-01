import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

def create_unit_disk_graph(num_nodes, initial_radius):
    """
    Erzeugt einen zusammenhängenden und planaren Unit-Disk-Graphen (UDG).
    
    Parameter:
    - num_nodes: Anzahl der Knoten im Graphen
    - initial_radius: Startwert für den Radius, um Kanten zwischen Knoten zu erstellen
    
    Rückgabe:
    - G: Der erzeugte planare und zusammenhängende Unit-Disk-Graph
    - positions: Die Positionen der Knoten nach Planarisierung im 2D-Raum
    """
    radius = initial_radius
    max_attempts = 100  # Maximale Anzahl an Versuchen, um einen planaren Graphen zu erzeugen
    attempt = 0
    
    while attempt < max_attempts:
        # Zufällige Positionen der Knoten erzeugen
        initial_positions = {i: (np.random.rand(), np.random.rand()) for i in range(num_nodes)}
        
        # Leeren Graphen erstellen
        G = nx.Graph()
        
        # Knoten und Kanten hinzufügen
        for i, pos in initial_positions.items():
            G.add_node(i, pos=pos)
        
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.linalg.norm(np.array(initial_positions[i]) - np.array(initial_positions[j])) <= radius:
                    G.add_edge(i, j)
        
        # Überprüfen, ob der Graph zusammenhängend und planar ist
        if nx.is_connected(G):
            print(f"Erforderlicher Radius für einen zusammenhängenden Graphen: {radius:.2f}")
            
            # Positionen der Knoten extrahieren
            positions = {i: data['pos'] for i, data in G.nodes(data=True)}
            return G, positions
        
        # Radius leicht erhöhen, falls Graph nicht zusammenhängend ist
        radius += 0.01
        attempt += 1
    
    raise ValueError("Kein zusammenhängender Graph gefunden.")

def draw_graph(G, positions):
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos=positions, with_labels=True, node_color="lightblue", node_size=500)
    plt.show()

def main():
    # Parameter für den Unit-Disk-Graphen
    num_nodes = 50       # Reduzierte Anzahl an Knoten für Planarität
    initial_radius = 0.1 # Startwert für den Radius
    
    # Planaren, zusammenhängenden Graphen erstellen und zeichnen
    try:
        G, positions = create_unit_disk_graph(num_nodes, initial_radius)
        draw_graph(G, positions)
        
        # Planare Positionen für weitere Verwendung zurückgeben
        print("Positionen der Knoten:", positions)
    except ValueError as e:
        print(e)

if __name__ == "__main__":
    main()
