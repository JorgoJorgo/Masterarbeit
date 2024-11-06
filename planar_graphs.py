import random
import networkx as nx
import numpy as np
from scipy.spatial import Delaunay

#from clustered_experiments import shuffle_and_run WENN ICH MIT CLUSTERED FAILURES RUNNEN WILL

from one_tree_experiments import shuffle_and_run


def create_unit_disk_graph(num_nodes, initial_radius=0.1):
    """
    Erzeugt einen zusammenhängenden Unit-Disk-Graph mit Positionsattributen.
    """
    radius = initial_radius
    max_attempts = 100
    attempt = 0
    while attempt < max_attempts:
        initial_positions = {i: (np.random.rand(), np.random.rand()) for i in range(num_nodes)}
        G = nx.Graph()
        for i, pos in initial_positions.items():
            G.add_node(i, pos=pos)
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                if np.linalg.norm(np.array(initial_positions[i]) - np.array(initial_positions[j])) <= radius:
                    G.add_edge(i, j)
        if nx.is_connected(G):
            return G
        radius += 0.01
        attempt += 1
    raise ValueError("Kein zusammenhängender Graph gefunden.")

def apply_delaunay_triangulation(G):
    """
    Wendet die Delaunay-Triangulation auf die gespeicherten Knotenpositionen an.
    """
    positions = {i: G.nodes[i]['pos'] for i in G.nodes}
    points = np.array(list(positions.values()))
    tri = Delaunay(points)
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for simplex in tri.simplices:
        for i in range(len(simplex)):
            for j in range(i + 1, len(simplex)):
                node1, node2 = simplex[i], simplex[j]
                H.add_edge(node1, node2)
    return H

def apply_gabriel_graph(G):
    """
    Wendet den Gabriel-Graph auf die gespeicherten Knotenpositionen an.
    """
    positions = {i: G.nodes[i]['pos'] for i in G.nodes}
    H = nx.Graph()
    H.add_nodes_from(G.nodes(data=True))
    for i in G.nodes:
        for j in G.nodes:
            if i < j:
                dist = np.linalg.norm(np.array(positions[i]) - np.array(positions[j]))
                midpoint = (np.array(positions[i]) + np.array(positions[j])) / 2
                if dist <= 1.0 and all(
                    np.linalg.norm(np.array(positions[k]) - midpoint) >= dist / 2
                    for k in G.nodes if k != i and k != j
                ):
                    H.add_edge(i, j)
    return H

def run_planar(out=None, seed=0, rep=5, method="Delaunay", num_nodes=50, f_num=0):
    """
    Führt Experimente mit planaren Graphen durch. Erstellt einen Unit-Disk-Graph,
    der anschließend mit einer Methode wie Delaunay oder Gabriel planarisierbar gemacht wird.
    
    Parameter:
    - out: File-Handle für das Schreiben der Ergebnisse
    - seed: Seed für Zufallszahlengenerator
    - rep: Anzahl der Wiederholungen
    - method: Planarisierungsmethode, entweder "Delaunay" oder "Gabriel"
    - num_nodes: Anzahl der Knoten im Graphen
    - f_num: Anzahl der Kanten, die als fehlgeschlagen markiert werden
    """
    random.seed(seed)
    try:
        # Erstelle den Unit-Disk-Graphen mit der gewünschten Anzahl an Knoten
        G = create_unit_disk_graph(num_nodes)
        
        # Wähle die Planarisierungsmethode
        if method.lower() == "delaunay":
            planar_graph = apply_delaunay_triangulation(G)
        elif method.lower() == "gabriel":
            planar_graph = apply_gabriel_graph(G)
        else:
            raise ValueError("Unbekannte Methode für Planarisierung")

        # Setze die Konnektivität und die fehlgeschlagenen Kanten
        planar_graph.graph['k'] = 5  # Beispiel für Basis-Konnektivität
        fails = random.sample(list(planar_graph.edges()), min(len(planar_graph.edges()), f_num))
        planar_graph.graph['fails'] = fails

        # Führe das Experiment aus
        shuffle_and_run(planar_graph, out, seed, rep, method)
        
    except ValueError as e:
        print("Fehler bei der Erstellung eines zusammenhängenden planaren Graphen:", e)

