import networkx as nx
import random
import time
from objective_function_experiments import *
from planar_graphs import apply_delaunay_triangulation, apply_gabriel_graph, create_unit_disk_graph
from trees import one_tree_pre
from routing import RouteOneTree, RouteWithOneCheckpointOneTree
from masterarbeit_trees_with_cp import one_tree_with_degree_checkpoint_pre, one_tree_with_random_checkpoint_pre
import matplotlib.pyplot as plt
DEBUG = True

algos = {#'One Tree PE': [one_tree_pre, RouteOneTree],
         'One Tree Random Checkpoint PE': [one_tree_with_random_checkpoint_pre, RouteWithOneCheckpointOneTree],
         'One Tree Degree Checkpoint PE': [one_tree_with_degree_checkpoint_pre, RouteWithOneCheckpointOneTree],}

def one_experiment(g, seed, out, algo):
    [precomputation_algo, routing_algo] = algo[:2]
    if DEBUG: print('experiment for ', algo[0])

    reset_arb_attribute(g)
    random.seed(seed)
    t = time.time()
    precomputation = precomputation_algo(g)
    print('Done with precomputation algo')
    pt = time.time() - t
    if precomputation == -1:  # error...
        out.write(', %f, %f, %f, %f, %f, %f\n' %
                  (float('inf'), float('inf'), float('inf'), 0, 0, pt))
        score = 1000*1000*1000
        return score

    print("Start routing")
    stat = Statistic(routing_algo, str(routing_algo))
    stat.reset(g.nodes())
    random.seed(seed)
    t = time.time()
    SimulateGraph(g, True, [stat], f_num, samplesize, precomputation=precomputation)
    print("After simulate")
    rt = (time.time() - t)/samplesize
    success_ratio = stat.succ / samplesize
    if stat.succ > 0:
        if DEBUG: print('success', stat.succ, algo[0])
        out.write(', %i, %i, %i, %f, %f, %f\n' %
                  (np.max(stat.stretch), stat.load, np.max(stat.hops),
                   success_ratio, rt, pt))
        score = (2 - success_ratio) * (np.max(stat.stretch) + stat.load)
    else:
        if DEBUG: print('no success_ratio', algo[0])
        out.write(', %f, %f, %f, %f, %f, %f\n' %
                  (float('inf'), float('inf'), float('inf'), 0, rt, pt))
        score = 1000*1000
    return score

def shuffle_and_run(g, out, seed, rep, x):
    random.seed(seed)
    nodes = list(g.nodes())
    random.shuffle(nodes)
    count = random.randint(1, rep)
    g.graph['root'] = nodes[count % len(nodes)]
    print("[planar_experiments] root:", g.graph['root'])
    for (algoname, algo) in algos.items():
        if(algoname == "One Tree PE"):
            converted_back_to_graph = convert_planar_embedding_to_graph(g)
            converted_back_to_graph.graph['k'] = g.graph['k']
            converted_back_to_graph.graph['fails'] = g.graph['fails']
            converted_back_to_graph.graph['root'] = g.graph['root']
            out.write('%s, %i, %i, %s, %i' % (x, len(nodes), g.graph['k'], algoname, count))
            algos[algoname] += [one_experiment(converted_back_to_graph, seed + count, out, algo)]
        else:

            out.write('%s, %i, %i, %s, %i' % (x, len(nodes), g.graph['k'], algoname, count))
            algos[algoname] += [one_experiment(g, seed + count, out, algo)]

def start_file(filename):
    out = open(filename + ".txt", 'w')
    out.write(
        "#graph, size, connectivity, algorithm, index, " +
        "stretch, load, hops, success, " +
        "routing computation time, pre-computation time in seconds\n")
    out.write(
        "#" + str(time.asctime(time.localtime(time.time()))) + "\n")
    return out

def convert_planar_embedding_to_graph(planar_embedding):
    """
    Konvertiert einen PlanarEmbedding-Graph zurück in einen normalen networkx-Graph.
    Knoten und Kanten werden kopiert, inklusive der Knotenpositionen.
    """
    G = nx.Graph()
    
    # Kopiere die Knoten und ihre Positionen (falls vorhanden)
    for node in planar_embedding.nodes():
        pos = planar_embedding.nodes[node].get('pos')
        if pos is not None:
            G.add_node(node, pos=pos)
        else:
            G.add_node(node)
    
    # Kopiere die Kanten
    for edge in planar_embedding.edges():
        G.add_edge(*edge)
    
    return G


def convert_to_planar_embedding(graph):
    """
    Konvertiert einen planaren Graphen in eine PlanarEmbedding-Struktur.
    """
    is_planar, embedding = nx.check_planarity(graph)
    if not is_planar:
        raise ValueError("Graph ist nicht planar und kann nicht in eine PlanarEmbedding umgewandelt werden.")
    # Übertrage die Knotenpositionen in das PlanarEmbedding-Objekt
    for node, data in graph.nodes(data=True):
        embedding.add_node(node, **data)
    return embedding


# Funktion für gezielte Angriffe auf Kanten um Cluster, angepasst für PlanarEmbedding
def targeted_attacks_against_clusters(g, f_num):
    candidate_links_to_fail = list()
    links_to_fail = list()
    clustering_coefficients = nx.clustering(g)

    # Durchlaufe alle Knoten und wähle nur die mit einem Cluster-Koeffizienten > 0
    for (v, cc) in clustering_coefficients.items():
        if cc == 0.0:
            continue
        neighbors = list(g.neighbors(v))  # Für PlanarEmbedding sollte neighbors() funktionieren
        for w in neighbors:
            edge = (v, w) if v < w else (w, v)
            if edge not in candidate_links_to_fail:
                candidate_links_to_fail.append(edge)

    # Wähle bis zu f_num bidirektionale Kanten, die deaktiviert werden sollen
    if len(candidate_links_to_fail) > f_num:
        links_to_fail = random.sample(candidate_links_to_fail, f_num)
    else:
        links_to_fail.extend(candidate_links_to_fail)

    # Sicherstellen, dass alle Fails in konsistenter Reihenfolge sind
    links_to_fail = [tuple(sorted(edge)) for edge in links_to_fail]

    # Überprüfe, ob alle Fails gültige Kanten im Graphen sind
    invalid_fails = [edge for edge in links_to_fail if edge not in g.edges()]
    if invalid_fails:
        print("[targeted_attacks] Warnung: Einige Fails sind keine gültigen Kanten im Graphen.")
        print("Ungültige Fails:", invalid_fails)
        input("Checke die Fehler Liste")

    return links_to_fail


# Anpassung der run_planar Funktion
def run_planar(out=None, seed=0, rep=5, method="Delaunay", num_nodes=50, f_num=0):
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

        # Wandelt den Graphen in eine PlanarEmbedding-Struktur um
        planar_embedding = convert_to_planar_embedding(planar_graph)

        # Erstelle die Fails basierend auf dem gewählten Angriffstyp
        if attack == "RANDOM":
            print("Ausgewähltes Fehlermuster : RANDOM")
            fails = random.sample(list(planar_embedding.edges()), min(len(planar_embedding.edges()), f_num))
        elif attack  == "CLUSTER":
            print("Ausgewähltes Fehlermuster : CLUSTER")
            fails = targeted_attacks_against_clusters(planar_embedding, f_num)
        else:
            raise ValueError("Unbekannter Angriffstyp: " + attack)

        # Setze die Konnektivität und speichere die Fails im Graph
        planar_embedding.graph['k'] = 5  # Beispiel für Basis-Konnektivität
        planar_embedding.graph['fails'] = fails

        # Überprüfe, ob alle Fails gültige Kanten im Graphen sind
        invalid_fails = [edge for edge in fails if edge not in planar_embedding.edges()]
        if invalid_fails:
            print("[run_planar] Warnung: Einige Fails sind keine gültigen Kanten im Graphen.")
            print("Ungültige Fails:", invalid_fails)
            input("Checke die Fehler Liste")

        # Debug-Informationen
        #print("[run_planar] len(nodes) : ", len(planar_embedding.nodes))
        #print("[run_planar] nodes :", planar_embedding.nodes)
        #print("[run_planar] len(edges) : ", len(planar_embedding.edges))
        #print("[run_planar] edges :", planar_embedding.edges)
        #print("[run_planar] len(fails) : ", len(fails))
        #print("[run_planar] fails :", fails)

        # Führe die Experimente durch
        shuffle_and_run(planar_embedding, out, seed, rep, method)
        
    except ValueError as e:
        print("Fehler bei der Erstellung eines zusammenhängenden planaren Graphen:", e)


def draw_graph_with_positions(G, title="Graph"):
    """
    Zeichnet einen Graphen mit gespeicherten Knotenpositionen.
    """
    pos = nx.get_node_attributes(G, 'pos')  # Holt die Positionen der Knoten
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_size=100, node_color="skyblue", edge_color="gray")
    plt.title(title)
    plt.show()

def experiments(switch="all", seed=33, rep=100, num_nodes=60, f_num=0, main_loop_index=0):
    method = "delaunay"
    
    if switch in ["planar", "all"]:
        
        filename = f"results/benchmark-planar-{method.lower()}-{attack}-FR{main_loop_index}"
        out = start_file(filename)
        
        for i in range(rep):
            run_planar(out=out, seed=seed, rep=rep, method="Delaunay", num_nodes=num_nodes, f_num=f_num)

        out.close()

if __name__ == "__main__":
    f_num = 0
    for i in range(1, 50):
        f_num = 3 + f_num
        n = 100
        k = 5
        samplesize = 1
        rep = 3
        switch = 'all'
        attack = "CLUSTER" # RANDOM or CLUSTER

        seed = random.randint(1,20000)
        
        start = time.time()
        
        print(time.asctime(time.localtime(start)))
        
        print("[main] i : ", i)
        
        if len(sys.argv) > 1:
            switch = sys.argv[1]
        if len(sys.argv) > 2:
            seed = int(sys.argv[2])
        if len(sys.argv) > 3:
            rep = int(sys.argv[3])
        if len(sys.argv) > 4:
            n = int(sys.argv[4])
        if len(sys.argv) > 5:
            samplesize = int(sys.argv[5])
        if len(sys.argv) > 6:
            attack = str(sys.argv[6])

        random.seed(seed)
        set_parameters([n, rep, k, samplesize, f_num, seed, "benchmark-"])

        experiments(switch=switch, seed=seed, rep=rep, num_nodes=n, f_num=f_num, main_loop_index=i)
        
        end = time.time()
        print("time elapsed", end - start)
        print("start time", time.asctime(time.localtime(start)))
        print("end time", time.asctime(time.localtime(end)))
