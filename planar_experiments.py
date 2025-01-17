import networkx as nx
import random
import time

from networkx import node_connectivity
from arborescences import reset_arb_attribute
from extra_links import DegreeMaxDAG, GreedyMaximalDAG, RouteDetCircSkip
from objective_function_experiments import *
from planar_graphs import apply_delaunay_triangulation, apply_gabriel_graph, create_unit_disk_graph
from trees import multiple_trees_pre, one_tree_pre
from routing import PrepareSQ1, RouteDetCirc, RouteMultipleTrees, RouteOneTree, RouteSQ1, RouteWithOneCheckpointMultipleTrees, RouteWithOneCheckpointOneTree, RouteWithTripleCheckpointMultipleTrees, RouteWithTripleCheckpointOneTree, SimulateGraph, Statistic
from masterarbeit_trees_with_cp import multiple_trees_triple_checkpooint_pre, multiple_trees_with_middle_checkpoint_pre, one_tree_triple_checkpooint_pre, one_tree_with_betweenness_checkpoint_pre, one_tree_with_closeness_checkpoint_pre, one_tree_with_degree_checkpoint_pre, one_tree_with_middle_checkpoint_pre, one_tree_with_middle_checkpoint_shortest_edp_pre
import matplotlib.pyplot as plt
DEBUG = True

algos = {
        #'MaxDAG': [DegreeMaxDAG, RouteDetCirc],
        #'SquareOne':[PrepareSQ1,RouteSQ1],
        #'MultipleTrees':[multiple_trees_pre, RouteMultipleTrees],
        #'MultipleTrees Random Checkpoint':[multiple_trees_with_middle_checkpoint_pre, RouteWithOneCheckpointMultipleTrees],
        #'One Tree PE': [one_tree_pre, RouteOneTree],
        #'One Tree Middle Checkpoint PE': [one_tree_with_middle_checkpoint_pre, RouteWithOneCheckpointOneTree],
        #'One Tree Degree Checkpoint PE': [one_tree_with_degree_checkpoint_pre, RouteWithOneCheckpointOneTree],
        #'One Tree Betweenness Checkpoint PE': [one_tree_with_betweenness_checkpoint_pre, RouteWithOneCheckpointOneTree],
        #'One Tree Closeness Checkpoint PE': [one_tree_with_closeness_checkpoint_pre, RouteWithOneCheckpointOneTree],
        #'One Tree Shortest EDP Checkpoint PE': [one_tree_with_middle_checkpoint_shortest_edp_pre, RouteWithOneCheckpointOneTree],
        #'Triple Checkpoint OneTree': [one_tree_triple_checkpooint_pre,RouteWithTripleCheckpointOneTree],
        'Triple Checkpoint MultipleTrees': [multiple_trees_triple_checkpooint_pre,RouteWithTripleCheckpointMultipleTrees]
        }

def one_experiment(g, seed, out, algo):
    [precomputation_algo, routing_algo] = algo[:2]
    if DEBUG: print('experiment for ', algo[0])
    
    #print("[one_experiment] fails:",g.graph['fails'])
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
    if routing_algo == RouteDetCircSkip: # or routing_algo == KeepForwardingRouting:# braucht der DAG Algorithmus (entnommen aus alten Experiment Dateien)
        g_orig = g.to_undirected()
        stat = Statistic(routing_algo, str(routing_algo), g_orig)
    else:
        stat = Statistic(routing_algo, str(routing_algo))
    stat.reset(g.nodes())
    random.seed(seed)
    t = time.time()
    #print("[one_experiment] fails:", g.graph['fails'])
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
    print("[planar_experiments] len(edges):", len(g.edges()))
    
    for (algoname, algo) in algos.items():
        
        if(algoname in ["One Tree PE", "MaxDAG", "MultipleTrees","SquareOne"] ): #da Algorithmen ohne GeoRouting die Eigenschaften der Planar Embeddings nicht benötigen
            converted_back_to_graph = convert_planar_embedding_to_graph(g)
            converted_back_to_graph.graph['k'] = g.graph['k']
            converted_back_to_graph.graph['fails'] = g.graph['fails']
            converted_back_to_graph.graph['root'] = g.graph['root']
            out.write('%s, %i, %i, %s, %i' % (x, len(nodes), g.graph['k'], algoname, count))
            if algoname == "MaxDAG":
                if not converted_back_to_graph.is_directed():
                    converted_back_to_graph = converted_back_to_graph.to_directed()
                    stat = Statistic(algoname, str(algoname), converted_back_to_graph)
                    

            algos[algoname] += [one_experiment(converted_back_to_graph, seed + count, out, algo)]
        else:
            print("[shuffle_and_run] fails:", g.graph['fails'])
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


# run experiments with zoo graphs
def run_zoo(out=None, seed=0, rep=2, attack="RANDOM", fr=1):
    global f_num
    min_connectivity = 2
    original_params = [n, rep, k, samplesize, f_num, seed, name]
    if DEBUG:
        print('n_before, n_after, m_after, connectivity, degree')

    zoo_list = list(glob.glob("./benchmark_graphs/*.graphml"))

    for graph_index in range(len(zoo_list)):
        random.seed(seed)
        g = read_zoo(graph_index, min_connectivity)

        # Nur spezifische Graphen auswählen
        if g is None or (len(g.nodes) < 60 and len(g.nodes) > 90) :
            continue

        print("Len(g) = ", len(g.nodes))
        kk = nx.edge_connectivity(g)  # Berechnung der Konnektivität
        nn = len(g.nodes())

        if nn < 200:
            print("Passender Graph")
            mm = len(g.edges())
            ss = min(int(nn / 2), samplesize)

            # Berechne f_num basierend auf `i` und `kk`
            f_num = i * kk
            fn = min(int(mm / 4), f_num)
            if fn == int(mm / 4):
                print("SKIP ITERATION")
                continue
            print("Fehleranzahl (f_num): ", f_num)
            print("Fehleranzahl (fn): ", fn)

            # Prüfe, ob der Graph planar ist
            is_planar, planar_embedding = nx.check_planarity(g)
            if not is_planar:
                print(f"Graph {graph_index} ist nicht planar, wird übersprungen.")
                continue

            # Wandle den planaren Graph in eine PlanarEmbedding-Struktur um
            planar_graph = nx.Graph(planar_embedding)
            planar_embedding = convert_to_planar_embedding(planar_graph)

            # Füge Positionen zu den Knoten hinzu
            pos = nx.planar_layout(planar_graph)
            nx.set_node_attributes(planar_embedding, pos, 'pos')

            # Erstelle die Fails basierend auf dem gewählten Angriffstyp
            if attack == "RANDOM":
                print("Ausgewähltes Fehlermuster: RANDOM")
                fails = random.sample(list(planar_embedding.edges()), min(len(planar_embedding.edges()), fn))
            elif attack == "CLUSTER":
                print("Ausgewähltes Fehlermuster: CLUSTER")
                fails = targeted_attacks_against_clusters(planar_embedding, fn)
            else:
                raise ValueError("Unbekannter Angriffstyp: " + attack)

            # Setze die Konnektivität und speichere die Fails im Graph
            planar_embedding.graph['k'] = kk
            planar_embedding.graph['fails'] = fails

            # Überprüfe, ob alle Fails gültige Kanten im Graphen sind
            invalid_fails = [edge for edge in fails if edge not in planar_embedding.edges()]
            if invalid_fails:
                print("[run_zoo] Warnung: Einige Fails sind keine gültigen Kanten im Graphen.")
                print("Ungültige Fails:", invalid_fails)
                input("Checke die Fehler Liste")

            set_parameters([nn, rep, kk, ss, fn, seed, name + "zoo-"])
            print("[run_zoo] Parameter:")
            print("Node Number: ", nn)
            print("Connectivity: ", kk)
            print("Failure Number: ", fn)
            print("Fails: ", len(fails))

            # Shuffle and run experiments
            shuffle_and_run(planar_embedding, out, seed, rep, str(graph_index))
            set_parameters(original_params)

            # Ausgabe der Zwischenergebnisse
            for (algoname, algo) in algos.items():
                index_1 = len(algo) - rep
                index_2 = len(algo)
                print('intermediate result: %s \t %.5E' % (algoname, np.mean(algo[index_1:index_2])))




# Anpassung der run_planar Funktion
def run_planar(out=None, seed=0, rep=5, method="Delaunay", num_nodes=50, f_num=0):
    random.seed(seed)
    
    # Erstelle den Unit-Disk-Graphen mit der gewünschten Anzahl an Knoten
    print("Erstelle Unit-Disk-Graph...")
    G = create_unit_disk_graph(num_nodes)
    #print("Graph erstellt:", G)
    print("Anzahl Knoten:", len(G.nodes()), "Anzahl Kanten:", len(G.edges()))
    #draw_graph_with_positions(G)
    # Wähle die Planarisierungsmethode
    if method.lower() == "delaunay":
        print("Wende Delaunay-Triangulation an...")
        planar_graph = apply_delaunay_triangulation(G)
        print("Delaunay-Triangulation abgeschlossen. Knoten:", len(planar_graph.nodes()), "Kanten:", len(planar_graph.edges()))
    elif method.lower() == "gabriel":
        print("Wende Gabriel-Graph an...")
        planar_graph = apply_gabriel_graph(G)
        print("Gabriel-Graph abgeschlossen. Knoten:", len(planar_graph.nodes()), "Kanten:", len(planar_graph.edges()))
    else:
        raise ValueError("Unbekannte Methode für Planarisierung")

    # Wandelt den Graphen in eine PlanarEmbedding-Struktur um
    #print("Konvertiere in PlanarEmbedding...")
    planar_embedding = convert_to_planar_embedding(planar_graph)
    #print("PlanarEmbedding abgeschlossen. Knoten:", len(planar_embedding.nodes()), "Kanten:", len(planar_embedding.edges()))

    # Erstelle die Fails basierend auf dem gewählten Angriffstyp
    if attack == "RANDOM":
        print("Ausgewähltes Fehlermuster : RANDOM")
        fails = random.sample(list(planar_embedding.edges()), min(len(planar_embedding.edges()), f_num))
    elif attack == "CLUSTER":
        print("Ausgewähltes Fehlermuster : CLUSTER")
        fails = targeted_attacks_against_clusters(planar_embedding, f_num)
    else:
        raise ValueError("Unbekannter Angriffstyp: " + attack)

    # Setze die Konnektivität und speichere die Fails im Graph
    #print("Berechne Konnektivität...")
    
    
    #fails_to_append = ((2,9),(9,2),(0,9),(9,0))
    fails_to_append = () #hier kann man seine eigenen fehler extra rein machen
    for fail in fails_to_append:
        fails.append(fail)
        f_num = f_num +1

    planar_embedding.graph['k'] = node_connectivity(planar_graph)
    planar_embedding.graph['fails'] = fails
    #print("[run_planar] planar_embedding.graph['fails']", planar_embedding.graph['fails'])

    # Überprüfe, ob alle Fails gültige Kanten im Graphen sind
    #print("Überprüfe die Fehlerliste...")
    invalid_fails = [edge for edge in fails if edge not in planar_embedding.edges()]
    if invalid_fails:
        print("[run_planar] Warnung: Einige Fails sind keine gültigen Kanten im Graphen.")
        print("Ungültige Fails:", invalid_fails)
        input("Checke die Fehlerliste")

    # Debug-Informationen
    print("[run_planar] Anzahl der Fails: ", len(fails))
    print("[run_planar] Fails: ", fails)
    #print("[run_planar] Fails: ", fails)

    # Führe die Experimente durch
    #print("Starte Experimente...")
    shuffle_and_run(planar_embedding, out, seed, rep, method)
    #print("[run_planar] Checkpoint END")

    


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
    
    if switch in ["zoo", "all"]:
        filename = f"results/benchmark-zoo-{method.lower()}-{attack}-FR{main_loop_index}"
        out = start_file(filename)

        for i in range(rep):
            run_zoo(out=out, seed=seed, rep=rep)

        out.close()

if __name__ == "__main__":
    start_FR = 5       #Anfangswert um die Anfänglichen Experimente zu skippen, da Algorihtmen erst später Probleme bekommen
    f_num = 5*start_FR #bei jeder Ausführung des Experiments kommen 4 Fehler dazu
    
    for i in range(start_FR, 100):
        f_num = 5 + f_num
        #f_num = 0
        n = 80
        k = 5
        samplesize = 10
        rep = 2
        switch = 'all'
        attack = "CLUSTER" # RANDOM or CLUSTER

        seed = random.randint(1,20000)
        
        start = time.time()
        
        print(time.asctime(time.localtime(start)))
        
        print("[main] i : ", i)
        
        if len(sys.argv) > 1:
            switch = sys.argv[1]
            print("[main] switch:", switch)
        if len(sys.argv) > 2:
            seed = int(sys.argv[2])
            print("[main] seed:",seed)
        if len(sys.argv) > 3:
            rep = int(sys.argv[3])
            print("[main] rep:",rep)
        if len(sys.argv) > 4:
            n = int(sys.argv[4])
            print("[main] n:",n)
        if len(sys.argv) > 5:
            samplesize = int(sys.argv[5])
            print("[main] samplesize:",samplesize)
        if len(sys.argv) > 6:
            attack = str(sys.argv[6])
            print("[main] attack:",attack)

        #für komplette randomness:
        #seed = int(time.time())
        random.seed(seed)
        set_parameters([n, rep, k, samplesize, f_num, seed, "benchmark-"])

        experiments(switch=switch, seed=seed, rep=rep, num_nodes=n, f_num=f_num, main_loop_index=i)
        
        end = time.time()
        print("time elapsed", end - start)
        print("start time", time.asctime(time.localtime(start)))
        print("end time", time.asctime(time.localtime(end)))
