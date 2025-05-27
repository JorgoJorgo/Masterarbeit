import networkx as nx
import random
import time

from networkx import node_connectivity
from arborescences import reset_arb_attribute
from cut_algorithms import multipleTrees_with_cuts_extended_pre, multipleTrees_with_cuts_pre, squareOne_with_cuts_pre
from extra_links import DegreeMaxDAG, GreedyMaximalDAG, RouteDetCircSkip
from faces import route_faces_with_paths, route_greedy_faces_with_paths, route_greedy_perimeter
from objective_function_experiments import *
from planar_graphs import apply_delaunay_triangulation, apply_gabriel_graph, create_unit_disk_graph
from trees import multiple_trees_pre, one_tree_pre
from routing import PrepareSQ1, RouteDetCirc, RouteMultipleTrees, RouteOneTree, RouteSQ1, RouteWithOneCheckpointGREEDYMultipleTrees, RouteWithOneCheckpointMultipleTrees, RouteWithOneCheckpointOneTree, RouteWithTripleCheckpointMultipleTrees, RouteWithTripleCheckpointOneTree, SimulateGraph, Statistic
from masterarbeit_trees_with_cp import multiple_trees_for_faces_extended_pre, multiple_trees_for_faces_pre, multiple_trees_invers_with_betweenness_checkpoint_pre, multiple_trees_invers_with_closeness_checkpoint_pre, multiple_trees_invers_with_degree_checkpoint_extended_pre, multiple_trees_invers_with_degree_checkpoint_pre, multiple_trees_invers_with_middle_checkpoint_pre, multiple_trees_triple_checkpooint_pre, multiple_trees_with_betweenness_checkpoint_pre, multiple_trees_with_closeness_checkpoint_pre, multiple_trees_with_degree_checkpoint_extended_pre, multiple_trees_with_degree_checkpoint_pre, multiple_trees_with_middle_checkpoint_parallel_pre, multiple_trees_with_middle_checkpoint_pre, one_tree_triple_checkpooint_pre, one_tree_with_betweenness_checkpoint_pre, one_tree_with_closeness_checkpoint_pre, one_tree_with_degree_checkpoint_extended_pre, one_tree_with_degree_checkpoint_pre, one_tree_with_middle_checkpoint_pre, one_tree_with_middle_checkpoint_shortest_edp_extended_pre, one_tree_with_middle_checkpoint_shortest_edp_pre
import matplotlib.pyplot as plt
DEBUG = True

# (un)comment algorithms you want to experiment with
algos = {
          'MaxDAG': [DegreeMaxDAG, RouteDetCirc],
        #   'SquareOne':[PrepareSQ1,RouteSQ1],
        
          'One Tree': [one_tree_pre, RouteOneTree],
          'MultipleTrees':[multiple_trees_pre, RouteMultipleTrees],

           'MultipleTrees Random Checkpoint':[multiple_trees_with_middle_checkpoint_pre, RouteWithOneCheckpointMultipleTrees],
          'MultipleTrees Random Checkpoint Parallel':[multiple_trees_with_middle_checkpoint_parallel_pre, RouteWithOneCheckpointMultipleTrees],
          'MultipleTrees Closeness Checkpoint':[multiple_trees_with_closeness_checkpoint_pre, RouteWithOneCheckpointMultipleTrees],
          'MultipleTrees Betweenness Checkpoint':[multiple_trees_with_betweenness_checkpoint_pre, RouteWithOneCheckpointMultipleTrees],
          'MultipleTrees Degree Checkpoint':[multiple_trees_with_degree_checkpoint_pre, RouteWithOneCheckpointMultipleTrees],

          'MultipleTrees Inverse Middle Checkpoint':[multiple_trees_invers_with_middle_checkpoint_pre, RouteWithOneCheckpointMultipleTrees],
          'MultipleTrees Inverse Middle Greedy Checkpoint':[multiple_trees_invers_with_middle_checkpoint_pre, RouteWithOneCheckpointGREEDYMultipleTrees],
          'MultipleTrees Inverse Degree Checkpoint':[multiple_trees_invers_with_degree_checkpoint_pre, RouteWithOneCheckpointMultipleTrees],
          'MultipleTrees Inverse Betweenness Checkpoint':[multiple_trees_invers_with_betweenness_checkpoint_pre, RouteWithOneCheckpointMultipleTrees],
          'MultipleTrees Inverse Closeness Checkpoint':[multiple_trees_invers_with_closeness_checkpoint_pre, RouteWithOneCheckpointMultipleTrees],
        
          'MultipleTrees Inverse Degree Checkpoint Extended':[multiple_trees_invers_with_degree_checkpoint_extended_pre, RouteWithOneCheckpointMultipleTrees],
          'MultipleTrees Degree Checkpoint Extended':[multiple_trees_with_degree_checkpoint_extended_pre, RouteWithOneCheckpointMultipleTrees],
          'One Tree Shortest EDP Checkpoint Extended PE': [one_tree_with_middle_checkpoint_shortest_edp_extended_pre, RouteWithOneCheckpointOneTree],
          'One Tree Degree Checkpoint Extended PE': [one_tree_with_degree_checkpoint_extended_pre, RouteWithOneCheckpointOneTree],

          'One Tree Middle Checkpoint PE': [one_tree_with_middle_checkpoint_pre, RouteWithOneCheckpointOneTree],
          'One Tree Degree Checkpoint PE': [one_tree_with_degree_checkpoint_pre, RouteWithOneCheckpointOneTree],
          'One Tree Betweenness Checkpoint PE': [one_tree_with_betweenness_checkpoint_pre, RouteWithOneCheckpointOneTree],
          'One Tree Closeness Checkpoint PE': [one_tree_with_closeness_checkpoint_pre, RouteWithOneCheckpointOneTree],
          'One Tree Shortest EDP Checkpoint PE': [one_tree_with_middle_checkpoint_shortest_edp_pre, RouteWithOneCheckpointOneTree],
        
          'Triple Checkpoint OneTree': [one_tree_triple_checkpooint_pre,RouteWithTripleCheckpointOneTree],
          'Triple Checkpoint MultipleTrees': [multiple_trees_triple_checkpooint_pre,RouteWithTripleCheckpointMultipleTrees],
        
          'SquareOne Cuts': [squareOne_with_cuts_pre, route_faces_with_paths],
          'MultipleTrees Cuts': [multipleTrees_with_cuts_pre, route_faces_with_paths],
          'MultipleTrees Cuts Greedy': [multipleTrees_with_cuts_pre, route_greedy_faces_with_paths],
          'MultipleTrees Cuts Extended': [multipleTrees_with_cuts_extended_pre, route_faces_with_paths],
          'MultipleTrees Faces': [multiple_trees_for_faces_pre, route_faces_with_paths],
        #   'MultipleTrees Faces Extended': [multiple_trees_for_faces_extended_pre, route_faces_with_paths],

        
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
        print("[ERROR] Precomputation failed, aborting experiment!")
        out.write(', %f, %f, %f, %f, %f, %f\n' %
                  (float('inf'), float('inf'), float('inf'), 0, 0, pt))
        score = 1000*1000*1000
        return score

    print("Start routing")
    if routing_algo == RouteDetCircSkip: # or routing_algo == KeepForwardingRouting:# required by the DAG algorithm (taken from old experiment files)
        g_orig = g.to_undirected()
        stat = Statistic(routing_algo, str(routing_algo), g_orig)   
    else:
        stat = Statistic(routing_algo, str(routing_algo))
    stat.reset(g.nodes())
    random.seed(seed)
    t = time.time()
    #print("[one_experiment] fails:", g.graph['fails'])
    if attack == "CLUSTER":
        SimulateGraph(g, True, [stat], f_num, samplesize, precomputation=precomputation,targeted=True)
    else:
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
    node_list = list(g.nodes())
    random.shuffle(node_list)
    #g.graph['root'] = nodes[count % len(nodes)]
    g.graph['root']= node_list[0]
    print("[planar_experiments] root:", g.graph['root'])
    print("[planar_experiments] len(edges):", len(g.edges()))
    
    for (algoname, algo) in algos.items():
        
        if(algoname in ["One Tree PE", "MaxDAG", "MultipleTrees","SquareOne"] ): # since algorithms without GeoRouting do not require the properties of planar embeddings
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
    Converts a PlanarEmbedding graph back into a normal networkx graph.
    Nodes and edges are copied, including node positions.
    """
    G = nx.Graph()
    
    # Copy the nodes and their positions (if available)
    for node in planar_embedding.nodes():
        pos = planar_embedding.nodes[node].get('pos')
        if pos is not None:
            G.add_node(node, pos=pos)
        else:
            G.add_node(node)
    
    # Copy the edges
    for edge in planar_embedding.edges():
        G.add_edge(*edge)
    
    return G


def convert_to_planar_embedding(graph):
    """
    Converts a planar graph into a PlanarEmbedding structure.
    """
    is_planar, embedding = nx.check_planarity(graph)
    if not is_planar:
        raise ValueError("Graph is not planar and cannot be converted into a PlanarEmbedding.")
    # Transfer node positions into the PlanarEmbedding object
    for node, data in graph.nodes(data=True):
        embedding.add_node(node, **data)
    return embedding


# Function for targeted attacks on edges around clusters, adapted for PlanarEmbedding
def targeted_attacks_against_clusters(g, f_num):
    candidate_links_to_fail = list()
    links_to_fail = list()
    clustering_coefficients = nx.clustering(g)

    # Iterate through all nodes and select only those with a clustering coefficient > 0
    for (v, cc) in clustering_coefficients.items():
        if cc == 0.0:
            continue
        neighbors = list(g.neighbors(v))  # For PlanarEmbedding, neighbors() should work
        for w in neighbors:
            edge = (v, w) if v < w else (w, v)
            if edge not in candidate_links_to_fail:
                candidate_links_to_fail.append(edge)

    # Select up to f_num bidirectional edges to be disabled
    if len(candidate_links_to_fail) > f_num:
        links_to_fail = random.sample(candidate_links_to_fail, f_num)
    else:
        links_to_fail.extend(candidate_links_to_fail)

    # Ensure all fails are in consistent order
    links_to_fail = [tuple(sorted(edge)) for edge in links_to_fail]

    # Check if all fails are valid edges in the graph
    invalid_fails = [edge for edge in links_to_fail if edge not in g.edges()]
    if invalid_fails:
        print("[targeted_attacks] Warning: Some fails are not valid edges in the graph.")
        print("Invalid fails:", invalid_fails)
        input("Check the error list")

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

        # Select only specific graphs
        if g is None or (len(g.nodes) < 60 and len(g.nodes) > 90):
            continue

        print("Len(g) = ", len(g.nodes))
        kk = nx.edge_connectivity(g)  # Calculate connectivity
        nn = len(g.nodes())

        if nn < 200:
            print("Suitable graph")
            mm = len(g.edges())
            ss = min(int(nn / 2), samplesize)

            # Calculate f_num based on `i` and `kk`
            f_num = i * kk
            fn = min(int(mm / 4), f_num)
            if fn == int(mm / 4):
                print("SKIP ITERATION")
                continue
            print("Number of failures (f_num): ", f_num)
            print("Number of failures (fn): ", fn)

            # Check if the graph is planar
            is_planar, planar_embedding = nx.check_planarity(g)
            if not is_planar:
                print(f"Graph {graph_index} is not planar, skipping.")
                continue

            # Convert the planar graph into a PlanarEmbedding structure
            planar_graph = nx.Graph(planar_embedding)
            planar_embedding = convert_to_planar_embedding(planar_graph)

            # Add positions to the nodes
            pos = nx.planar_layout(planar_graph)
            nx.set_node_attributes(planar_embedding, pos, 'pos')

            # Create the fails based on the selected attack type
            if attack == "RANDOM":
                print("Selected failure pattern: RANDOM")
                fails = random.sample(list(planar_embedding.edges()), min(len(planar_embedding.edges()), fn))
            elif attack == "CLUSTER":
                print("Selected failure pattern: CLUSTER")
                fails = targeted_attacks_against_clusters(planar_embedding, fn)
            else:
                raise ValueError("Unknown attack type: " + attack)

            # Set connectivity and store the fails in the graph
            planar_embedding.graph['k'] = kk
            planar_embedding.graph['fails'] = fails

            # Check if all fails are valid edges in the graph
            invalid_fails = [edge for edge in fails if edge not in planar_embedding.edges()]
            if invalid_fails:
                print("[run_zoo] Warning: Some fails are not valid edges in the graph.")
                print("Invalid fails:", invalid_fails)
                input("Check the error list")

            set_parameters([nn, rep, kk, ss, fn, seed, name + "zoo-"])
            print("[run_zoo] Parameters:")
            print("Node Number: ", nn)
            print("Connectivity: ", kk)
            print("Failure Number: ", fn)
            print("Fails: ", len(fails))

            # Shuffle and run experiments
            shuffle_and_run(planar_embedding, out, seed, rep, str(graph_index))
            set_parameters(original_params)

            # Output intermediate results
            for (algoname, algo) in algos.items():
                index_1 = len(algo) - rep
                index_2 = len(algo)
                print('intermediate result: %s \t %.5E' % (algoname, np.mean(algo[index_1:index_2])))

# Adjustment of the run_planar function
def run_planar(out=None, seed=0, rep=5, method="gabriel", num_nodes=50, f_num=0):
    random.seed(seed)
    print("[run_planar] Method:", method)
    # Create the Unit-Disk-Graph with the desired number of nodes
    print("Create Unit-Disk-Graph...")
    G = create_unit_disk_graph(num_nodes)
    #print("Graph created:", G)
    print("Number of nodes:", len(G.nodes()), "Number of edges:", len(G.edges()))
    #draw_graph_with_positions(G)
    # Choose the planarization method
    if method.lower() == "delaunay":
        print("Apply Delaunay triangulation...")
        planar_graph = apply_delaunay_triangulation(G)
        print("Delaunay triangulation completed. Nodes:", len(planar_graph.nodes()), "Edges:", len(planar_graph.edges()))
    elif method.lower() == "gabriel":
        print("Apply Gabriel graph...")
        planar_graph = apply_gabriel_graph(G)
        print("Gabriel graph completed. Nodes:", len(planar_graph.nodes()), "Edges:", len(planar_graph.edges()))
    else:
        raise ValueError("Unknown method for planarization")

    # Convert the graph into a PlanarEmbedding structure
    #print("Convert to PlanarEmbedding...")
    planar_embedding = convert_to_planar_embedding(planar_graph)
    #print("PlanarEmbedding completed. Nodes:", len(planar_embedding.nodes()), "Edges:", len(planar_embedding.edges()))

    # Create the fails based on the selected attack type
    if attack == "RANDOM":
        print("Selected failure pattern: RANDOM")
        fails = random.sample(list(planar_embedding.edges()), min(len(planar_embedding.edges()), f_num))
    elif attack == "CLUSTER":
        print("Selected failure pattern: CLUSTER")
        fails = targeted_attacks_against_clusters(planar_embedding, f_num)
    else:
        raise ValueError("Unknown attack type: " + attack)

    # Set connectivity and store the fails in the graph
    #print("Calculate connectivity...")
    
    
    #fails_to_append = ((2,9),(9,2),(0,9),(9,0))
    fails_to_append = () # Here you can add your own extra failures
    for fail in fails_to_append:
        fails.append(fail)
        f_num = f_num +1

    planar_embedding.graph['k'] = node_connectivity(planar_graph)
    planar_embedding.graph['fails'] = fails
    #print("[run_planar] planar_embedding.graph['fails']", planar_embedding.graph['fails'])

    # Check if all fails are valid edges in the graph
    #print("Check the error list...")
    invalid_fails = [edge for edge in fails if edge not in planar_embedding.edges()]
    if invalid_fails:
        print("[run_planar] Warning: Some fails are not valid edges in the graph.")
        print("Invalid fails:", invalid_fails)
        input("Check the error list")

    # Debug information
    print("[run_planar] Number of fails: ", len(fails))
    print("[run_planar] Fails: ", fails)
    #print("[run_planar] Fails: ", fails)

    # Run the experiments
    #print("Start experiments...")
    shuffle_and_run(planar_embedding, out, seed, rep, method)
    #print("[run_planar] Checkpoint END")

def draw_graph_with_positions(G, title="Graph"):
    """
    Draws a graph with stored node positions.
    """
    pos = nx.get_node_attributes(G, 'pos')  # Retrieves the positions of the nodes
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_size=100, node_color="skyblue", edge_color="gray")
    plt.title(title)
    plt.show()

def experiments(switch="all", seed=33, rep=100, num_nodes=60, f_num=0, main_loop_index=0):
###########################################################################################################################################################
    
    method = "delaunay" #CHANGE THE METHOD HERE
    #method = "gabriel"

###########################################################################################################################################################

    if switch in ["planar", "all"]:
        
        filename = f"results/benchmark-planar-{method.lower()}-{attack}-FR{main_loop_index}"
        out = start_file(filename)
        
        for i in range(rep):
            run_planar(out=out, seed=seed, rep=rep, method=method, num_nodes=num_nodes, f_num=f_num)

        out.close()
    
    if switch in ["zoo", "all"]:
        filename = f"results/benchmark-zoo-{method.lower()}-{attack}-FR{main_loop_index}"
        out = start_file(filename)

        for i in range(rep):
            run_zoo(out=out, seed=seed, rep=rep)

        out.close()

if __name__ == "__main__":
    
    start_FR = 1      #Starting value to skip the initial experiments, as algorithms only encounter problems later
    
    if len(sys.argv) > 7:
            start_FR = int(sys.argv[7])
            print("[main] start_fr:",start_FR)
    
    f_num = 3*start_FR #with each execution of the experiment, 4 failures are added
    
    for i in range(start_FR, 42):
        f_num = 3 + f_num
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

        #for complete randomness:
        #seed = int(time.time())
        random.seed(seed)
        set_parameters([n, rep, k, samplesize, f_num, seed, "benchmark-"])

        experiments(switch=switch, seed=seed, rep=rep, num_nodes=n, f_num=f_num, main_loop_index=i)
        
        end = time.time()
        print("time elapsed", end - start)
        print("start time", time.asctime(time.localtime(start)))
        print("end time", time.asctime(time.localtime(end)))
