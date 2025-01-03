import sys
from typing import List, Any, Union

import networkx as nx
import numpy as np
import itertools
import random
import time
import glob
from objective_function_experiments import *
from trees import one_tree_pre
from routing import RouteOneTree, RouteWithOneCheckpointOneTree
from trees_with_cp import one_tree_with_random_checkpoint_pre
DEBUG = True

# Data structure containing the algorithms under
# scrutiny. Each entry contains a name and a pair
# of algorithms.
#
# The first algorithm is used for any precomputation
# to produce data structures later needed for routing
# on the graph passed along in args. If the precomputation
# fails, the algorithm must return -1.
# Examples for precomputation algorithms can be found in
# arborescences.py
#
# The second algorithm decides how to forward a
# packet from source s to destination d, despite the
# link failures fails using data structures from precomputation
# Examples for precomputation algorithms can be found in
# routing.py
#


#Hier erfolgt die Ausführung von OneTree
algos = {'One Tree OTE': [one_tree_pre, RouteOneTree],
         'One Tree Checkpoint OTE':[one_tree_with_random_checkpoint_pre,RouteWithOneCheckpointOneTree]}

# run one experiment with graph g
# out denotes file handle to write results to
# seed is used for pseudorandom number generation in this run
# returns a score for the performance:
#       if precomputation fails : 10^9
#       if success_ratio == 0: 10^6
#       otherwise (2 - success_ratio) * (stretch + load)
def one_experiment(g, seed, out, algo):
    [precomputation_algo, routing_algo] = algo[:2]
    if DEBUG: print('experiment for ', algo[0])

    # precomputation
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

    # routing simulation (hier gebe ich den routing algorithmus mit)#################################################################################################################################
    print("Start routing")
    stat = Statistic(routing_algo, str(routing_algo))
    stat.reset(g.nodes())
    random.seed(seed)
    t = time.time()
    print("Before simulate graph")
    #hier sage ich dass ich den routing algorithmus simulieren soll (in stat steht welchen routing algorithmus ich ausführen will))#################################################################################################################################
    SimulateGraph(g, True, [stat], f_num, samplesize, precomputation=precomputation)
    print("After simulate")
    rt = (time.time() - t)/samplesize
    success_ratio = stat.succ/ samplesize
    # write results
    if stat.succ > 0:
        if DEBUG: print('success', stat.succ, algo[0])
        # stretch, load, hops, success, routing time, precomputation time
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


# run experiments with AS graphs
# out denotes file handle to write results to
# seed is used for pseudorandom number generation in this run
# rep denotes the number of repetitions in the shuffle for loop
def run_AS(out=None, seed=0, rep=5):
    for i in range(4, 5):
        generate_trimmed_AS(i)
    files = glob.glob('./benchmark_graphs/AS*.csv')
    original_params = [n, rep, k, samplesize, f_num, seed, name]
    for x in files:
        random.seed(seed)
        kk = int(x[-5:-4])
        g = nx.read_edgelist(x).to_directed()
        g.graph['k'] = kk
        nn = len(g.nodes())
        mm = len(g.edges())
        ss = min(int(nn / 2), samplesize)
        fn = min(int(mm / 4), f_num)
        fails = random.sample(list(g.edges()), fn)
        g.graph['fails'] = fails
        set_parameters([nn, rep, kk, ss, fn, seed, name + "AS-"])
        shuffle_and_run(g, out, seed, rep, x)
        set_parameters(original_params)

# run experiments with zoo graphs
# out denotes file handle to write results to
# seed is used for pseudorandom number generation in this run
# rep denotes the number of repetitions in the shuffle for loop
def run_zoo(out=None, seed=0, rep=5):
    min_connectivity = 4
    original_params = [n, rep, k, samplesize, f_num, seed, name]
    if DEBUG:
        print('n_before, n_after, m_after, connectivity, degree')
    for i in range(261):
        random.seed(seed)
        g = read_zoo(i, min_connectivity)
        if g is None:
            continue
        kk = nx.edge_connectivity(g)
        nn = len(g.nodes())
        mm = len(g.edges())
        ss = min(int(nn / 2), samplesize)
        fn = min(int(mm / 4), f_num)
        set_parameters([nn, rep, kk, ss, fn, seed, name + "zoo-"])
        #print('parameters', nn, rep, kk, ss, fn, seed)
        shuffle_and_run(g, out, seed, rep, str(i))
        set_parameters(original_params)
        for (algoname, algo) in algos.items():
            index_1 = len(algo) - rep
            index_2 = len(algo)
            print('intermediate result: %s \t %.5E' % (algoname, np.mean(algo[index_1:index_2])))

# shuffle root nodes and run algorithm
def shuffle_and_run(g, out, seed, rep, x):
    random.seed(seed)
    nodes = list(g.nodes())
    random.shuffle(nodes)
    for count in range(rep):
        g.graph['root'] = nodes[count % len(nodes)]
        for (algoname, algo) in algos.items():
            # graph, size, connectivity, algorithm, index,
            out.write('%s, %i, %i, %s, %i' % (x, len(nodes), g.graph['k'], algoname, count))
            algos[algoname] += [one_experiment(g, seed + count, out, algo)]


# run experiments with d-regular graphs
# out denotes file handle to write results to
# seed is used for pseudorandom number generation in this run
# rep denotes the number of repetitions in the secondary for loop
def run_regular(out=None, seed=0, rep=5):
    ss = min(int(n / 2), samplesize)
    fn = min(int(n * k / 4), f_num)
    set_parameters([n, rep, k, ss, fn, seed, name + "regular-"])
    write_graphs()
    for i in range(rep):
        random.seed(seed + i)
        g = read_graph(i)
        random.seed(seed + i)
        for (algoname, algo) in algos.items():
            # graph, size, connectivity, algorithm, index,
            out.write('%s, %i, %i, %s, %i' % ("regular", n, k, algoname, i))
            algos[algoname] += [one_experiment(g, seed + i, out, algo)]


#hier möchte ich dann eine run_planar haben die die vorherigen funktionen zur planaren Graphen Erstellung nutzt

# start file to capture results
def start_file(filename):
    out = open(filename + ".txt", 'w')
    out.write(
        "#graph, size, connectivity, algorithm, index, " +
        "stretch, load, hops, success, " +
        "routing computation time, pre-computation time in seconds\n")
    out.write(
        "#" + str(time.asctime(time.localtime(time.time()))) + "\n")
    return out



#testen ob onetree auch trees baut mit breite mehr als 2
def run_custom(out=None, seed=0, rep=5):
    global f_num 

    original_params = [n, rep, k, samplesize, f_num, seed, name]
    graphs = []
    fails = []

    graph1, fail1 = create_custom_graph()

    graphs.append(graph1)
    fails.append(fail1) 

    for i in range(0, len(graphs)):
        f_num = len(fails[i]) # How many failed edges we selected in our create_custom_graph
        PG = nx.nx_pydot.write_dot(graphs[i] , "./customOneTree/custom_multipletrees_"+ str(i))
        print("Fails : ", fails[i])
        random.seed(seed)
        kk = 5
        g = graphs[i]
        g.graph['k'] = kk
        nn = len(g.nodes())
        mm = len(g.edges())
        ss = min(int(nn / 2), samplesize)
        fn = min(int(mm / 2), f_num)
        print("Minimum fn: ", fn, "MM/2: ", mm/2 , "f_num :", f_num )
        fails = fails[i]
        g.graph['fails'] = fails
        set_parameters([nn, rep, kk, ss, fn, seed, name + "CUSTOM"])
        print("Global f_num : ", f_num )
        shuffle_and_run(g, out, seed, rep, graphs[i])
        print("Global f_num after run : ", f_num )
        set_parameters(original_params)
        print("Global f_num after reset : ", f_num )

# run experiments
# seed is used for pseudorandom number generation in this run
# switch determines which experiments are run
def experiments(switch="all", seed=0, rep=100):
    if switch in ["regular", "all"]:
       out = start_file("results/benchmark-regular-onetree-" + str(n) + "-" + str(k)  + "-" + str(i))
       run_regular(out=out, seed=seed, rep=rep)
       out.close()

    if switch in ["custom", "all"]:
        #hier steht wo die ergebnisse des durchlaufs gespeichert werden : results/benchmark-cusutom-5.txt
        out = start_file("results/benchmark-custom-onetree-" + str(k))
        run_custom(out=out, seed=seed, rep=rep)
        out.close()


    if switch in ["zoo", "all"]:
        out = start_file("results/benchmark-zoo-onetree-" + str(k))
        run_zoo(out=out, seed=seed, rep=rep)
        out.close()

    if switch in ["AS"]:
        out = start_file("results/benchmark-as_seed_onetree-" + str(seed))
        run_AS(out=out, seed=seed, rep=rep)
        out.close()

    print()
    for (algoname, algo) in algos.items():
        print('%s \t %.5E' % (algoname, np.mean(algo[2:])))
    print("\nlower is better")



if __name__ == "__main__":
    f_num = 0
    for i in range(1, 50):
        f_num = 100 + f_num  # Anzahl der fehlgeschlagenen Verbindungen
        n = 60              # Anzahl der Knoten
        k = 5               # Basis-Konnektivität
        samplesize = 1      # Anzahl der Quellen, die zu einem Ziel weitergeleitet werden sollen
        rep = 2             # Anzahl der Experimente
        switch = 'all'      # Bestimmt, welche Experimente ausgeführt werden
        seed = 0            # Seed für den Zufallszahlengenerator
        name = "benchmark-" # Präfix für Ergebnisdateien
        short = None        # Falls true, werden nur kleine Zoo-Graphen (< 25 Knoten) ausgeführt
        start = time.time()
        print(time.asctime(time.localtime(start)))
        print("[main] i : ", i)
        
        # Falls Kommandozeilenargumente angegeben werden
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

        random.seed(seed)
        set_parameters([n, rep, k, samplesize, f_num, seed, "benchmark-"])

        # Aufruf der experiments-Funktion mit den Variablen f_num, n, rep und i (als main_loop_index)
        experiments(switch=switch, seed=seed, rep=rep, num_nodes=n, f_num=f_num, main_loop_index=i)
        
        end = time.time()
        print("time elapsed", end - start)
        print("start time", time.asctime(time.localtime(start)))
        print("end time", time.asctime(time.localtime(end)))

