import statistics
import sys
import networkx as nx
import numpy as np
import itertools
import random
import time
from routing import *

# global variables
seed = 1
n = 10
rep = 1
k = 8
f_num = 40
samplesize = 20
name = "experiment-objective-function"


# set global parameters in this file and in routing_stats
def set_parameters(params):
    #print("params:" , params)
    set_objective_parameters(params)

# set global parameters in this file and in routing_stats
def set_objective_parameters(params):
    global seed, n, rep, k, samplesize, name, f_num
    [n, rep, k, samplesize, f_num, seed, name] = params
    #print("f_num set_obj_params", f_num)
    set_routing_params(params)

# print global parameters in this file and in routing_stats
def print_objective_parameters():
    print(n, rep, k, samplesize, f_num, seed, name)

# objective functions
def measure_dividedbyhops(g, DEBUG=False):
    return measure_obj(g, 'hops', DEBUG=DEBUG)


def measure_load(g, DEBUG=False):
    return measure_obj(g, 'load', DEBUG=DEBUG)


def measure_stretch(g, DEBUG=False):
    return measure_obj(g, 'stretch', DEBUG=DEBUG)


def measure_product(g, DEBUG=False):
    return measure_obj(g, 'product', DEBUG=DEBUG)


# evaluate routes with simulations for sample size
def measure_obj(g, obj, DEBUG=False):
    # calculate x = maximum number of hops/load/stretch with f failures for a set of samplesize source-root pairs
    T = get_arborescence_list(g)
    stat = statistics(RouteDetCirc, "DetCirc")
    success = 0
    for i in range(f_num + 1):
        stat.reset(g.nodes())
        SimulateGraph(g, True, [stat], i, samplesize, tree=T)
        success += stat.succ
    if stat.succ > 0:
        if obj == 'hops':
            return -10000 * success + np.max(stat.hops)
        if obj == 'load':
            return -10000 * success + stat.load
        if obj == 'stretch':
            return -10000 * success + np.max(stat.stretch)
        if obj == 'product':
            return -10000 * success + np.max(stat.stretch) * stat.load
    else:
        return float("inf")


# count the number of independent paths to the root in arborescences T1 and T2
def num_independent_paths(T1, T2, root):
    SP1 = nx.shortest_path(T1, target=root)
    SP2 = nx.shortest_path(T2, target=root)
    count = 0
    for v in T1.nodes():
        if v in SP1 and v in SP2 and set(SP1[v][1:-1]).isdisjoint(set(SP2[v][1:-1])):
            count += 1
    return count


# count the number of independent paths to the root in decomposition associated with g
def num_independent_paths_in_arbs(g):
    root = g.graph['root']
    T = get_arborescence_list(g)
    n = len(g.nodes())
    count = 0
    for T1, T2 in itertools.combinations(T, 2):
        if root in T1.nodes() and root in T2.nodes():
            count += num_independent_paths(T1, T2, root)
        else:
            return 0
    return count

#hier wird in die results  result reingeschrieben
# run experiment for the objective function with the decomposition method,
# string for the method and parameters over a subset only
def experiment_objective_subset(obj_func, method, objstr=None, seed=11, gml=False, torus=False):
    if objstr == None:
        objstr = str(obj_func)
    random.seed(seed)
    filename = "results/" + name + "_objective_" + \
               str(n) + "_" + str(k) + "_" + str(seed) + "_" + objstr + ".txt"
    filename_time = "results/" + name + "_objective_" + \
                    str(n) + "_" + str(k) + "_" + str(seed) + "_" + objstr + "_time.txt"
    if gml:
        filename = "results/" + name + "-gml_failure_objective_" + \
                   str(n) + "_" + str(k) + "_" + str(seed) + "_" + objstr + ".txt"
    if torus:
        filename = "results/" + name + "-torus_failure_objective_" + \
                   str(n) + "_" + str(k) + "_" + str(seed) + "_" + objstr + ".txt"
    outstretch = open(filename, 'a')
    outstretch.write(
        "#n= %d, connectivity= %d, repetitions= %d\n" % (n, k, rep))
    outstretch.write(
        "#graph, before/after, intensity, 'objective', success rate, switches, load, load, max stretch, mean stretch, max hops, mean hops\n")
    outtime = open(filename_time, 'a')
    outtime.write(
        "#n= %d, connectivity= %d, repetitions= %d\n" % (n, k, rep))
    outtime.write("#n, time to compute arborescence, time for swapping in seconds\n")
    stat = Statistic(RouteDetCirc, "DetCirc")
    failure_range = range(f_num, 0, -1)
    data = {i: {'before': {'succ': [], 'hops': []}, 'after': {
        'succ': [], 'hops': []}} for i in failure_range}
    for j in range(rep):
        random.seed(j)
        if gml:
            g = read_zoo(seed, k)
        else:
            g = read_graph(j)
        t_arb = time.time()
        method(g)
        t_arb = time.time() - t_arb
        if num_complete_nodes(g) == n:
            before = obj_func(g)
            T1 = get_arborescence_list(g)
            t_swap = time.time()
            count = greedy_swap_obj(g, obj_func)
            t_swap = time.time() - t_swap
            outtime.write("%i, %.6f, %.6f\n" % (n, t_arb, t_swap))
            after = obj_func(g)
            if before < after:
                print("objective",objstr, "repetition",j, "before", before, "after", after, "t_swap",t_swap, "number of swaps", count, 'has not been optimized')
                sys.exit(-1)
            T2 = get_arborescence_list(g)
            stat.reset(g.nodes())
            fails = g.graph['fails']
            ss = min(samplesize,len(set(connected_component_nodes_with_d_after_failures(g,fails[:f_num],g.graph['root'])))- 1)
            SimulateGraph(g, True, [stat], f_num, ss, tree=T1)
            for f1 in failure_range:
                stat.reset(g.nodes())
                random.seed(j)
                SimulateGraph(g, True, [stat], f1, ss, tree=T1)
                brs = int(stat.succ) / samplesize
                brh = (stat.totalSwitches)
                data[f1]['before']['succ'].append(brs)
                data[f1]['before']['hops'].append(brh)
                outstretch.write("regular, before, %d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n" % (
                    f1, -1 * before, brs, brh, stat.load, stat.load, np.max(stat.stretch), np.mean(stat.stretch),
                    np.max(stat.hops), np.mean(stat.hops)))

                if np.mean(stat.stretch) < 0 and stat.succ > 0:
                    print('before', j, f1, brs, np.max(stat.stretch), np.mean(
                        stat.stretch), np.max(stat.hops), np.mean(stat.hops))
                    print('stretch', stat.stretch)
                    print('hops', stat.hops)
                    sys.exit()
                if (np.max(stat.stretch) > n - 1):
                    print('large stretch, line 273', np.max(stat.stretch))
                    print(stat.stretch)
                    print(stat.succ, 'successes')
                    sys.exit()
                stat.reset(g.nodes())
                random.seed(g.graph['seed'])
                SimulateGraph(g, True, [stat], f1, ss, tree=T2)
                ars = int(stat.succ) / samplesize
                arh = (stat.totalSwitches)
                data[f1]['after']['succ'].append(ars)
                data[f1]['after']['hops'].append(arh)
                outstretch.write("regular, after, %d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n" % (
                    f1, -1 * after, ars, arh, stat.load, stat.load, np.max(stat.stretch), np.mean(stat.stretch),
                    np.max(stat.hops), np.mean(stat.hops)))
                if np.mean(stat.stretch) < 0 and stat.succ > 0:
                    print('after', j, f1, brs, np.max(stat.stretch), np.mean(
                        stat.stretch), np.max(stat.hops), np.mean(stat.hops))
                    print('stretch', stat.stretch)
                    print('hops', stat.hops)
                    sys.exit()
            sys.stdout.flush()
            outstretch.flush()
            if rep == 1:
                break
    outstretch.close()


# run experiment for the objective function with the decomposition method,
# string for the method and parameters
def experiment_objective(obj_func, method, objstr=None, seed=1):
    if objstr == None:
        objstr = str(obj_func)
    random.seed(seed)
    filename = "results/srds-objective_" + \
               str(n) + "_" + str(k) + "_" + str(seed) + "_" + objstr + ".txt"
    outstretch = open(filename, 'a')
    outstretch.write(
        "#n= %d, connectivity= %d, repetitions= %d\n" % (n, k, rep))
    if "independent" in objstr:
        outstretch.write(
            "graph type, before, objective, after, objective\n")
    else:
        outstretch.write(
            "#graph, before/after, intensity, 'objective', success rate, switches, max load, mean load, max stretch, mean stretch, max hops, mean hops\n")
    stat = Statistic(RouteDetCirc, "DetCirc")
    failure_range = [int(n / 10 * i) for i in range(1, 5 * k)]
    data = {i: {'before': {'succ': [], 'hops': []}, 'after': {
        'succ': [], 'hops': []}} for i in failure_range}
    for j in range(rep):
        g = read_graph(j)
        method(g)
        if num_complete_nodes(g) == n:
            before = obj_func(g)
            T1 = get_arborescence_list(g)
            if "independent" in objstr:
                greedy_swap_obj(g, obj_func, max=True)
            else:
                greedy_swap_obj(g, obj_func)
            after = obj_func(g)
            T2 = get_arborescence_list(g)
            print(j, before, after, obj_func)
            if "independent" in objstr:
                outstretch.write("regular, before, %d, after, %d\n" % (before, after))
                continue
            ss = min(samplesize,len(set(connected_component_nodes_with_d_after_failures(g,fails[:max(failure_range)],g.graph['root'])))- 1)
            for f in failure_range:
                stat.reset(g.nodes())
                # , fails=fails) #replace True by False to use fails
                SimulateGraph(g, True, [stat], f, ss, tree=T1)
                brs = int(stat.succ) / n
                brh = (stat.totalSwitches)
                data[f]['before']['succ'].append(brs)
                data[f]['before']['hops'].append(brh)
                #                                  alg, f, succ, switches, load      stretch,    hops
                outstretch.write("regular, before, %d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n" % (
                    f, before, brs, brh, stat.load, stat.load, np.max(stat.stretch), np.mean(stat.stretch),
                    np.max(stat.hops), np.mean(stat.hops)))

                stat.reset(g.nodes())
                SimulateGraph(g, True, [stat], f, ss, tree=T2)  # , fails=fails)
                ars = int(stat.succ) / n
                arh = (stat.totalSwitches)
                data[f]['after']['succ'].append(ars)
                data[f]['after']['hops'].append(arh)

                #                                  alg,f, succ, switches, load      stretch,    hops
                outstretch.write("regular, after, %d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n" % (
                    f, after, ars, arh, stat.load, stat.load, np.max(stat.stretch), np.mean(stat.stretch),
                    np.max(stat.hops), np.mean(stat.hops)))
                if (np.max(stat.stretch) > n - 1):
                    print('large stretch, line 376', np.max(stat.stretch))
                    print(stat.stretch)
                    print(stat.succ, 'successes')
                    sys.exit()
            print(objstr, j, before, after)
            sys.stdout.flush()
            outstretch.flush()

    if "independent" not in objstr:
        for f in failure_range:
            brs = np.mean(data[f]['before']['succ'])
            bsh = np.mean(data[f]['before']['hops'])
            ars = np.mean(data[f]['after']['succ'])
            arh = np.mean(data[f]['after']['hops'])
            print('%d failures, avg before success hops, after success hops %.2f, %.2f, %.2f, %.2f' % (
                f, brs, bsh, ars, arh))
            brs = np.min(data[f]['before']['succ'])
            bsh = np.min(data[f]['before']['hops'])
            ars = np.min(data[f]['after']['succ'])
            arh = np.min(data[f]['after']['hops'])
            print('%d failures, min before success hops, after success hops %.2f, %.2f, %.2f, %.2f' % (
                f, brs, bsh, ars, arh))
            sys.stdout.flush()
            
    outstretch.close()


# return the number of links in the shared risk link group belong to the last two arborescences
def count_SRLG(g, k, SRLG):
    count = 0
    for (u, v) in SRLG:
        if g[u][v]['arb'] in [k - 1, k - 2]:
            count += 1
    return count


# run SLRG experiments for infocom 2019 paper
# seed is used for pseudorandom number generation in this run
# switch determines which experiments are run
def experiment_SRLG(method, name, seed=11):
    random.seed(seed)
    filename = "results/srds-SRLG_" + str(n) + "_" + str(k) + "_" + str(seed) + "_" + name
    outstretch = open(filename + ".txt", 'a')
    outstretch.write(
        "#n= %d, connectivity= %d, repetitions= %d\n" % (n, k, rep))
    outstretch.write(
        "#graph, before/after, random, intensity, SRLG in last arbs, # successes, switches, max load, mean load, max stretch, mean stretch, max hops, mean hops\n")
    stat = Statistic(RouteDetCirc, "DetCirc")
    failure_range = [int(n / 10 * i) for i in range(1, 5 * k)]
    data = {f: {'before': {'random': {'succ': [], 'hops': []}, 'SRLG': {'succ': [], 'hops': []}}, 'after': {
        'random': {'succ': [], 'hops': []}, 'SRLG': {'succ': [], 'hops': []}}} for f in failure_range}
    for f in failure_range:
        for j in range(rep):
            g = read_graph(j)
            edg = list(g.edges())
            SRLG = random.sample(edg, f)
            method(g)
            if num_complete_nodes(g) == n:
                before = count_SRLG(g, k, SRLG)
                T1 = get_arborescence_list(g)

                for (u, v) in SRLG:
                    index = g[u][v]['arb']
                    if index in range(k - 2) and v != g.graph['root']:
                        for vv in g[u]:
                            if vv != g.graph['root'] and (u, vv) not in SRLG and (vv, u) not in SRLG \
                                    and g[u][vv]['arb'] in [k - 1, k - 2]:
                                swap(g, u, v, u, vv)

                after = count_SRLG(g, k, SRLG)
                T2 = get_arborescence_list(g)

                fails = random.sample(edg, f)
                g.graph['fails'] = fails
                stat.reset(g.nodes())
                samplessize = len(set(connected_component_nodes_with_d_after_failures(g,fails[:f],g.graph['root'])))- 1
                SimulateGraph(g, False, [stat], f, samplessize, tree=T1)
                brs = int(stat.succ) / n
                brh = (stat.totalSwitches)
                outstretch.write("regular, before, True, %d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n" % (
                f, before, brs, brh, np.max(
                    stat.load), np.mean(stat.load), np.max(stat.stretch), np.mean(stat.stretch), np.max(stat.hops),
                np.mean(stat.hops)))

                stat.reset(g.nodes())
                g.graph['fails'] = SRLG
                samplessize = len(set(connected_component_nodes_with_d_after_failures(g,fails[:f],g.graph['root'])))- 1
                SimulateGraph(g, False, [stat], f, samplessize, tree=T1)
                bss = int(stat.succ) / n
                bsh = (stat.totalSwitches)
                outstretch.write(
                    "regular, before, False, %d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n" % (
                    f, before, bss, bsh, np.max(
                        stat.load), np.mean(stat.load), np.max(stat.stretch), np.mean(stat.stretch), np.max(stat.hops),
                    np.mean(stat.hops)))

                stat.reset(g.nodes())
                g.graph['fails'] = fails
                samplessize = len(set(connected_component_nodes_with_d_after_failures(g,fails[:f],g.graph['root'])))- 1
                SimulateGraph(g, False, [stat], f, samplessize, tree=T2)
                ars = int(stat.succ)
                arh = (stat.totalSwitches)
                outstretch.write("regular, after, True, %d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n" % (
                f, after, ars, arh, np.max(
                    stat.load), np.mean(stat.load), np.max(stat.stretch), np.mean(stat.stretch), np.max(stat.hops),
                np.mean(stat.hops)))

                stat.reset(g.nodes())
                g.graph['fails'] = SRLG
                samplessize = len(set(connected_component_nodes_with_d_after_failures(g,fails[:f],g.graph['root'])))- 1
                SimulateGraph(g, False, [stat], f, samplessize, tree=T2)

                ass = int(stat.succ) / n
                ash = (stat.totalSwitches)
                outstretch.write("regular, after, False, %d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n" % (
                f, after, ass, ash, np.max(
                    stat.load), np.mean(stat.load), np.max(stat.stretch), np.mean(stat.stretch), np.max(stat.hops),
                np.mean(stat.hops)))
                if (np.max(stat.stretch) > n - 1):
                    print('large stretch, line 462', np.max(stat.stretch))
                    print(stat.stretch)
                    print(stat.succ, 'successes')
                    sys.exit()
                sys.stdout.flush()
                outstretch.flush()

                data[f]['before']['random']['succ'].append(brs)
                data[f]['before']['random']['hops'].append(brh)
                data[f]['before']['SRLG']['succ'].append(bss)
                data[f]['before']['SRLG']['hops'].append(bsh)
                data[f]['after']['random']['succ'].append(ars)
                data[f]['after']['random']['hops'].append(arh)
                data[f]['after']['SRLG']['succ'].append(ass)
                data[f]['after']['SRLG']['hops'].append(ash)
        brs = np.mean(data[f]['before']['random']['succ'])
        brh = np.mean(data[f]['before']['random']['hops'])
        bss = np.mean(data[f]['before']['SRLG']['succ'])
        bsh = np.mean(data[f]['before']['SRLG']['hops'])
        ars = np.mean(data[f]['after']['random']['succ'])
        ash = np.mean(data[f]['after']['random']['hops'])
        ass = np.mean(data[f]['after']['SRLG']['succ'])
        ash = np.mean(data[f]['after']['SRLG']['hops'])
        print('%d avg before %.2f, %.2f, %.2f, %.2f' % (f, brs, bss, brh, bsh))
        print('%d avg after %.2f, %.2f, %.2f, %.2f' % (f, ars, ass, arh, ash))
        sys.stdout.flush()
    outstretch.close()


# run SLRG experiments for infocom 2019 paper with node failures
# seed is used for pseudorandom number generation in this run
# switch determines which experiments are run
def experiment_SRLG_node_failures(method, name, seed=11):
    random.seed(seed)
    filename = "results/srds-SRLG_" + str(n) + "_" + str(k) + "_" + str(seed) + "_" + name
    outstretch = open(filename + ".txt", 'a')
    outstretch.write(
        "#n= %d, connectivity= %d, repetitions= %d\n" % (n, k, rep))
    outstretch.write(
        "#graph, before/after, random, intensity, SRLG in last arbs, # successes, switches, max load, mean load, max stretch, mean stretch, max hops, mean hops\n")
    stat = Statistic(RouteDetCirc, "DetCirc")
    failure_range = range(1, f_num + 1)
    data = {f: {'before': {'random': {'succ': [], 'hops': []}, 'SRLG': {'succ': [], 'hops': []}}, 'after': {
        'random': {'succ': [], 'hops': []}, 'SRLG': {'succ': [], 'hops': []}}} for f in failure_range}
    for f in failure_range:
        for j in range(rep):
            g = read_graph(j)
            edg = list(g.edges())
            SRLG = g.graph['fails'][:f_num]
            method(g)
            if num_complete_nodes(g) == n:
                before = count_SRLG(g, k, SRLG)
                T1 = get_arborescence_list(g)

                for (u, v) in SRLG:
                    index = g[u][v]['arb']
                    if index in range(k - 2) and v != g.graph['root']:
                        for vv in g[u]:
                            if vv != g.graph['root'] and (u, vv) not in SRLG and (vv, u) not in SRLG and g[u][vv][
                                'arb'] in [k - 1, k - 2]:
                                swap(g, u, v, u, vv)

                after = count_SRLG(g, k, SRLG)
                T2 = get_arborescence_list(g)

                stat.reset(g.nodes())

                samplessize = len(set(connected_component_nodes_with_d_after_failures(g,SRLG,g.graph['root'])))- 1
                SimulateGraph(g, False, [stat], f, samplessize, tree=T1)
                brs = int(stat.succ) / n
                brh = (stat.totalSwitches)

                stat.reset(g.nodes())
                SimulateGraph(g, False, [stat], f, samplessize, tree=T2)
                ars = int(stat.succ) / n
                arh = (stat.totalSwitches)

                outstretch.write("regular, before, True, %d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n" % (
                f, before, brs, brh, np.max(
                    stat.load), np.mean(stat.load), np.max(stat.stretch), np.mean(stat.stretch), np.max(stat.hops),
                np.mean(stat.hops)))
                if ars >= brs:
                    outstretch.write(
                        "regular, after, True, %d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n" % (
                        f, after, ars, arh, np.max(
                            stat.load), np.mean(stat.load), np.max(stat.stretch), np.mean(stat.stretch),
                        np.max(stat.hops), np.mean(stat.hops)))
                else:
                    outstretch.write(
                        "regular, after, True, %d, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f\n" % (
                        f, before, brs, brh, np.max(
                            stat.load), np.mean(stat.load), np.max(stat.stretch), np.mean(stat.stretch),
                        np.max(stat.hops), np.mean(stat.hops)))
                    print("success rate suffered")
                sys.stdout.flush()
                outstretch.flush()
                print(method, name, seed, f)

    outstretch.close()
# erstellt die graphen
# generate rep random k-regular graphs with connectivity k using seed and
# write them to file
def write_graphs():
    d = []
    ecc = []
    sp = []
    for i in range(rep):
        g = nx.random_regular_graph(k, n).to_directed()
        while nx.edge_connectivity(g) < k:
            g = nx.random_regular_graph(k, n).to_directed()
        prepare_graph(g,k,0)
        GreedyArborescenceDecomposition(g)
        d.append(depth(g))
        ecc.append(nx.eccentricity(g, 0))
        sp.append(nx.average_shortest_path_length(g))
        s = ''
        for e in g.graph['fails']:
            s = s + str(e[0]) + ' ' + str(e[1]) + '\n'
        f = open('results/' + name + str(seed) + '_graph_' +
                 str(n) + '_' + str(i) + '.txt', 'w')
        f.write(s[:-1])
        f.close()



# def create_custom_graph():
#     g = nx.Graph()
#     max_node = 50
    
#     # Knoten hinzufügen
#     for i in range(1, max_node + 1):
#         g.add_node(i)
    
#     # Kanten hinzufügen
#     edges = [
#         (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10),
#         (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20),
#         (2, 21), (2, 22), (2, 23), (2, 24), (2, 25),
#         (3, 26), (3, 27), (3, 28), (3, 29), (3, 30),
#         (4, 31), (4, 32), (4, 33), (4, 34), (4, 35),
#         (5, 36), (5, 37), (5, 38), (5, 39), (5, 40),
#         (6, 41), (6, 42), (6, 43), (6, 44), (6, 45),
#         (7, 46), (7, 47), (7, 48), (7, 49), (7, 50),
#         (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 17),
#         (17, 18), (18, 19), (19, 20), (20, 21), (21, 22), (22, 23), (23, 24), (24, 25), (25, 26),
#         (26, 27), (27, 28), (28, 29), (29, 30), (30, 31), (31, 32), (32, 33), (33, 34), (34, 35),
#         (35, 36), (36, 37), (37, 38), (38, 39), (39, 40), (40, 41), (41, 42), (42, 43), (43, 44),
#         (44, 45), (45, 46), (46, 47), (47, 48), (48, 49), (49, 50)
#     ]
    
#     for edge in edges:
#         g.add_edge(*edge)
    
#     fails = [
#         (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10),
#         (1, 11), (1, 12), (1, 13), (1, 14), (1, 15), (1, 16), (1, 17), (1, 18), (1, 19), (1, 20),
#         (2, 21), (2, 22), (2, 23), (2, 24), (2, 25), (3, 26), (3, 27), (3, 28), (3, 29), (3, 30)
#     ]
    
#     return g, fails


#n=50, k=5 außer bei [1] k=9, fnum = 30, bei [1]fnum = 6
#n=50, k=5 außer bei [1] k=9, fnum = 40, bei [1]fnum = 7
def create_custom_graph():
    g = nx.Graph()
    max_node = 50
    
    # Knoten hinzufügen
    for i in range(1, max_node + 1):
        g.add_node(i)
    
    # Kanten hinzufügen, um die Konnektivität sicherzustellen
    edges = [
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (1, 10),
        (2, 11), (2, 12), (2, 13), (2, 14), (2, 15),
        (3, 16), (3, 17), (3, 18), (3, 19), (3, 20),
        (4, 21), (4, 22), (4, 23), (4, 24), (4, 25),
        (5, 26), (5, 27), (5, 28), (5, 29), (5, 30),
        (6, 31), (6, 32), (6, 33), (6, 34), (6, 35),
        (7, 36), (7, 37), (7, 38), (7, 39), (7, 40),
        (8, 41), (8, 42), (8, 43), (8, 44), (8, 45),
        (9, 46), (9, 47), (9, 48), (9, 49), (9, 50),
        (10, 11), (10, 12), (10, 13), (10, 14), (10, 15),
        (11, 16), (11, 17), (11, 18), (11, 19), (11, 20),
        (12, 21), (12, 22), (12, 23), (12, 24), (12, 25),
        (13, 26), (13, 27), (13, 28), (13, 29), (13, 30),
        (14, 31), (14, 32), (14, 33), (14, 34), (14, 35),
        (15, 36), (15, 37), (15, 38), (15, 39), (15, 40),
        (16, 41), (16, 42), (16, 43), (16, 44), (16, 45),
        (17, 46), (17, 47), (17, 48), (17, 49), (17, 50),
        (18, 19), (18, 20), (18, 21), (18, 22), (18, 23),
        (19, 24), (19, 25), (19, 26), (19, 27), (19, 28),
        (20, 29), (20, 30), (20, 31), (20, 32), (20, 33),
        (21, 34), (21, 35), (21, 36), (21, 37), (21, 38),
        (22, 39), (22, 40), (22, 41), (22, 42), (22, 43),
        (23, 44), (23, 45), (23, 46), (23, 47), (23, 48),
        (24, 49), (24, 50), (24, 25), (24, 26), (24, 27),
        (25, 28), (25, 29), (25, 30), (25, 31), (25, 32),
        (26, 33), (26, 34), (26, 35), (26, 36), (26, 37),
        (27, 38), (27, 39), (27, 40), (27, 41), (27, 42),
        (28, 43), (28, 44), (28, 45), (28, 46), (28, 47),
        (29, 48), (29, 49), (29, 50), (30, 31), (30, 32),
        (30, 33), (30, 34), (30, 35), (31, 36), (31, 37),
        (31, 38), (31, 39), (31, 40), (32, 41), (32, 42),
        (32, 43), (32, 44), (32, 45), (33, 46), (33, 47),
        (33, 48), (33, 49), (33, 50)
    ]
    
    for edge in edges:
        g.add_edge(*edge)
    
    
    #40 fails
    # fails = [
    #     (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1,8), (2, 11), (2, 12), (2, 13),
    #     (2, 14), (3, 16), (3, 17), (3, 18), (3, 19), (4, 21), (4, 22), (4, 23), (4, 24),
    #     (5, 26), (5, 27), (5, 28), (6, 31), (6, 32), (6, 33), (7, 36), (7, 37), (7, 38),
    #     (8, 41), (8, 42), (8, 43),
    #     (2, 15), (3, 20), (4, 25), (5, 29), (6, 34), (7, 39), (8, 44), (9, 46), (10, 15), (11, 20)
    # ]
    
    #50 fails
    # fails = [
    #     (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 11), (2, 12), (2, 13),
    #     (2, 14), (3, 16), (3, 17), (3, 18), (3, 19), (4, 21), (4, 22), (4, 23), (4, 24),
    #     (5, 26), (5, 27), (5, 28), (6, 31), (6, 32), (6, 33), (7, 36), (7, 37), (7, 38),
    #     (8, 41), (8, 42), (8, 43), (2, 15), (3, 20), (4, 25), (5, 29), (6, 34), (7, 39), 
    #     (8, 44), (9, 46), (10, 15), (11, 20), (12, 22), (13, 27), (14, 32), (15, 37), 
    #     (16, 42), (17, 47), (18, 23), (19, 24), (20, 30), (21, 34), (22, 39)
    # ]
    
    #60 fails
    fails = [
        (1, 2), (1, 3), (1, 4), (1, 5), (1, 6), (1, 7), (1, 8), (2, 11), (2, 12), (2, 13),
        (2, 14), (3, 16), (3, 17), (3, 18), (3, 19), (4, 21), (4, 22), (4, 23), (4, 24),
        (5, 26), (5, 27), (5, 28), (6, 31), (6, 32), (6, 33), (7, 36), (7, 37), (7, 38),
        (8, 41), (8, 42), (8, 43), (2, 15), (3, 20), (4, 25), (5, 29), (6, 34), (7, 39), 
        (8, 44), (9, 46), (10, 15), (11, 20), (12, 22), (13, 27), (14, 32), (15, 37), 
        (16, 42), (17, 47), (18, 23), (19, 24), (20, 30), (21, 34), (22, 39),
        (23, 45), (24, 26), (25, 27), (26, 29), (27, 31), (28, 33), (29, 34), (30, 35), 
        (31, 39), (32, 40)
    ]
    return g, fails



import random

def create_clustered_failures_graph(n, k, f_num):
    g = nx.random_regular_graph(k,  n)
    fails = []

    # Choose a random node to create clustered failures
    failed_node = random.choice(list(g.nodes))

    # Find neighbors of the chosen node
    neighbors = list(g.neighbors(failed_node))
    
    # Ensure f_num is not larger than the possible number of connections
    possible_nodes = set(g.nodes) - {failed_node}
    if f_num > len(possible_nodes):
        f_num = len(possible_nodes)
    
    if len(neighbors) < f_num:
        # If there are not enough neighbors, choose random nodes
        additional_nodes = random.sample(possible_nodes - set(neighbors), f_num - len(neighbors))
        neighbors.extend(additional_nodes)

    # Choose f_num neighbors (or random nodes) and mark them as faulty edges
    for neighbor in neighbors[:f_num]:
        if g.has_edge(failed_node, neighbor):
            fails.append((failed_node, neighbor))
        else:
            # Add the edge to the graph if it doesn't exist, but mark it as faulty
            g.add_edge(failed_node, neighbor)
            fails.append((failed_node, neighbor))

    return g, fails


# read generated k-regular graphs from file system
def read_graph(i):
    g = nx.read_edgelist('results/' + name + str(seed) + '_graph_' +
                         str(n) + '_' + str(i) + '.txt', nodetype=int).to_directed()
    for (u, v) in g.edges():
        g[u][v]['arb'] = -1
    g.graph['seed'] = 0
    g.graph['k'] = k
    g.graph['root'] = 0
    fails = []
    f = open('results/' + name + str(seed) +
             '_graph_' + str(n) + '_' + str(i) + '.txt', 'r')
    for line in f:
        s = line.replace('\n', '').split(' ')
        fails.append((int(s[0]), int(s[1])))
    f.close()
    g.graph['fails'] = fails
    return g

# generate random ring of clique graphs with n nodes and connectivity k1-1
# in cliques and k2 between neighboring cliques
def create_ring_of_cliques(l,k1, k2, seed):
    #print('l', l, 'k1', k1, 'k2', k2)
    if k2 >= k1*k1:
        print('k2 must be at most k1*k1 for create_ring_of_cliques')
        sys.exit()
    n = l*(k1)
    m = l*(k1*(k1-1)/2+k2)
    random.seed(seed)
    g = nx.Graph()
    g.add_nodes_from(range(n))
    for i in range(l):
        ## wire inside each clique
        for u in range(i*k1, (i+1)*k1):
            for v in range(u,(i+1)*k1):
                g.add_edge(u,v)
        ## wire between cliques
        if i>0:
            for j in range(k2):
                u = random.choice(range(i*k1, (i+1)*k1))
                v = random.choice(range((i-1)*k1, (i)*k1))
                while (u,v) in g.edges():
                    u = random.choice(range(i*k1, (i+1)*k1))
                    v = random.choice(range((i-1)*k1, (i)*k1))
                g.add_edge(u,v)
        else:
            for j in range(k2):
                u = random.choice(range(0, k1))
                v = random.choice(range((l-1)*k1, (l)*k1))
                while (u,v) in g.edges():
                    u = random.choice(range(0,k1))
                    v = random.choice(range((l-1)*k1, (l)*k1))
                g.add_edge(u,v)
    # n selfloops to be removed
    g.remove_edges_from(nx.selfloop_edges(g))
    if (len(g.edges())!= m):
        print("Bug in ring of clique generation")
        sys.exit()
    g = g.to_directed()
    prepare_graph(g,2*k2,seed)
    return g

# set attributes for algorithms
def prepare_graph(g,k,seed):
    g.graph['seed'] = seed
    g.graph['k'] = k
    g.graph['root'] = 0
    g2 = g.to_undirected()
    g2.remove_edges_from(nx.selfloop_edges(g2))
    fails = list(g2.edges())
    random.seed(seed)

    good = False
    count = 0
    while not good:
        count += 1
        random.shuffle(fails)
        G = g.to_undirected()
        n = len(g.nodes())
        G.remove_edges_from(fails[:n])
        Gcc = sorted(nx.connected_components(G), key=len, reverse=True)
        if 0 in Gcc[0]:
            good = True
        elif count > 10:
            g.graph['root'] = list(Gcc[0])[0]
            good = True
        #else:
        #    print('reshuffle in prepare graph', count)
    g.graph['fails'] = fails

# return j th zoo graph if it can be trimmed into a graph of connectivity at least 4 and at
# least 10 nodes
def read_zoo(j, min_connectivity):
    zoo_list = list(glob.glob("./benchmark_graphs/*.graphml"))
    if len(zoo_list) == 0:
        print("Add Internet Topology Zoo graphs (*.graphml files) to directory benchmark_graphs")
        print("(download them from http://www.topology-zoo.org/dataset.html")
        sys.exit()
    if len(zoo_list) <= j:
        return None
    #print(" J : " , j ) 
    #print(" Zoo_List  : " , zoo_list )
    #print("Zoo_List[j] : " , zoo_list[j]) 
    g1 = nx.Graph(nx.read_graphml(zoo_list[j]))
    g2 = nx.convert_node_labels_to_integers(g1)
    g2.remove_edges_from(nx.selfloop_edges(g2))
    g2 = g2.to_directed()
    # print(nx.edge_connectivity(g2),',', len(g2.nodes))
    n_before = len(g2.nodes)

    
    #print("Nodes befor Trimming : " , len(g2.nodes))
    degree = min(3, min_connectivity)
    degree = min(1, min_connectivity)
    #print("Start Trimming")
    while nx.edge_connectivity(g2) < min_connectivity:
        #print("G2 Degree before Trim : " , nx.edge_connectivity(g2) , " Nodes : " , len(g2.nodes))
        g2 = trim2(g2, degree)
        if len(g2.nodes) == 0:
            # print(zoo_list[j],"too sparse",len(g1.nodes), len(g1.edges))
            return None
        #print("G2 Degree after Trim : " , nx.edge_connectivity(g2) , " Nodes : " , len(g2.nodes))
        degree += 1
    #if len(g2.nodes) <= 10:
    #    return None
    #print(" ")
    g = g2.to_directed()
    print(j, zoo_list[j],'n_before=', n_before, 'n_after=', len(g.nodes), 'm_after=', len(g.edges), 'connectivity=', nx.edge_connectivity(g2), 'degree=', degree)

    #print(" ")
    #print(" ")
    for (u, v) in g.edges():
        g[u][v]['arb'] = -1
    prepare_graph(g, nx.edge_connectivity(g), seed)
    g.graph['undirected failures'] = False
    g.graph['pos'] = nx.spring_layout(g)
    return g


# read AS graphs and trims them to be of connectivity at least conn
def generate_trimmed_AS(conn):
    import fnss
    files = glob.glob('./benchmark_graphs/*.cch')
    if len(files) == 0:
        print("Add Rocketfuel Graphs (*.cch) to directory benchmark_graphs")
        sys.exit()
    for x in files:
        if 'r0' in x or 'r1' in x or 'pop' in x or 'README' in x:
            continue
        g = nx.Graph()
        print(x)
        g.add_edges_from(fnss.parse_rocketfuel_isp_map(x).edges())
        # print("Trimming to connectivity %i" %conn)
        gt = trim_merge(g, conn)
        # relabelling
        gtL = nx.convert_node_labels_to_integers(gt)
        if (gtL.number_of_nodes() == 0):
            print("AS-Graph %s contains no node after trimming" % x)
            continue
        if (gtL.number_of_nodes() >= 1000):
            print("AS-Graph %s contains too many nodes" % x, gtL.number_of_nodes())
            continue
        if (nx.edge_connectivity(gtL) < conn):
            print("AS-Graph %s is not connected enough for connectivity %i" % (x, conn))
            continue
        else:
            print("AS-Graph %s with %i nodes is good" % (x, gtL.number_of_nodes()))
            nx.write_edgelist(gtL, x[:-4].replace("graphs/", "graphs/AS") + "-" + str(conn) + ".csv")
