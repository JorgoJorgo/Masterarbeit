import sys
import time
import math
import random
import traceback
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from typing import List, Any, Union
from itertools import combinations, permutations
from matplotlib.patches import Patch, Polygon
from matplotlib.collections import PatchCollection
from networkx import PlanarEmbedding
from arborescences import *
from faces import find_faces_pre, draw_graph_with_highlighted_edge
from trees import all_edps, connect_leaf_to_destination, multiple_trees, multiple_trees_parallel, rank_tree, remove_redundant_paths, remove_single_node_trees
import os


#################################################### BASE ALGORITHMS FOR TREE-BUILDING ################################################

#ein großer baum für alle edps, längster EDP wird als erstes erweitert, für normale baumstrukturen mit tree routing
def multiple_trees_with_checkpoint(source, destination, graph, all_edps):
    print(f"[MultipleTreesWithCheckpoint] Start for {source} -> {destination}")
    removed_edges = 0
    trees = [] 
    debug = should_debug(source, destination)

    for i in range(len(all_edps)):
        current_edp = all_edps[i]
        tree = nx.DiGraph()
        tree.add_node(source)
        for j in range(1, len(current_edp) - 1):
            tree.add_node(current_edp[j])
            tree.add_edge(current_edp[j - 1], current_edp[j])
        trees.append(tree)
    
    assert len(trees) == len(all_edps), 'Not every EDP got a tree!'

    for i in range(len(all_edps)):
        tree = trees[i]
        pathToExtend = all_edps[i]
        nodes = pathToExtend[:-1]

        for j in range(len(pathToExtend) - 1):
            it = 0
            while it < len(nodes):
                neighbors = list(nx.neighbors(graph, nodes[it]))
                for k in range(len(neighbors)):
                    if neighbors[k] != nodes[it] and neighbors[k] != destination:
                            is_in_other_tree = False
                            for tree_to_check in trees: 
                                    if (tree_to_check.has_edge(nodes[it],neighbors[k]) or tree_to_check.has_edge(neighbors[k], nodes[it])):
                                        
                                        is_in_other_tree = True
                                        break
                                    
                                    #endif
                            #endfor
                            
                            if not ( is_in_other_tree or (tree.has_node(neighbors[k])) ):
                                nodes.append(neighbors[k]) 
                                tree.add_node(neighbors[k])
                                tree.add_edge(nodes[it],neighbors[k])
                            #endif
                        
                it += 1

        changed = True
        old_edges = len(tree.edges)
        while changed:
            old_tree = tree.copy()
            remove_redundant_paths(source, destination, tree, graph)
            changed = len(tree.nodes) != len(old_tree.nodes)
        new_edges = len(tree.edges)
        removed_edges += (old_edges - new_edges)
        
        if len(tree.nodes) > 1:
            rank_tree(tree, source, all_edps[i])
            connect_leaf_to_destination(tree, source, destination)
            tree.add_edge(all_edps[i][-2], destination)
            tree.nodes[destination]["rank"] = -1
        
        if len(tree.nodes) == 1 and len(all_edps[i]) == 2:
            tree.add_edge(source, destination)
            tree.nodes[destination]["rank"] = -1
        
        if debug:
            print(f"[Tree {i} Final] Completed Tree with {len(tree.nodes)} nodes and {len(tree.edges)} edges.")
            draw_graph(tree, source, destination, graph)
    
    removed_edges_multtrees.append(removed_edges)
    return trees

#ein großer baum für alle edps, längster EDP wird als erstes erweitert, NUR FÜR STRUKTUREN BEI DENEN FACE ROUTING BENUTZT WIRD
def multiple_trees_with_checkpoint_for_faces(source, destination, graph, all_edps):

    """Erstellt mehrere Bäume mit einem Checkpoint unter Berücksichtigung der Faces."""

    debug = False
    tree = nx.Graph()
    tree.add_node(source)
    
    debug = should_debug(source, destination)

    print(f"Start building Trees for {source} -> {destination}")

    #jeder tree besteht anfänglich aus dem edp
    for i in range(len(all_edps)):

        current_edp = all_edps[i]
        
        tree.nodes[source]['pos'] = graph.nodes[source]['pos']

        for j in range(1, len(current_edp)):

            tree.add_node(current_edp[j])

            tree.add_edge(current_edp[j - 1], current_edp[j])

            tree.nodes[current_edp[j]]['pos'] = graph.nodes[current_edp[j]]['pos']

        if len(current_edp) == 2:
            tree.add_edge(source, destination)
            tree.nodes[destination]['pos'] = graph.nodes[destination]['pos']

    faces  = find_faces_pre(tree,source,destination)
    smallest_face = None
    smallest_face_size = 1000000
    for face in faces:
        if len(face) < smallest_face_size:
            smallest_face = face
            smallest_face_size = len(face)


    #draw_graph_with_highlighted_edge(tree, source, destination, ())
    #jeden edp durchgehen und ihn erweitern
    for i in range(len(all_edps)):

        #pathToExtend beinhaltet immer alle Knoten des Trees, damit sie geprüft werden, welche Nachbarn noch hinzugefügt werden können

        pathToExtend = all_edps[i]
        
        nodes = pathToExtend

        #print(f"[Tree {i} Start] Building Tree with {len(nodes)} nodes.")

        #print(f"Treenodes Nodes: {tree.nodes}")
        previous_edge = None

        for j in range(len(pathToExtend)):

            it = 0
            
            while it < len(nodes):
                #print("Nodes: ", nodes)
                #print("Nodes[it]", nodes[it])

                neighbors = list(nx.neighbors(graph, nodes[it]))

                for k in range(len(neighbors)):
                    if ((nodes[it], neighbors[k])in tree.edges) or ((neighbors[k], nodes[it]) in tree.edges):
                         continue
                    #endif
                    if (nodes[it], neighbors[k]) == previous_edge or (neighbors[k],nodes[it]) == previous_edge:
                        continue

                    previous_edge = (nodes[it], neighbors[k])
                    fake_tree = tree.copy()
                    
                    neighbor_accepted = False 
                    
                    #füge Kante (nodes[it], neighbors[k]) dem fake_tree hinzu, um zu prüfen ob die face bedingung kaputt geht
                    
                    fake_tree.add_node(neighbors[k])
                    
                    fake_tree.nodes[neighbors[k]]['pos'] = graph.nodes[neighbors[k]]['pos']
                    
                    fake_tree.add_edge(nodes[it], neighbors[k])
                    
                    fake_tree.nodes[nodes[it]]['pos'] = graph.nodes[nodes[it]]['pos']
                    #draw_graph_with_highlighted_edge(fake_tree, source, destination, (nodes[it], neighbors[k]))
                    
                    extra_edge = False
                    
                    if graph.has_edge(destination, neighbors[k]) or graph.has_edge(neighbors[k], destination):
                        extra_edge = True
                        fake_tree.add_edge(neighbors[k], destination)
                        fake_tree.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    #endif
                    
                    for node in fake_tree.nodes:
                        fake_tree.nodes[node]['pos'] = graph.nodes[node]['pos']

                    faces  = find_faces_pre(fake_tree,source,destination)
                    #print("Faces: ", faces)
                    if(debug):
                        draw_graph_with_highlighted_edge(fake_tree, source, destination, (nodes[it], neighbors[k]))
                    if len(faces) > 0:
                        for face in faces:
                            if source in face and destination in face and smallest_face in faces:
                                neighbor_accepted = True
                                break
                            #endif
                        #endfor
                    #endif

                    #wenn der Nachbar akzeptiert werden kann, dann füge die Kante dem OG Tree hinzu
                    if neighbor_accepted:

                        
                        tree.add_node(neighbors[k])
                        tree.add_edge(nodes[it], neighbors[k])
                        if extra_edge:
                            tree.add_edge(neighbors[k], destination)
                            nodes.append(destination)
                        #endif
                        
                        for node in tree.nodes:
                            tree.nodes[node]['pos'] = graph.nodes[node]['pos']
                        nodes.append(neighbors[k])
                    #endif
                #endfor   

                it += 1
            #endwhile
            #draw_graph_with_highlighted_edge(tree, source, destination, ())

            
            #pruning the tree by removing redundant paths, which do not lead to the destination
            
            changed = True
            #print("Starting Pruning")
            while changed:
                old_edges = len(tree.edges)
                nodes_to_check = list(tree.nodes)  # Create a list of nodes to iterate over

                for node in nodes_to_check:
                    #print("checking node: ", node)
                    
                    if node != source and node != destination and tree.degree(node) == 1:
                        accept_removal = False
                        fake_tree = tree.copy()
                        neighbor = list(nx.neighbors(fake_tree, node))[0]
                        #print("Neighbor: ", neighbor)
                        
                        if (node, neighbor) in fake_tree.edges:
                            #draw_graph_with_highlighted_edge(fake_tree, source, destination, (node, neighbor))
                            fake_tree.remove_edge(node, neighbor)
                        else:
                            #draw_graph_with_highlighted_edge(fake_tree, source, destination, (neighbor, node))
                            fake_tree.remove_edge(neighbor, node)
                        
                        fake_tree.remove_node(node)
                        
                        for node2 in fake_tree.nodes:
                            fake_tree.nodes[node2]['pos'] = graph.nodes[node2]['pos']
                        
                        faces = find_faces_pre(fake_tree, source, destination)
                        
                        for face in faces:
                            if source in face and destination in face:
                                accept_removal = True
                                break
                        
                        if accept_removal:
                            if (node, neighbor) in tree.edges:
                                tree.remove_edge(node, neighbor)
                            else:
                                tree.remove_edge(neighbor, node)
                            tree.remove_node(node)
                
                new_edges = len(tree.edges)
                changed = old_edges != new_edges
        #endfor
    #draw_graph_with_highlighted_edge(tree, source, destination, ())
    return tree

def multiple_trees_parallel_cp(source, destination, graph, all_edps):
    """Erstellt mehrere Bäume mit einem Checkpoint unter Berücksichtigung der Faces parallel."""
    
    tree = nx.Graph()
    tree.add_node(source)

    tree.nodes[source]['pos'] = graph.nodes[source]['pos']

    
    edge_lists = []
    edgelist_counter = 0
    for edp in all_edps:
        edge_lists.append([])
        for j in range(len(edp)):
            tree.add_node(edp[j])
            
            tree.nodes[edp[j]]['pos'] = graph.nodes[edp[j]]['pos']
            if j > 0:
                tree.add_edge(edp[j - 1], edp[j])
                edge_lists[edgelist_counter].append((edp[j - 1], edp[j]))

        edgelist_counter += 1
    paths_to_extend = all_edps.copy()
    it_list = [0] * len(paths_to_extend)
    changed = True

    while changed:
        changed = False
        print("Starting a new iteration of expansion")
        
        for i in range(len(paths_to_extend)):
            checked_all = False
            checked_number = 0
            print(f"Expanding path {i}")
            
            while not checked_all:
                if it_list[i] >= len(paths_to_extend[i]):
                    checked_all = True
                    continue
                
                node = paths_to_extend[i][it_list[i]]
                checked_number += 1
                neighbors = list(nx.neighbors(graph, node))

                added_edge = False
                
                for neighbor in neighbors:
                    #print(f"Checking neighbor {neighbor}")
                    
                    if tree.has_edge(node, neighbor) or neighbor in tree.nodes:
                        continue
                    
                    fake_tree = tree.copy()
                    fake_tree.add_node(neighbor)
                    fake_tree.add_edge(node, neighbor)
                    
                    fake_tree.nodes[neighbor]['pos'] = graph.nodes[neighbor]['pos']

                    
                    #print(f"Calling find_faces_pre for node {node} and neighbor {neighbor}")
                    faces = find_faces_pre(fake_tree, source, destination)
                    smallest_face = min(faces, key=len, default=None)
                                        
                    if smallest_face and source in smallest_face and destination in smallest_face:
                        tree.add_node(neighbor)
                        tree.add_edge(node, neighbor)
                        edge_lists[i].append((node, neighbor))

                        #check if every edge in the edge list is in the tree
                        for edge in edge_lists[i]:
                            assert edge in tree.edges, f"Edge list {edge_lists[i]} not in tree edges!"  
                            assert edge in fake_tree.edges, f"Edge list {edge_lists[i]} not in fake_tree edges!"  

                        # Position des neuen Knotens hinzufügen
                        tree.nodes[neighbor]['pos'] = graph.nodes[neighbor]['pos']
                        
                        
                        paths_to_extend[i].append(neighbor)
                        changed = True
                        added_edge = True
                        #print(f"Added edge ({node}, {neighbor})")
                        
                    if added_edge:
                        break
                
                #if no edge was added then we need to increase the iterator an check the next node of the current path
                if not added_edge:
                    it_list[i] += 1
                
                #if an edge was added we can switch to the next path and look for the next node to add
                else:
                    break
                
                #if we checked all nodes of the current path we can switch to the next path
                if checked_number >= len(paths_to_extend[i]):
                    print(f"Checked all nodes of {paths_to_extend[i]}")
                    checked_all = True

                #reset the iterator if we checked to the end of the path, in order to start at the beginning of the path again
                if it_list[i] >= len(paths_to_extend[i]):
                    it_list[i] = 0
    

    #now the tree needs to bre pruned
    changed = True
    while changed:
        changed = False
        old_edges = len(tree.edges)
        nodes_to_check = list(tree.nodes)
        
        for node in nodes_to_check:
            if node != source and node != destination and tree.degree(node) == 1:
                accept_removal = False
                fake_tree = tree.copy()
                neighbor = list(nx.neighbors(fake_tree, node))[0]
                
                if (node, neighbor) in fake_tree.edges:
                    fake_tree.remove_edge(node, neighbor)
                else:
                    fake_tree.remove_edge(neighbor, node)
                
                fake_tree.remove_node(node)
                
                for node2 in fake_tree.nodes:
                    fake_tree.nodes[node2]['pos'] = graph.nodes[node2]['pos']
                
                faces = find_faces_pre(fake_tree, source, destination)
                
                for face in faces:
                    if source in face and destination in face:
                        accept_removal = True
                        break
                
                if accept_removal:
                    if (node, neighbor) in tree.edges:
                        tree.remove_edge(node, neighbor)
                    else:
                        tree.remove_edge(neighbor, node)
                    tree.remove_node(node)
                    changed = True

    #draw_graph_with_highlighted_edge2(tree, source, destination, edge_lists, ())

    return tree

#ein großer baum, durch die erweiterung des mitgegebenen edps "longest_edp", für normale baumstrukturen mit tree routing
def one_tree_with_checkpoint(source, destination, graph, longest_edp):
    #print("[one_tree_with_checkpoint] source:",source)
    #print("[one_tree_with_checkpoint] destination:",destination)
    #print("[one_tree_with_checkpoint] longest_edp:",longest_edp)
    tree = nx.DiGraph()
    assert source == longest_edp[0] , 'Source is not start of edp'
    tree.add_node(source) # source = longest_edp[0]
    tree.nodes[source]['pos'] = graph.nodes[source]['pos']

    # We need to include the EDP itself here
    for i in range(1,len(longest_edp)-1): # -2 since we don't want to insert the destination
        tree.add_node(longest_edp[i])
        tree.nodes[longest_edp[i]]['pos'] = graph.nodes[longest_edp[i]]['pos']
        tree.add_edge(longest_edp[i-1],longest_edp[i])

    pathToExtend = longest_edp
    
    for i in range(0,len(pathToExtend)-1): # i max 7
        
        nodes = pathToExtend[:len(pathToExtend) -2]
        it = 0 # to get the neighbors of the neighbors
        while it < len(nodes):

            neighbors = list(nx.neighbors(graph, nodes[it]))
            for j in neighbors:
                if (not tree.has_node(j)) and (j!= destination): #not part of tree already and not the destiantion
                    nodes.append(j)
                    tree.add_node(j) #add neighbors[j] to tree
                    tree.nodes[j]['pos'] = graph.nodes[j]['pos']
                    tree.add_edge(nodes[it], j) # add edge to new node
                #end if
            #end for
            it = it+1
        #end while
    #end for
    
    changed = True 
    

    while changed == True: #keep trying to shorten until no more can be shortened 
        
        old_tree = tree.copy()
        remove_redundant_paths(source, destination, tree, graph)
        changed = tree.order() != old_tree.order() # order returns the number of nodes in the graph.

    
    
        rank_tree(tree , source,longest_edp)
        connect_leaf_to_destination(tree, source, destination)
        tree.add_edge(longest_edp[len(longest_edp)-2],destination)
    
        #add 'rank' property to the added destinaton, -1 for highest priority in routing
        tree.nodes[destination]["rank"] = -1

        for node in tree.nodes:
            tree.nodes[node]['pos'] = graph.nodes[node]['pos']
            #node['pos'] = list(graph.nodes)[node]['pos']

        return tree    
        
    #end if

#ein großer baum, durch die erweiterung des mitgegebenen edps "longest_edp", NUR FÜR STRUKTUREN BEI DENEN FACE ROUTING BENUTZT WIRD
def one_tree_with_checkpoint_for_faces(source, destination, graph, longest_edp):
    """Erstellt einen Baum aus dem EDP für das Face Routing"""
    #print(f"[one_tree_with_checkpoint_for_faces] Initializing tree from source {source} to destination {destination}")
    
    tree = nx.Graph()
    assert source == longest_edp[0], 'Source is not start of edp'
    tree.add_node(source)
    tree.nodes[source]['pos'] = graph.nodes[source]['pos']
    #print(f"[one_tree_with_checkpoint_for_faces] Added source node {source} to tree")
    
    # Adding the EDP (Edge Disjoint Path) to the tree
    for i in range(1, len(longest_edp)):
        tree.add_node(longest_edp[i])
        tree.nodes[longest_edp[i]]['pos'] = graph.nodes[longest_edp[i]]['pos']
        tree.add_edge(longest_edp[i-1], longest_edp[i])
        #print(f"[one_tree_with_checkpoint_for_faces] Added edge ({longest_edp[i-1]}, {longest_edp[i]}) to tree")

    pathToExtend = longest_edp
    #draw_graph_with_highlighted_edge(tree, source, destination, ())

    for i in range(0, len(pathToExtend) - 1):
        #print(f"[one_tree_with_checkpoint_for_faces] Extending path, iteration {i}")
        
        nodes = pathToExtend[:len(pathToExtend) - 2]
        it = 0
        
        while it < len(nodes):
            neighbors = list(nx.neighbors(graph, nodes[it]))
            #print(f"[one_tree_with_checkpoint_for_faces] Node {nodes[it]} neighbors: {neighbors}")
            
            for j in neighbors:
                #print(f"[one_tree_with_checkpoint_for_faces] Checking neighbor {j}")
                
                fake_tree = tree.copy()
                fake_tree.add_node(j)
                fake_tree.nodes[j]['pos'] = graph.nodes[j]['pos']
                fake_tree.add_edge(nodes[it], j)
                faces = find_faces_pre(fake_tree, source, destination)
                
                face_accepted = False
                for face in faces:
                    if source in face and destination in face:
                        face_accepted = True
                        break
                
                extra_edge = False
                face_accepted_extra_edge = False
                if graph.has_edge(destination, j) or graph.has_edge(j, destination):
                    fake_tree.add_edge(j, destination)
                    fake_tree.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    faces = find_faces_pre(fake_tree, source, destination)
                    
                    for face in faces:
                        if source in face and destination in face:
                            face_accepted_extra_edge = True
                            break

                if face_accepted:
                    tree.add_node(j)
                    tree.nodes[j]['pos'] = graph.nodes[j]['pos']
                    tree.add_edge(nodes[it], j)
                    #print(f"[one_tree_with_checkpoint_for_faces] Added edge ({nodes[it]}, {j}) to tree")
                
                if face_accepted_extra_edge:
                    tree.add_edge(j, destination)
                    tree.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    #print(f"[one_tree_with_checkpoint_for_faces] Added edge ({j}, {destination}) to tree")
                
            it += 1

    #draw_graph_with_highlighted_edge(tree, source, destination, ())

    changed = True
    while changed:
        #print(f"[one_tree_with_checkpoint_for_faces] Attempting to shorten the tree")
        
        old_edges = len(tree.edges)
        nodes_to_check = list(tree.nodes)
        
        for node in nodes_to_check:
            if node != source and node != destination and tree.degree(node) == 1:
                accept_removal = False
                fake_tree = tree.copy()
                neighbor = list(nx.neighbors(fake_tree, node))[0]
                #print(f"[one_tree_with_checkpoint_for_faces] Checking leaf node {node} with neighbor {neighbor}")
                
                if (node, neighbor) in fake_tree.edges:
                    fake_tree.remove_edge(node, neighbor)
                else:
                    fake_tree.remove_edge(neighbor, node)
                
                fake_tree.remove_node(node)
                
                for node2 in fake_tree.nodes:
                    fake_tree.nodes[node2]['pos'] = graph.nodes[node2]['pos']
                
                faces = find_faces_pre(fake_tree, source, destination)
                
                for face in faces:
                    if source in face and destination in face:
                        accept_removal = True
                        break
                
                if accept_removal:
                    if (node, neighbor) in tree.edges:
                        tree.remove_edge(node, neighbor)
                    else:
                        tree.remove_edge(neighbor, node)
                    tree.remove_node(node)
                    #print(f"[one_tree_with_checkpoint_for_faces] Removed leaf node {node}")

        new_edges = len(tree.edges)
        changed = old_edges != new_edges
        #if not changed:
            #print("[one_tree_with_checkpoint_for_faces] No more edges to shorten")

    #print("[one_tree_with_checkpoint_for_faces] Final tree constructed")
    return tree
  
    
#################################################### MULTIPLETREES WITH MIDDLE CHECKPOINT ################################################

##########################################################################################################################################

def multiple_trees_with_middle_checkpoint_parallel_pre(graph):
    paths = {}
    #draw_tree_with_highlights(graph)
    print("[MultipleTreesOneCheckpointParallelPre] Start Precomputation")
    for source in graph.nodes:
       
        for destination in graph.nodes:
            
            if source != destination:
                
                edps = all_edps(source, destination, graph) #Bildung der EDPs
                
                edps.sort(key=len, reverse=True) #Sortierung der EDPs
                
                longest_edp = edps[len(edps)-1]

                #special case if the s,d pair is connected and this is the only edp
                if(len(longest_edp) == 2):

                    if source not in paths:
                        paths[source] = {}
                    #print("Special case for : ", source, "-", destination)


                    tree_from_s = nx.DiGraph()
                    tree_from_s.add_node(source)
                    tree_from_s.add_node(destination)
                    tree_from_d = nx.DiGraph()
                    tree_from_d.add_node(source)
                    tree_from_d.add_node(destination)

                    
                    tree_from_s.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_d.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_s.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_d.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_s.add_edge(source,destination)
                    tree_from_d.add_edge(source,destination)

                    paths[source][destination] = {
                                                'cp': destination,
                                                'trees_cp_to_s': tree_from_s, 
                                                'edps_cp_to_s': [[source,destination]],
                                                'trees_cp_to_d':tree_from_d, 
                                                'edps_cp_to_d': [[source,destination]],
                                                'edps_s_to_d':[[source,destination]]
                                            }
                    continue
                
                cp = longest_edp[ int(len(longest_edp)/2)]

                edps_cp_to_s = all_edps(cp, source, graph)
                edps_cp_to_d = all_edps(cp, destination, graph)
                edps_cp_to_s.sort(key=len)
                edps_cp_to_d.sort(key=len)
                
                trees_cp_to_s = multiple_trees_parallel_cp(cp,source,graph,edps_cp_to_s)
                
           
                #then build multiple trees cp->d
                
                trees_cp_to_d = multiple_trees_parallel(cp,destination,graph,edps_cp_to_d)
                
                for tree in trees_cp_to_d:
                    for node in tree:
                        tree.nodes[node]['pos'] = graph.nodes[node]['pos']

                #EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben, führen nicht zum Ziel und müssen gelöscht werden
                trees_cp_to_d = remove_single_node_trees(trees_cp_to_d)
                                                        
                if source in paths:
                    paths[source][destination] = { 
                                                    'cp': cp,
                                                'edps_cp_to_s': edps_cp_to_s,
                                                'trees_cp_to_s': trees_cp_to_s,
                                                'trees_cp_to_d': trees_cp_to_d, 
                                                'edps_cp_to_d': edps_cp_to_d,
                                                'edps_s_to_d': edps
                                                }
                else:
                    paths[source] = {}
                    paths[source][destination] = {
                                                'cp': cp,
                                                'trees_cp_to_s': trees_cp_to_s, 
                                                'edps_cp_to_s': edps_cp_to_s,
                                                'trees_cp_to_d': trees_cp_to_d, 
                                                'edps_cp_to_d': edps_cp_to_d,
                                                'edps_s_to_d': edps
                    }
                
                # Überprüfen, ob der Graph planar ist
                # print("Nodes of combined tree:", combined_tree.nodes)
                # print("Edges of combined tree:", combined_tree.edges)
                # is_planar, embedding = nx.check_planarity(combined_tree)

                # if is_planar:
                #     planar_embedding = create_planar_embedding_new(combined_tree, graph)
                    
                    
                    
                # else:
                #     print("Der Graph ist nicht planar!")

                # #jetzt müssen die Bäume cp->s als planare einbettung hinzugefügt werden
                # #damit die Faces gefunden werden können


               
                    
                # faces = find_faces(planar_embedding)

                # one_correct_face = False

                # for face in faces:
                #     if source in face and destination in face:
                #         one_correct_face = True
                #         break

                # assert one_correct_face, f"Source {source} and Destination {destination} are not in the same face."

                #if( len(trees_cp_to_s.nodes)>14): 
                #    print_cut_structure(highlighted_nodes=[source,cp],structure=trees_cp_to_s,source=source,destination=cp,save_plot=True,filename=f"graphen/MultipleTreesParallel_{source}_{cp}.png")
            
    
    return paths


#################################################### MULTIPLETREES WITH MIDDLE CHECKPOINT ################################################

##########################################################################################################################################
removed_edges_multtrees = []

def multiple_trees_with_middle_checkpoint_pre(graph):
    paths = {}
    #draw_tree_with_highlights(graph)
    print("[MultipleTreesOneCheckpointPre] Start Precomputation")
    print("All Combinations: ", (len(graph.nodes) * len(graph.nodes)) - len(graph.nodes))
    combinations = 0
    for source in graph.nodes:
       
        for destination in graph.nodes:
            
            
            if source != destination:
                combinations += 1
                #print("Current Combination: ", combinations, " of ", (len(graph.nodes) * len(graph.nodes)) - len(graph.nodes))
                edps = all_edps(source, destination, graph) #Bildung der EDPs
                
                edps.sort(key=len, reverse=True) #Sortierung der EDPs
                
                longest_edp = edps[len(edps)-1]

                #special case if the s,d pair is connected and this is the only edp
                if(len(longest_edp) == 2):

                    if source not in paths:
                        paths[source] = {}
                    #print("Special case for : ", source, "-", destination)


                    tree_from_s = nx.DiGraph()
                    tree_from_s.add_node(source)
                    tree_from_s.add_node(destination)
                    tree_from_d = nx.DiGraph()
                    tree_from_d.add_node(source)
                    tree_from_d.add_node(destination)

                    
                    tree_from_s.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_d.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_s.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_d.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_s.add_edge(source,destination)
                    tree_from_d.add_edge(source,destination)

                    paths[source][destination] = {
                                                'cp': destination,
                                                'trees_cp_to_s': tree_from_s, 
                                                'edps_cp_to_s': [[source,destination]],
                                                'trees_cp_to_d':tree_from_d, 
                                                'edps_cp_to_d': [[source,destination]],
                                                'edps_s_to_d':[[source,destination]]
                                            }
                    continue
                
                cp = longest_edp[ int(len(longest_edp)/2)]

                edps_cp_to_s = all_edps(cp, source, graph)
                edps_cp_to_d = all_edps(cp, destination, graph)
                edps_cp_to_s.sort(key=len)
                edps_cp_to_d.sort(key=len)
                
                trees_cp_to_s = multiple_trees_with_checkpoint_for_faces(cp,source,graph,edps_cp_to_s)

                #EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben, führen nicht zum Ziel und müssen gelöscht werden
                #trees_cp_to_s = remove_single_node_trees(trees_cp_to_s)
                
                #then build multiple trees cp->d
                
                trees_cp_to_d = multiple_trees_with_checkpoint(cp,destination,graph,edps_cp_to_d)
                
                for tree in trees_cp_to_d:
                    for node in tree:
                        tree.nodes[node]['pos'] = graph.nodes[node]['pos']

                #EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben, führen nicht zum Ziel und müssen gelöscht werden
                trees_cp_to_d = remove_single_node_trees(trees_cp_to_d)
                                                        
                if source in paths:
                    paths[source][destination] = { 
                                                'cp': cp,
                                                'edps_cp_to_s': edps_cp_to_s,
                                                'trees_cp_to_s': trees_cp_to_s,
                                                'trees_cp_to_d': trees_cp_to_d, 
                                                'edps_cp_to_d': edps_cp_to_d,
                                                'edps_s_to_d': edps
                                                }
                else:
                    paths[source] = {}
                    paths[source][destination] = {
                                                'cp': cp,
                                                'trees_cp_to_s': trees_cp_to_s, 
                                                'edps_cp_to_s': edps_cp_to_s,
                                                'trees_cp_to_d': trees_cp_to_d, 
                                                'edps_cp_to_d': edps_cp_to_d,
                                                'edps_s_to_d': edps
                    }
                #if( len(trees_cp_to_s.nodes)>14): 
                #    print_cut_structure(highlighted_nodes=[source,cp],structure=trees_cp_to_s,source=source,destination=cp,save_plot=True,filename=f"graphen/MultipleTreesWithMiddle_{source}_{cp}.png")
    return paths


#################################################### MULTIPLETREES INVERS WITH MIDDLE CHECKPOINT ################################################

##########################################################################################################################################
removed_edges_multtrees = []

def multiple_trees_invers_with_middle_checkpoint_pre(graph):
    paths = {}
    #draw_tree_with_highlights(graph)
    print("[MultipleTreesOneCheckpointPre] Start Precomputation")
    print("All Combinations: ", (len(graph.nodes) * len(graph.nodes)) - len(graph.nodes))
    combinations = 0
    for source in graph.nodes:
       
        for destination in graph.nodes:
            
            
            if source != destination:
                combinations += 1
                #print("Current Combination: ", combinations, " of ", (len(graph.nodes) * len(graph.nodes)) - len(graph.nodes))
                edps = all_edps(source, destination, graph) #Bildung der EDPs
                
                edps.sort(key=len, reverse=False) #Sortierung der EDPs
                
                longest_edp = edps[len(edps)-1]

                #special case if the s,d pair is connected and this is the only edp
                if(len(longest_edp) == 2):

                    if source not in paths:
                        paths[source] = {}
                    #print("Special case for : ", source, "-", destination)


                    tree_from_s = nx.DiGraph()
                    tree_from_s.add_node(source)
                    tree_from_s.add_node(destination)
                    tree_from_d = nx.DiGraph()
                    tree_from_d.add_node(source)
                    tree_from_d.add_node(destination)

                    
                    tree_from_s.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_d.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_s.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_d.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_s.add_edge(source,destination)
                    tree_from_d.add_edge(source,destination)

                    paths[source][destination] = {
                                                'cp': destination,
                                                'trees_cp_to_s': tree_from_s, 
                                                'edps_cp_to_s': [[source,destination]],
                                                'trees_cp_to_d':tree_from_d, 
                                                'edps_cp_to_d': [[source,destination]],
                                                'edps_s_to_d':[[source,destination]]
                                            }
                    continue
                
                cp = longest_edp[ int(len(longest_edp)/2)]

                edps_cp_to_s = all_edps(cp, source, graph)
                edps_cp_to_d = all_edps(cp, destination, graph)
                edps_cp_to_s.sort(key=len)
                edps_cp_to_d.sort(key=len)
                
                trees_cp_to_s = multiple_trees_with_checkpoint_for_faces(cp,source,graph,edps_cp_to_s)

                #EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben, führen nicht zum Ziel und müssen gelöscht werden
                #trees_cp_to_s = remove_single_node_trees(trees_cp_to_s)
                
                #then build multiple trees cp->d
                
                trees_cp_to_d = multiple_trees_with_checkpoint(cp,destination,graph,edps_cp_to_d)
                
                for tree in trees_cp_to_d:
                    for node in tree:
                        tree.nodes[node]['pos'] = graph.nodes[node]['pos']

                #EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben, führen nicht zum Ziel und müssen gelöscht werden
                trees_cp_to_d = remove_single_node_trees(trees_cp_to_d)
                                                        
                if source in paths:
                    paths[source][destination] = { 
                                                'cp': cp,
                                                'edps_cp_to_s': edps_cp_to_s,
                                                'trees_cp_to_s': trees_cp_to_s,
                                                'trees_cp_to_d': trees_cp_to_d, 
                                                'edps_cp_to_d': edps_cp_to_d,
                                                'edps_s_to_d': edps
                                                }
                else:
                    paths[source] = {}
                    paths[source][destination] = {
                                                'cp': cp,
                                                'trees_cp_to_s': trees_cp_to_s, 
                                                'edps_cp_to_s': edps_cp_to_s,
                                                'trees_cp_to_d': trees_cp_to_d, 
                                                'edps_cp_to_d': edps_cp_to_d,
                                                'edps_s_to_d': edps
                    }
                #if( len(trees_cp_to_s.nodes)>14): 
                #    print_cut_structure(highlighted_nodes=[source,cp],structure=trees_cp_to_s,source=source,destination=cp,save_plot=True,filename=f"graphen/MultipleTreesWithMiddle_{source}_{cp}.png")
    return paths


#################################################### MULTIPLETREES INVERS WITH DEGREE CHECKPOINT ################################################

##########################################################################################################################################
removed_edges_multtrees = []

def multiple_trees_invers_with_degree_checkpoint_pre(graph):
    paths = {}
    #draw_tree_with_highlights(graph)
    print("[MultipleTreesOneCheckpointPre] Start Precomputation")
    print("All Combinations: ", (len(graph.nodes) * len(graph.nodes)) - len(graph.nodes))
    combinations = 0
    for source in graph.nodes:
       
        for destination in graph.nodes:
            
            
            if source != destination:
                combinations += 1
                #print("Current Combination: ", combinations, " of ", (len(graph.nodes) * len(graph.nodes)) - len(graph.nodes))
                edps = all_edps(source, destination, graph) #Bildung der EDPs
                
                edps.sort(key=len, reverse=False) #Sortierung der EDPs
                
                longest_edp = edps[len(edps)-1]

                #special case if the s,d pair is connected and this is the only edp
                if(len(longest_edp) == 2):

                    if source not in paths:
                        paths[source] = {}
                    #print("Special case for : ", source, "-", destination)


                    tree_from_s = nx.DiGraph()
                    tree_from_s.add_node(source)
                    tree_from_s.add_node(destination)
                    tree_from_d = nx.DiGraph()
                    tree_from_d.add_node(source)
                    tree_from_d.add_node(destination)

                    
                    tree_from_s.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_d.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_s.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_d.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_s.add_edge(source,destination)
                    tree_from_d.add_edge(source,destination)

                    paths[source][destination] = {
                                                'cp': destination,
                                                'trees_cp_to_s': tree_from_s, 
                                                'edps_cp_to_s': [[source,destination]],
                                                'trees_cp_to_d':tree_from_d, 
                                                'edps_cp_to_d': [[source,destination]],
                                                'edps_s_to_d':[[source,destination]]
                                            }
                    continue
                
                # Calculate Degree Centrality for nodes in the graph
                degree_centrality = nx.degree_centrality(graph)
                
                # Filter out source and destination from the longest EDP
                filtered_edp = [node for node in longest_edp if node != source and node != destination]
                
                # Handle the case where no valid cp is available after filtering
                if not filtered_edp:
                    tree_from_s = nx.DiGraph()
                    tree_from_s.add_node(source)
                    tree_from_s.add_node(destination)
                    tree_from_d = nx.DiGraph()
                    tree_from_d.add_node(source)
                    tree_from_d.add_node(destination)

                    
                    tree_from_s.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_d.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_s.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_d.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_s.add_edge(source,destination)
                    tree_from_d.add_edge(source,destination)

                    paths[source][destination] = {
                                                'cp': destination,
                                                'trees_cp_to_s': tree_from_s, 
                                                'edps_cp_to_s': [[source,destination]],
                                                'trees_cp_to_d':tree_from_d, 
                                                'edps_cp_to_d': [[source,destination]],
                                                'edps_s_to_d':[[source,destination]]
                                            }
                    continue
                
                # Select the node with the highest Degree Centrality in the filtered EDP as the checkpoint
                cp = max(filtered_edp, key=lambda node: degree_centrality[node])

                edps_cp_to_s = all_edps(cp, source, graph)
                edps_cp_to_d = all_edps(cp, destination, graph)
                edps_cp_to_s.sort(key=len)
                edps_cp_to_d.sort(key=len)
                
                trees_cp_to_s = multiple_trees_with_checkpoint_for_faces(cp,source,graph,edps_cp_to_s)

                #EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben, führen nicht zum Ziel und müssen gelöscht werden
                #trees_cp_to_s = remove_single_node_trees(trees_cp_to_s)
                
                #then build multiple trees cp->d
                
                trees_cp_to_d = multiple_trees_with_checkpoint(cp,destination,graph,edps_cp_to_d)
                
                for tree in trees_cp_to_d:
                    for node in tree:
                        tree.nodes[node]['pos'] = graph.nodes[node]['pos']

                #EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben, führen nicht zum Ziel und müssen gelöscht werden
                trees_cp_to_d = remove_single_node_trees(trees_cp_to_d)
                                                        
                if source in paths:
                    paths[source][destination] = { 
                                                'cp': cp,
                                                'edps_cp_to_s': edps_cp_to_s,
                                                'trees_cp_to_s': trees_cp_to_s,
                                                'trees_cp_to_d': trees_cp_to_d, 
                                                'edps_cp_to_d': edps_cp_to_d,
                                                'edps_s_to_d': edps
                                                }
                else:
                    paths[source] = {}
                    paths[source][destination] = {
                                                'cp': cp,
                                                'trees_cp_to_s': trees_cp_to_s, 
                                                'edps_cp_to_s': edps_cp_to_s,
                                                'trees_cp_to_d': trees_cp_to_d, 
                                                'edps_cp_to_d': edps_cp_to_d,
                                                'edps_s_to_d': edps
                    }
                #if( len(trees_cp_to_s.nodes)>14): 
                #    print_cut_structure(highlighted_nodes=[source,cp],structure=trees_cp_to_s,source=source,destination=cp,save_plot=True,filename=f"graphen/MultipleTreesWithMiddle_{source}_{cp}.png")
    return paths


#################################################### MULTIPLETREES INVERS WITH CLOSENESS CHECKPOINT ################################################

##########################################################################################################################################
removed_edges_multtrees = []

def multiple_trees_invers_with_closeness_checkpoint_pre(graph):
    paths = {}
    #draw_tree_with_highlights(graph)
    print("[MultipleTreesOneCheckpointPre] Start Precomputation")
    print("All Combinations: ", (len(graph.nodes) * len(graph.nodes)) - len(graph.nodes))
    combinations = 0
    for source in graph.nodes:
       
        for destination in graph.nodes:
            
            
            if source != destination:
                combinations += 1
                #print("Current Combination: ", combinations, " of ", (len(graph.nodes) * len(graph.nodes)) - len(graph.nodes))
                edps = all_edps(source, destination, graph) #Bildung der EDPs
                
                edps.sort(key=len, reverse=False) #Sortierung der EDPs
                
                longest_edp = edps[len(edps)-1]

                #special case if the s,d pair is connected and this is the only edp
                if(len(longest_edp) == 2):

                    if source not in paths:
                        paths[source] = {}
                    #print("Special case for : ", source, "-", destination)


                    tree_from_s = nx.DiGraph()
                    tree_from_s.add_node(source)
                    tree_from_s.add_node(destination)
                    tree_from_d = nx.DiGraph()
                    tree_from_d.add_node(source)
                    tree_from_d.add_node(destination)

                    
                    tree_from_s.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_d.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_s.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_d.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_s.add_edge(source,destination)
                    tree_from_d.add_edge(source,destination)

                    paths[source][destination] = {
                                                'cp': destination,
                                                'trees_cp_to_s': tree_from_s, 
                                                'edps_cp_to_s': [[source,destination]],
                                                'trees_cp_to_d':tree_from_d, 
                                                'edps_cp_to_d': [[source,destination]],
                                                'edps_s_to_d':[[source,destination]]
                                            }
                    continue
                
                # Calculate Degree Centrality for nodes in the graph
                closeness_centrality = nx.closeness_centrality(graph)
                
                # Filter out source and destination from the longest EDP
                filtered_edp = [node for node in longest_edp if node != source and node != destination]
                
                # Handle the case where no valid cp is available after filtering
                if not filtered_edp:
                    tree_from_s = nx.DiGraph()
                    tree_from_s.add_node(source)
                    tree_from_s.add_node(destination)
                    tree_from_d = nx.DiGraph()
                    tree_from_d.add_node(source)
                    tree_from_d.add_node(destination)

                    
                    tree_from_s.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_d.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_s.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_d.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_s.add_edge(source,destination)
                    tree_from_d.add_edge(source,destination)

                    paths[source][destination] = {
                                                'cp': destination,
                                                'trees_cp_to_s': tree_from_s, 
                                                'edps_cp_to_s': [[source,destination]],
                                                'trees_cp_to_d':tree_from_d, 
                                                'edps_cp_to_d': [[source,destination]],
                                                'edps_s_to_d':[[source,destination]]
                                            }
                    continue
                
                # Select the node with the highest Degree Centrality in the filtered EDP as the checkpoint
                cp = max(filtered_edp, key=lambda node: closeness_centrality[node])

                edps_cp_to_s = all_edps(cp, source, graph)
                edps_cp_to_d = all_edps(cp, destination, graph)
                edps_cp_to_s.sort(key=len)
                edps_cp_to_d.sort(key=len)
                
                trees_cp_to_s = multiple_trees_with_checkpoint_for_faces(cp,source,graph,edps_cp_to_s)

                #EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben, führen nicht zum Ziel und müssen gelöscht werden
                #trees_cp_to_s = remove_single_node_trees(trees_cp_to_s)
                
                #then build multiple trees cp->d
                
                trees_cp_to_d = multiple_trees_with_checkpoint(cp,destination,graph,edps_cp_to_d)
                
                for tree in trees_cp_to_d:
                    for node in tree:
                        tree.nodes[node]['pos'] = graph.nodes[node]['pos']

                #EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben, führen nicht zum Ziel und müssen gelöscht werden
                trees_cp_to_d = remove_single_node_trees(trees_cp_to_d)
                                                        
                if source in paths:
                    paths[source][destination] = { 
                                                'cp': cp,
                                                'edps_cp_to_s': edps_cp_to_s,
                                                'trees_cp_to_s': trees_cp_to_s,
                                                'trees_cp_to_d': trees_cp_to_d, 
                                                'edps_cp_to_d': edps_cp_to_d,
                                                'edps_s_to_d': edps
                                                }
                else:
                    paths[source] = {}
                    paths[source][destination] = {
                                                'cp': cp,
                                                'trees_cp_to_s': trees_cp_to_s, 
                                                'edps_cp_to_s': edps_cp_to_s,
                                                'trees_cp_to_d': trees_cp_to_d, 
                                                'edps_cp_to_d': edps_cp_to_d,
                                                'edps_s_to_d': edps
                    }
                #if( len(trees_cp_to_s.nodes)>14): 
                #    print_cut_structure(highlighted_nodes=[source,cp],structure=trees_cp_to_s,source=source,destination=cp,save_plot=True,filename=f"graphen/MultipleTreesWithMiddle_{source}_{cp}.png")
    return paths

#################################################### MULTIPLETREES INVERS WITH BETWEENNESS CHECKPOINT ################################################

##########################################################################################################################################
removed_edges_multtrees = []

def multiple_trees_invers_with_betweenness_checkpoint_pre(graph):
    paths = {}
    #draw_tree_with_highlights(graph)
    print("[MultipleTreesOneCheckpointPre] Start Precomputation")
    print("All Combinations: ", (len(graph.nodes) * len(graph.nodes)) - len(graph.nodes))
    combinations = 0
    for source in graph.nodes:
       
        for destination in graph.nodes:
            
            
            if source != destination:
                combinations += 1
                #print("Current Combination: ", combinations, " of ", (len(graph.nodes) * len(graph.nodes)) - len(graph.nodes))
                edps = all_edps(source, destination, graph) #Bildung der EDPs
                
                edps.sort(key=len, reverse=False) #Sortierung der EDPs
                
                longest_edp = edps[len(edps)-1]

                #special case if the s,d pair is connected and this is the only edp
                if(len(longest_edp) == 2):

                    if source not in paths:
                        paths[source] = {}
                    #print("Special case for : ", source, "-", destination)


                    tree_from_s = nx.DiGraph()
                    tree_from_s.add_node(source)
                    tree_from_s.add_node(destination)
                    tree_from_d = nx.DiGraph()
                    tree_from_d.add_node(source)
                    tree_from_d.add_node(destination)

                    
                    tree_from_s.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_d.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_s.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_d.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_s.add_edge(source,destination)
                    tree_from_d.add_edge(source,destination)

                    paths[source][destination] = {
                                                'cp': destination,
                                                'trees_cp_to_s': tree_from_s, 
                                                'edps_cp_to_s': [[source,destination]],
                                                'trees_cp_to_d':tree_from_d, 
                                                'edps_cp_to_d': [[source,destination]],
                                                'edps_s_to_d':[[source,destination]]
                                            }
                    continue
                
                # Calculate Degree Centrality for nodes in the graph
                betweenness_centrality = nx.betweenness_centrality(graph)
                
                # Filter out source and destination from the longest EDP
                filtered_edp = [node for node in longest_edp if node != source and node != destination]
                
                # Handle the case where no valid cp is available after filtering
                if not filtered_edp:
                    tree_from_s = nx.DiGraph()
                    tree_from_s.add_node(source)
                    tree_from_s.add_node(destination)
                    tree_from_d = nx.DiGraph()
                    tree_from_d.add_node(source)
                    tree_from_d.add_node(destination)

                    
                    tree_from_s.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_d.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_s.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_d.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_s.add_edge(source,destination)
                    tree_from_d.add_edge(source,destination)

                    paths[source][destination] = {
                                                'cp': destination,
                                                'trees_cp_to_s': tree_from_s, 
                                                'edps_cp_to_s': [[source,destination]],
                                                'trees_cp_to_d':tree_from_d, 
                                                'edps_cp_to_d': [[source,destination]],
                                                'edps_s_to_d':[[source,destination]]
                                            }
                    continue
                
                # Select the node with the highest Degree Centrality in the filtered EDP as the checkpoint
                cp = max(filtered_edp, key=lambda node: betweenness_centrality[node])

                edps_cp_to_s = all_edps(cp, source, graph)
                edps_cp_to_d = all_edps(cp, destination, graph)
                edps_cp_to_s.sort(key=len)
                edps_cp_to_d.sort(key=len)
                
                trees_cp_to_s = multiple_trees_with_checkpoint_for_faces(cp,source,graph,edps_cp_to_s)

                #EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben, führen nicht zum Ziel und müssen gelöscht werden
                #trees_cp_to_s = remove_single_node_trees(trees_cp_to_s)
                
                #then build multiple trees cp->d
                
                trees_cp_to_d = multiple_trees_with_checkpoint(cp,destination,graph,edps_cp_to_d)
                
                for tree in trees_cp_to_d:
                    for node in tree:
                        tree.nodes[node]['pos'] = graph.nodes[node]['pos']

                #EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben, führen nicht zum Ziel und müssen gelöscht werden
                trees_cp_to_d = remove_single_node_trees(trees_cp_to_d)
                                                        
                if source in paths:
                    paths[source][destination] = { 
                                                'cp': cp,
                                                'edps_cp_to_s': edps_cp_to_s,
                                                'trees_cp_to_s': trees_cp_to_s,
                                                'trees_cp_to_d': trees_cp_to_d, 
                                                'edps_cp_to_d': edps_cp_to_d,
                                                'edps_s_to_d': edps
                                                }
                else:
                    paths[source] = {}
                    paths[source][destination] = {
                                                'cp': cp,
                                                'trees_cp_to_s': trees_cp_to_s, 
                                                'edps_cp_to_s': edps_cp_to_s,
                                                'trees_cp_to_d': trees_cp_to_d, 
                                                'edps_cp_to_d': edps_cp_to_d,
                                                'edps_s_to_d': edps
                    }
                #if( len(trees_cp_to_s.nodes)>14): 
                #    print_cut_structure(highlighted_nodes=[source,cp],structure=trees_cp_to_s,source=source,destination=cp,save_plot=True,filename=f"graphen/MultipleTreesWithMiddle_{source}_{cp}.png")
    return paths


################################################## MULTIPLETREES WITH DEGREE CHECKPOINT ################################################

##########################################################################################################################################

def multiple_trees_with_degree_checkpoint_pre(graph):
    paths = {}
    
    #draw_tree_with_highlights(graph)
    print("[MultipleTreesOneCheckpointPre] Start Precomputation")
    for source in graph.nodes:
       
        for destination in graph.nodes:
            
            if source != destination:
                
                edps = all_edps(source, destination, graph) #Bildung der EDPs
                
                edps.sort(key=len, reverse=True) #Sortierung der EDPs
                
                longest_edp = edps[len(edps)-1]

                #special case if the s,d pair is connected and this is the only edp
                if(len(longest_edp) == 2):

                    if source not in paths:
                        paths[source] = {}
                    #print("Special case for : ", source, "-", destination)


                    tree_from_s = nx.DiGraph()
                    tree_from_s.add_node(source)
                    tree_from_s.add_node(destination)
                    tree_from_d = nx.DiGraph()
                    tree_from_d.add_node(source)
                    tree_from_d.add_node(destination)

                    
                    tree_from_s.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_d.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_s.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_d.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_s.add_edge(source,destination)
                    tree_from_d.add_edge(source,destination)

                    paths[source][destination] = {
                                                'cp': destination,
                                                'trees_cp_to_s': tree_from_s, 
                                                'edps_cp_to_s': [[source,destination]],
                                                'trees_cp_to_d':tree_from_d, 
                                                'edps_cp_to_d': [[source,destination]],
                                                'edps_s_to_d':[[source,destination]]
                                            }
                    continue

                # Calculate Degree Centrality for nodes in the graph
                degree_centrality = nx.degree_centrality(graph)
                
                # Filter out source and destination from the longest EDP
                filtered_edp = [node for node in longest_edp if node != source and node != destination]
                
                # Handle the case where no valid cp is available after filtering
                if not filtered_edp:
                    tree_from_s = nx.DiGraph()
                    tree_from_s.add_node(source)
                    tree_from_s.add_node(destination)
                    tree_from_d = nx.DiGraph()
                    tree_from_d.add_node(source)
                    tree_from_d.add_node(destination)

                    
                    tree_from_s.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_d.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_s.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_d.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_s.add_edge(source,destination)
                    tree_from_d.add_edge(source,destination)

                    paths[source][destination] = {
                                                'cp': destination,
                                                'trees_cp_to_s': tree_from_s, 
                                                'edps_cp_to_s': [[source,destination]],
                                                'trees_cp_to_d':tree_from_d, 
                                                'edps_cp_to_d': [[source,destination]],
                                                'edps_s_to_d':[[source,destination]]
                                            }
                    continue
                
                # Select the node with the highest Degree Centrality in the filtered EDP as the checkpoint
                cp = max(filtered_edp, key=lambda node: degree_centrality[node])

                edps_cp_to_s = all_edps(cp, source, graph)
                edps_cp_to_d = all_edps(cp, destination, graph)
                edps_cp_to_s.sort(key=len)
                edps_cp_to_d.sort(key=len)
                
                trees_cp_to_s = multiple_trees_with_checkpoint_for_faces(cp,source,graph,edps_cp_to_s)
        
                
                #then build multiple trees cp->d
                
                trees_cp_to_d = multiple_trees_with_checkpoint(cp,destination,graph,edps_cp_to_d)
                
                for tree in trees_cp_to_d:
                    for node in tree:
                        tree.nodes[node]['pos'] = graph.nodes[node]['pos']

                #EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben, führen nicht zum Ziel und müssen gelöscht werden
                trees_cp_to_d = remove_single_node_trees(trees_cp_to_d)
                                                        
                if source in paths:
                    paths[source][destination] = { 
                                                    'cp': cp,
                                                'edps_cp_to_s': edps_cp_to_s,
                                                'trees_cp_to_s': trees_cp_to_s,
                                                'trees_cp_to_d': trees_cp_to_d, 
                                                'edps_cp_to_d': edps_cp_to_d,
                                                'edps_s_to_d': edps
                                                }
                else:
                    paths[source] = {}
                    paths[source][destination] = {
                                                'cp': cp,
                                                'trees_cp_to_s': trees_cp_to_s, 
                                                'edps_cp_to_s': edps_cp_to_s,
                                                'trees_cp_to_d': trees_cp_to_d, 
                                                'edps_cp_to_d': edps_cp_to_d,
                                                'edps_s_to_d': edps
                    }
                #if( len(trees_cp_to_s.nodes)>14): 
                #    print_cut_structure(highlighted_nodes=[source,cp],structure=trees_cp_to_s,source=source,destination=cp,save_plot=True,filename=f"graphen/MultipleTreesWithDegree_{source}_{cp}.png")
    return paths


#################################################### MULTIPLETREES WITH BETWEENNESS CHECKPOINT ###########################################

##########################################################################################################################################

def multiple_trees_with_betweenness_checkpoint_pre(graph):
    paths = {}
    #draw_tree_with_highlights(graph)
    print("[MultipleTreesOneCheckpointPre] Start Precomputation")
    for source in graph.nodes:
       
        for destination in graph.nodes:
            
            if source != destination:
                
                edps = all_edps(source, destination, graph) #Bildung der EDPs
                
                edps.sort(key=len, reverse=True) #Sortierung der EDPs
                
                longest_edp = edps[len(edps)-1]

                #special case if the s,d pair is connected and this is the only edp
                if(len(longest_edp) == 2):

                    if source not in paths:
                        paths[source] = {}
                    #print("Special case for : ", source, "-", destination)


                    tree_from_s = nx.DiGraph()
                    tree_from_s.add_node(source)
                    tree_from_s.add_node(destination)
                    tree_from_d = nx.DiGraph()
                    tree_from_d.add_node(source)
                    tree_from_d.add_node(destination)

                    
                    tree_from_s.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_d.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_s.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_d.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_s.add_edge(source,destination)
                    tree_from_d.add_edge(source,destination)

                    paths[source][destination] = {
                                                'cp': destination,
                                                'trees_cp_to_s': tree_from_s, 
                                                'edps_cp_to_s': [[source,destination]],
                                                'trees_cp_to_d':tree_from_d, 
                                                'edps_cp_to_d': [[source,destination]],
                                                'edps_s_to_d':[[source,destination]]
                                            }
                    continue

                # Calculate Degree Centrality for nodes in the graph
                betweenness_centrality = nx.betweenness_centrality(graph, normalized=True)
                
                # Filter out source and destination from the longest EDP
                filtered_edp = [node for node in longest_edp if node != source and node != destination]
                
                # Handle the case where no valid cp is available after filtering
                if not filtered_edp:
                    tree_from_s = nx.DiGraph()
                    tree_from_s.add_node(source)
                    tree_from_s.add_node(destination)
                    tree_from_d = nx.DiGraph()
                    tree_from_d.add_node(source)
                    tree_from_d.add_node(destination)

                    
                    tree_from_s.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_d.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_s.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_d.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_s.add_edge(source,destination)
                    tree_from_d.add_edge(source,destination)

                    paths[source][destination] = {
                                                'cp': destination,
                                                'trees_cp_to_s': tree_from_s, 
                                                'edps_cp_to_s': [[source,destination]],
                                                'trees_cp_to_d':tree_from_d, 
                                                'edps_cp_to_d': [[source,destination]],
                                                'edps_s_to_d':[[source,destination]]
                                            }
                    continue
                
                # Select the node with the highest Degree Centrality in the filtered EDP as the checkpoint
                cp = max(filtered_edp, key=lambda node: betweenness_centrality[node])

                edps_cp_to_s = all_edps(cp, source, graph)
                edps_cp_to_d = all_edps(cp, destination, graph)
                edps_cp_to_s.sort(key=len)
                edps_cp_to_d.sort(key=len)
                
                trees_cp_to_s = multiple_trees_with_checkpoint_for_faces(cp,source,graph,edps_cp_to_s)
                
                #then build multiple trees cp->d
                
                trees_cp_to_d = multiple_trees_with_checkpoint(cp,destination,graph,edps_cp_to_d)
                
                for tree in trees_cp_to_d:
                    for node in tree:
                        tree.nodes[node]['pos'] = graph.nodes[node]['pos']

                #EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben, führen nicht zum Ziel und müssen gelöscht werden
                trees_cp_to_d = remove_single_node_trees(trees_cp_to_d)
                                                        
                if source in paths:
                    paths[source][destination] = { 
                                                    'cp': cp,
                                                'edps_cp_to_s': edps_cp_to_s,
                                                'trees_cp_to_s': trees_cp_to_s,
                                                'trees_cp_to_d': trees_cp_to_d, 
                                                'edps_cp_to_d': edps_cp_to_d,
                                                'edps_s_to_d': edps
                                                }
                else:
                    paths[source] = {}
                    paths[source][destination] = {
                                                'cp': cp,
                                                'trees_cp_to_s': trees_cp_to_s, 
                                                'edps_cp_to_s': edps_cp_to_s,
                                                'trees_cp_to_d': trees_cp_to_d, 
                                                'edps_cp_to_d': edps_cp_to_d,
                                                'edps_s_to_d': edps
                    }
                #if( len(trees_cp_to_s.nodes)>14): 
                #    print_cut_structure(highlighted_nodes=[source,cp],structure=trees_cp_to_s,source=source,destination=cp,save_plot=True,filename=f"graphen/MultipleTreesWithBetween_{source}_{cp}.png")
    return paths


#################################################### MULTIPLETREES WITH CLOSENESS CHECKPOINT ###########################################

##########################################################################################################################################

def multiple_trees_with_closeness_checkpoint_pre(graph):
    paths = {}
    count = 1
    all_graph_edge_number = len(graph.edges)
    all_tree_edge_number = 0
    #draw_tree_with_highlights(graph)
    print("[MultipleTreesOneCheckpointPre] Start Precomputation")
    biggest_source = 0
    biggest_destination = 0
    biggest_structure = nx.DiGraph()
    for source in graph.nodes:
       
        for destination in graph.nodes:
            
            if source != destination:
                
                edps = all_edps(source, destination, graph) #Bildung der EDPs
                
                edps.sort(key=len, reverse=True) #Sortierung der EDPs
                
                longest_edp = edps[len(edps)-1]

                #special case if the s,d pair is connected and this is the only edp
                if(len(longest_edp) == 2):

                    if source not in paths:
                        paths[source] = {}
                    #print("Special case for : ", source, "-", destination)


                    tree_from_s = nx.DiGraph()
                    tree_from_s.add_node(source)
                    tree_from_s.add_node(destination)
                    tree_from_d = nx.DiGraph()
                    tree_from_d.add_node(source)
                    tree_from_d.add_node(destination)

                    
                    tree_from_s.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_d.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_s.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_d.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_s.add_edge(source,destination)
                    tree_from_d.add_edge(source,destination)

                    paths[source][destination] = {
                                                'cp': destination,
                                                'trees_cp_to_s': tree_from_s, 
                                                'edps_cp_to_s': [[source,destination]],
                                                'trees_cp_to_d':tree_from_d, 
                                                'edps_cp_to_d': [[source,destination]],
                                                'edps_s_to_d':[[source,destination]]
                                            }
                    continue

                # Calculate Degree Centrality for nodes in the graph
                closeness_centrality = nx.closeness_centrality(graph)
                
                # Filter out source and destination from the longest EDP
                filtered_edp = [node for node in longest_edp if node != source and node != destination]
                
                # Handle the case where no valid cp is available after filtering
                if not filtered_edp:
                    tree_from_s = nx.DiGraph()
                    tree_from_s.add_node(source)
                    tree_from_s.add_node(destination)
                    tree_from_d = nx.DiGraph()
                    tree_from_d.add_node(source)
                    tree_from_d.add_node(destination)

                    
                    tree_from_s.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_d.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_s.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_d.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_s.add_edge(source,destination)
                    tree_from_d.add_edge(source,destination)

                    paths[source][destination] = {
                                                'cp': destination,
                                                'trees_cp_to_s': tree_from_s, 
                                                'edps_cp_to_s': [[source,destination]],
                                                'trees_cp_to_d':tree_from_d, 
                                                'edps_cp_to_d': [[source,destination]],
                                                'edps_s_to_d':[[source,destination]]
                                            }
                    continue
                
                # Select the node with the highest Degree Centrality in the filtered EDP as the checkpoint
                cp = max(filtered_edp, key=lambda node: closeness_centrality[node])

                edps_cp_to_s = all_edps(cp, source, graph)
                edps_cp_to_d = all_edps(cp, destination, graph)
                edps_cp_to_s.sort(key=len)
                edps_cp_to_d.sort(key=len)
                
                trees_cp_to_s = multiple_trees_with_checkpoint(cp,source,graph,edps_cp_to_s)
                #EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben, führen nicht zum Ziel und müssen gelöscht werden
                trees_cp_to_s = remove_single_node_trees(trees_cp_to_s)
           
                # da kein tree-routing s->cp stattfindet, sondern face-routing, werden alle bäume (cp->s) zu einem großen baum eingefügt, auf dem  man face-routing machen kann
                # Combine all trees into one large undirected tree
                combined_tree = nx.Graph()
                for tree in trees_cp_to_s:
                    tree = tree.to_undirected()  # Ensure the tree is undirected
                    for node in tree.nodes:
                            combined_tree.add_node(node)  # Add node without position
                    combined_tree.add_edges_from(tree.edges())  # Add edges

                for node in combined_tree.nodes:
                    combined_tree.nodes[node]['pos'] = graph.nodes[node]['pos']
         
                #beinhaltet einen nx.Graph planar, alle Trees in einem Graphen mit Koordinaten
                trees_cp_to_s = combined_tree
                
                #then build multiple trees cp->d
                
                trees_cp_to_d = multiple_trees_with_checkpoint(cp,destination,graph,edps_cp_to_d)
                
                for tree in trees_cp_to_d:
                    for node in tree:
                        tree.nodes[node]['pos'] = graph.nodes[node]['pos']

                #EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben, führen nicht zum Ziel und müssen gelöscht werden
                trees_cp_to_d = remove_single_node_trees(trees_cp_to_d)
                                                        
                if source in paths:
                    paths[source][destination] = { 
                                                    'cp': cp,
                                                'edps_cp_to_s': edps_cp_to_s,
                                                'trees_cp_to_s': trees_cp_to_s,
                                                'trees_cp_to_d': trees_cp_to_d, 
                                                'edps_cp_to_d': edps_cp_to_d,
                                                'edps_s_to_d': edps
                                                }
                else:
                    paths[source] = {}
                    paths[source][destination] = {
                                                'cp': cp,
                                                'trees_cp_to_s': trees_cp_to_s, 
                                                'edps_cp_to_s': edps_cp_to_s,
                                                'trees_cp_to_d': trees_cp_to_d, 
                                                'edps_cp_to_d': edps_cp_to_d,
                                                'edps_s_to_d': edps
                    }

                #if( len(trees_cp_to_s.nodes)>14): 
                #    print_cut_structure(highlighted_nodes=[source,cp],structure=trees_cp_to_s,source=source,destination=cp,save_plot=True,filename=f"graphen/MultipleTreesWithCloseness_{source}_{cp}.png")
    return paths


#################################################### ONETREE WITH MIDDLE CHECKPOINT ######################################################

##########################################################################################################################################

#function to generate the 2 trees for each s->d pair (s->cp & cp->d)
#each tree gets generated by expanding the longest edp of each pair
def one_tree_with_middle_checkpoint_pre(graph):
    debug = False
    paths = {}
    all_combinations = (len(graph.nodes) * len(graph.nodes) )- len(graph.nodes)
    current_combination = 1
    for source in graph.nodes:
        #print("[OTC Random Pre] check")
        for destination in graph.nodes:
            
            if source != destination:
                #print(f"Combination {current_combination} of {all_combinations}")
                current_combination = current_combination + 1
                if source not in paths:
                    paths[source] = {}
                
                #now compute the chosen checkpoint  
                #first get the longest edp s->d    
                edps = all_edps(source, destination, graph)
                
                edps.sort(key=len)
                
                longest_edp = edps[len(edps)-1]
                
                #special case if the s,d pair is connected and this is the only edp
                if(len(longest_edp) == 2):
                    
                    if source not in paths:
                        paths[source] = {}
                    #print("Special case for : ", source, "-", destination)


                    tree_from_s = nx.DiGraph()
                    tree_from_s.add_node(source)
                    tree_from_s.add_node(destination)
                    tree_from_d = nx.DiGraph()
                    tree_from_d.add_node(source)
                    tree_from_d.add_node(destination)

                    
                    tree_from_s.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_d.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_s.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_d.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_s.add_edge(source,destination)
                    tree_from_d.add_edge(source,destination)

                    paths[source][destination] = {
                                                'cp': destination,
                                                'edps_cp_to_s': [[source,destination]],
                                                'tree_cp_to_d':tree_from_d, 
                                                'tree_cp_to_s':tree_from_s,
                                                'edps_cp_to_d': [[source,destination]],
                                                'edps_s_to_d':[[source,destination]]
                                            }
                    continue
                
                #then select the middle node of the longest_edp
                
                cp = longest_edp[ int(len(longest_edp)/2)]
               
                #then get the edps + longest_edps_cp_s and the longest_edps_cp_d
                edps_cp_to_s = all_edps(cp, source, graph)
                edps_cp_to_d = all_edps(cp, destination, graph)
                edps_cp_to_s.sort(key=len)
                edps_cp_to_d.sort(key=len)
                
                #and build trees out of the longest_edps_cp_s and the longest_edps_cp_d
                tree_cp_to_s = one_tree_with_checkpoint_for_faces(cp,source,graph,edps_cp_to_s[len(edps_cp_to_s)-1])

                #draw_tree_with_highlighted_nodes(tree_cp_to_s,[source,cp])

                tree_cp_to_d = one_tree_with_checkpoint(cp,destination,graph,edps_cp_to_d[len(edps_cp_to_d)-1])
                
                #bc the tree cp->s got build reverse direction the edges need to be reversed again
                #data structure to give the needed information for the routing (edps, trees, checkpoint)
                
                paths[source][destination] = {
                                                'cp': cp,
                                                'edps_cp_to_s': edps_cp_to_s,
                                                'tree_cp_to_d': tree_cp_to_d, 
                                                'edps_cp_to_d': edps_cp_to_d,
                                                'edps_s_to_d': edps,
                                                'tree_cp_to_s':tree_cp_to_s
                                            }
                #if( len(tree_cp_to_s.nodes)>14): 
                #    print_cut_structure(highlighted_nodes=[source,cp],structure=tree_cp_to_s,source=source,destination=cp,save_plot=True,filename=f"graphen/OneTreeMiddle_{source}_{cp}.png")
    return paths


########################################################################################################

#################################################### ONETREE WITH DEGREE CHECKPOINT ######################################################

##########################################################################################################################################


def one_tree_with_degree_checkpoint_pre(graph):
    debug = False
    paths = {}
    biggest_structure = nx.DiGraph()
    biggest_destination = None
    biggest_source = None
    for source in graph.nodes:

        for destination in graph.nodes:
            if source != destination:
                if source not in paths:
                    paths[source] = {}
                
                # Compute all EDPs between source and destination
                edps = all_edps(source, destination, graph)
                edps.sort(key=len)
                
                # Get the longest EDP
                longest_edp = edps[len(edps) - 1]
                
                # Special case if the source and destination are directly connected then all structures (edps + trees) are the edge between s-d
                if(len(longest_edp) == 2):
                    
                    if source not in paths:
                        paths[source] = {}


                    tree_from_s = nx.DiGraph()
                    tree_from_s.add_node(source)
                    tree_from_s.add_node(destination)
                    tree_from_d = nx.DiGraph()
                    tree_from_d.add_node(source)
                    tree_from_d.add_node(destination)

                    
                    tree_from_s.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_d.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_s.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_d.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_s.add_edge(source,destination)
                    tree_from_d.add_edge(source,destination)
                    
                    paths[source][destination] = {
                                                'cp': destination,
                                                'edps_cp_to_s': [[source,destination]],
                                                'tree_cp_to_d':tree_from_d, 
                                                'tree_cp_to_s':tree_from_s,
                                                'edps_cp_to_d': [[source,destination]],
                                                'edps_s_to_d':[[source,destination]]
                                            }
                    continue
                
                
                # Calculate Degree Centrality for nodes in the graph
                degree_centrality = nx.degree_centrality(graph)
                
                # Filter out source and destination from the longest EDP
                filtered_edp = [node for node in longest_edp if node != source and node != destination]
                
                # Handle the case where no valid cp is available after filtering
                if not filtered_edp:
                    tree_from_s = nx.DiGraph()
                    tree_from_s.add_node(source)
                    tree_from_s.add_node(destination)
                    tree_from_d = nx.DiGraph()
                    tree_from_d.add_node(source)
                    tree_from_d.add_node(destination)

                    
                    tree_from_s.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_d.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_s.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_d.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_s.add_edge(source,destination)
                    tree_from_d.add_edge(source,destination)

                    paths[source][destination] = {
                        'cp': destination,
                        'edps_cp_to_s': [[source,destination]],
                        'tree_cp_to_d':tree_from_d, 
                        'tree_cp_to_s':tree_from_s,
                        'edps_cp_to_d': [[source,destination]],
                        'edps_s_to_d':[[source,destination]]
                    }
                    continue
                
                # Select the node with the highest Degree Centrality in the filtered EDP as the checkpoint
                cp = max(filtered_edp, key=lambda node: degree_centrality[node])
                #print(f"[OneTreeDegreeCheckpoint] Selected Checkpoint (cp): Node {cp} with Centrality {degree_centrality[cp]:.4f}\n")
                
                # Get EDPs from the checkpoint to the source and destination
                edps_cp_to_s = all_edps(cp, source, graph)
                edps_cp_to_d = all_edps(cp, destination, graph)
                
                edps_cp_to_s.sort(key=len)
                edps_cp_to_d.sort(key=len)
                
                # Build trees and faces
                tree_cp_to_s = one_tree_with_checkpoint_for_faces(
                    cp, source, graph, edps_cp_to_s[-1]
                )
                
                tree_cp_to_d = one_tree_with_checkpoint(
                    cp, destination, graph, edps_cp_to_d[-1]
                )
                
                # Store the result in the paths dictionary
                paths[source][destination] = {
                    'cp': cp,
                    'edps_cp_to_s': edps_cp_to_s,
                    'tree_cp_to_d': tree_cp_to_d, 
                    'edps_cp_to_d': edps_cp_to_d,
                    'edps_s_to_d': edps,
                    'tree_cp_to_s':tree_cp_to_s
                }
                #if( len(tree_cp_to_s.nodes)>14): 
                #    print_cut_structure(highlighted_nodes=[source,cp],structure=tree_cp_to_s,source=source,destination=cp,save_plot=True,filename=f"graphen/OneTreeDegree_{source}_{cp}.png")
    return paths


######################################################################################################################################################

#################################################### ONETREE WITH BETWEENNESS CHECKPOINT ######################################################

##########################################################################################################################################


def one_tree_with_betweenness_checkpoint_pre(graph):
    paths = {}
    for source in graph.nodes:
        
        for destination in graph.nodes:
            if source != destination:
                if source not in paths:
                    paths[source] = {}
                
                # Compute all EDPs between source and destination
                edps = all_edps(source, destination, graph)
                edps.sort(key=len)
                
                # Get the longest EDP
                longest_edp = edps[len(edps) - 1]
                
                # Special case if the source and destination are directly connected
                if(len(longest_edp) == 2):
                    
                    if source not in paths:
                        paths[source] = {}


                    tree_from_s = nx.DiGraph()
                    tree_from_s.add_node(source)
                    tree_from_s.add_node(destination)
                    tree_from_d = nx.DiGraph()
                    tree_from_d.add_node(source)
                    tree_from_d.add_node(destination)

                    
                    tree_from_s.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_d.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_s.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_d.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_s.add_edge(source,destination)
                    tree_from_d.add_edge(source,destination)
                    
                    paths[source][destination] = {
                                                'cp': destination,
                                                'edps_cp_to_s': [[source,destination]],
                                                'tree_cp_to_d':tree_from_d, 
                                                'tree_cp_to_s':tree_from_s,
                                                'edps_cp_to_d': [[source,destination]],
                                                'edps_s_to_d':[[source,destination]]
                                            }
                    continue
                
                
                # Calculate Betweenness Centrality for nodes in the graph
                betweenness_centrality = nx.betweenness_centrality(graph, normalized=True)
                
                # Filter out source and destination from the longest EDP
                filtered_edp = [node for node in longest_edp if node != source and node != destination]
                
                # Handle the case where no valid cp is available after filtering
                if not filtered_edp:
                    tree_from_s = nx.DiGraph()
                    tree_from_s.add_node(source)
                    tree_from_s.add_node(destination)
                    tree_from_d = nx.DiGraph()
                    tree_from_d.add_node(source)
                    tree_from_d.add_node(destination)

                    
                    tree_from_s.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_d.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_s.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_d.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_s.add_edge(source,destination)
                    tree_from_d.add_edge(source,destination)

                    paths[source][destination] = {
                        'cp': destination,
                        'edps_cp_to_s': [[source,destination]],
                        'tree_cp_to_d':tree_from_d, 
                        'tree_cp_to_s':tree_from_s,
                        'edps_cp_to_d': [[source,destination]],
                        'edps_s_to_d':[[source,destination]]
                    }
                    continue
                
                # Select the node with the highest Betweenness Centrality in the filtered EDP as the checkpoint
                cp = max(filtered_edp, key=lambda node: betweenness_centrality[node])
                
                # Get EDPs from the checkpoint to the source and destination
                edps_cp_to_s = all_edps(cp, source, graph)
                edps_cp_to_d = all_edps(cp, destination, graph)
                edps_cp_to_s.sort(key=len)
                edps_cp_to_d.sort(key=len)
                
                # Build trees and faces
                tree_cp_to_s = one_tree_with_checkpoint_for_faces(
                    cp, source, graph, edps_cp_to_s[-1]
                )
                
                tree_cp_to_d = one_tree_with_checkpoint(
                    cp, destination, graph, edps_cp_to_d[-1]
                )
                
                # Store the result in the paths dictionary
                paths[source][destination] = {
                    'cp': cp,
                    'edps_cp_to_s': edps_cp_to_s,
                    'tree_cp_to_d': tree_cp_to_d, 
                    'edps_cp_to_d': edps_cp_to_d,
                    'edps_s_to_d': edps,
                    'tree_cp_to_s': tree_cp_to_s
                }

                # if( len(tree_cp_to_s.nodes)>14): 
                #     print_cut_structure(highlighted_nodes=[source,cp],structure=tree_cp_to_s,source=source,destination=cp,save_plot=True,filename=f"graphen/OneTreeBetween_{source}_{cp}.png")
    return paths



######################################################################################################################################################

#################################################### ONETREE WITH BETWEENNESS CHECKPOINT ######################################################

##########################################################################################################################################


def one_tree_with_closeness_checkpoint_pre(graph):
    debug = False
    paths = {}

    for source in graph.nodes:

        for destination in graph.nodes:
            if source != destination:
                if source not in paths:
                    paths[source] = {}
                
                # Compute all EDPs between source and destination
                edps = all_edps(source, destination, graph)
                edps.sort(key=len)
                
                # Get the longest EDP
                longest_edp = edps[len(edps) - 1]
                
                # Special case if the source and destination are directly connected
                if(len(longest_edp) == 2):
                    
                    if source not in paths:
                        paths[source] = {}
                    

                    tree_from_s = nx.DiGraph()
                    tree_from_s.add_node(source)
                    tree_from_s.add_node(destination)
                    tree_from_d = nx.DiGraph()
                    tree_from_d.add_node(source)
                    tree_from_d.add_node(destination)

                    
                    tree_from_s.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_d.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_s.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_d.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_s.add_edge(source,destination)
                    tree_from_d.add_edge(source,destination)
                    
                    paths[source][destination] = {
                                                'cp': destination,
                                                'edps_cp_to_s': [[source,destination]],
                                                'tree_cp_to_d':tree_from_d, 
                                                'tree_cp_to_s':tree_from_s,
                                                'edps_cp_to_d': [[source,destination]],
                                                'edps_s_to_d':[[source,destination]]
                                            }
                    continue
                
                
                # Calculate Closeness Centrality for nodes in the graph
                closeness_centrality = nx.closeness_centrality(graph)
                
                # Filter out source and destination from the longest EDP
                filtered_edp = [node for node in longest_edp if node != source and node != destination]
                
                # Handle the case where no valid cp is available after filtering
                if not filtered_edp:
                    tree_from_s = nx.DiGraph()
                    tree_from_s.add_node(source)
                    tree_from_s.add_node(destination)
                    tree_from_d = nx.DiGraph()
                    tree_from_d.add_node(source)
                    tree_from_d.add_node(destination)

                    
                    tree_from_s.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_d.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_s.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_d.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_s.add_edge(source,destination)
                    tree_from_d.add_edge(source,destination)

                    paths[source][destination] = {
                        'cp': destination,
                        'edps_cp_to_s': [[source,destination]],
                        'tree_cp_to_d':tree_from_d, 
                        'tree_cp_to_s':tree_from_s,
                        'edps_cp_to_d': [[source,destination]],
                        'edps_s_to_d':[[source,destination]]
                    }
                    continue
                
                # Select the node with the highest Closeness Centrality in the filtered EDP as the checkpoint
                cp = max(filtered_edp, key=lambda node: closeness_centrality[node])
                
                # Get EDPs from the checkpoint to the source and destination
                edps_cp_to_s = all_edps(cp, source, graph)
                edps_cp_to_d = all_edps(cp, destination, graph)
                
                edps_cp_to_s.sort(key=len)
                edps_cp_to_d.sort(key=len)
                
                # Build trees and faces
                tree_cp_to_s = one_tree_with_checkpoint_for_faces(
                    cp, source, graph, edps_cp_to_s[-1]
                )
                
                tree_cp_to_d = one_tree_with_checkpoint(
                    cp, destination, graph, edps_cp_to_d[-1]
                )
                
                # Store the result in the paths dictionary
                paths[source][destination] = {
                    'cp': cp,
                    'edps_cp_to_s': edps_cp_to_s,
                    'tree_cp_to_d': tree_cp_to_d, 
                    'edps_cp_to_d': edps_cp_to_d,
                    'edps_s_to_d': edps,
                    'tree_cp_to_s':tree_cp_to_s
                }

                #if( len(tree_cp_to_s.nodes)>14): 
                #    print_cut_structure(highlighted_nodes=[source,cp],structure=tree_cp_to_s,source=source,destination=cp,save_plot=True,filename=f"graphen/OneTreeCloseness_{source}_{cp}.png")
    return paths


############################################### ONETREECHECKPOINT WITH SHORTEST EDP ##############################################

def one_tree_with_middle_checkpoint_shortest_edp_pre(graph):
    debug = False
    paths = {}
    
    for source in graph.nodes:
        for destination in graph.nodes:
            if source != destination:
                if source not in paths:
                    paths[source] = {}

                # Compute all EDPs between source and destination
                edps = all_edps(source, destination, graph)
                edps.sort(key=len)

                # Filter EDPs to ensure they are at least 3 nodes long
                valid_edps = [edp for edp in edps if len(edp) >= 3]

                # Handle special case where no valid EDP >= 3 is found
                if not valid_edps:
                    # Handle the case where source and destination are directly connected (len == 2)
                    if len(edps[0]) == 2:
                        all_faces_special = find_faces(graph)
                        

                        tree_from_s = nx.DiGraph()
                        tree_from_s.add_node(source)
                        tree_from_s.add_node(destination)
                        tree_from_d = nx.DiGraph()
                        tree_from_d.add_node(source)
                        tree_from_d.add_node(destination)

                        
                        tree_from_s.nodes[source]['pos'] = graph.nodes[source]['pos']
                        tree_from_d.nodes[source]['pos'] = graph.nodes[source]['pos']
                        tree_from_s.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                        tree_from_d.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                        tree_from_s.add_edge(source,destination)
                        tree_from_d.add_edge(source,destination)

                        paths[source][destination] = {
                            'cp': destination,
                            'edps_cp_to_s': [[source, destination]],
                            'tree_cp_to_d':tree_from_d,
                            'tree_cp_to_s': tree_cp_to_s,
                            'edps_cp_to_d': [[source, destination]],
                            'edps_s_to_d': [[source, destination]]
                        }
                    continue

                # Select the shortest valid EDP (at least 3 nodes long)
                shortest_edp = valid_edps[0]

                # Select the middle node of the shortest EDP as checkpoint (cp)
                cp = shortest_edp[len(shortest_edp) // 2]

                # Compute EDPs from cp to source and cp to destination
                edps_cp_to_s = all_edps(cp, source, graph)
                edps_cp_to_d = all_edps(cp, destination, graph)

                edps_cp_to_s.sort(key=len)
                edps_cp_to_d.sort(key=len)

                # Build trees and faces
                tree_cp_to_s = one_tree_with_checkpoint_for_faces(
                    cp, source, graph, edps_cp_to_s[-1]
                )

                tree_cp_to_d = one_tree_with_checkpoint(
                    cp, destination, graph, edps_cp_to_d[-1]
                )

                # Store the result in the paths dictionary
                paths[source][destination] = {
                    'cp': cp,
                    'edps_cp_to_s': edps_cp_to_s,
                    'tree_cp_to_d': tree_cp_to_d,
                    'edps_cp_to_d': edps_cp_to_d,
                    'edps_s_to_d': edps,
                    'tree_cp_to_s': tree_cp_to_s
                }
                # if( len(tree_cp_to_s.nodes)>14): 
                #     print_cut_structure(highlighted_nodes=[source,cp],structure=tree_cp_to_s,source=source,destination=cp,save_plot=True,filename=f"graphen/OneTreeShortest_{source}_{cp}.png")
    
    return paths

############################################################################################################################

############################################## ONETREE TRIPLE CHECKPOINT ###################################################
def one_tree_triple_checkpooint_pre(graph):

    paths = {}
    for source in graph.nodes:
        for destination in graph.nodes:
            if source != destination:

                if source not in paths:
                    paths[source] = {}

                # Compute all EDPs between source and destination
                edps = all_edps(source, destination, graph)
                edps.sort(key=len)

                # Filter EDPs to ensure they are at least 5 nodes long s-cp1-cp2-cp3-d
                valid_edps = [edp for edp in edps if len(edp) >= 5]

                # Handle special case where no valid EDP >= 5 is found
                if not valid_edps:
                    # Handle the case where source and destination are directly connected (len == 2)
                    

                    tree_from_s = nx.DiGraph()
                    #print("[triple_checkpoint] edps:", edps)
                    for i in range(len(edps[0])):
                        #print("i:", i)
                        tree_from_s.add_node(edps[0][i])
                        if i > 0:
                            # Füge eine Kante zwischen zwei aufeinanderfolgenden Knoten hinzu
                            #print(f"Edge: ({edps[0][i-1]}, {edps[0][i]})")
                            tree_from_s.add_edge(edps[0][i-1], edps[0][i])



                    paths[source][destination] = {
                        'cps': [destination],
                        'edps_s_to_d': edps,
                        'edps_cp1_to_s':edps,
                        'edps_cp1_to_cp2':edps,
                        'edps_cp3_to_cp2':edps,
                        'edps_cp3_to_d': edps,
                        'tree_cp1_to_s':tree_from_s,
                        'tree_cp1_to_cp2':tree_from_s,
                        'tree_cp3_to_cp2':tree_from_s,
                        'tree_cp3_to_d':tree_from_s
                    }
                    #plot_paths_element(paths[source][destination],graph,source,destination)
                    continue

                # Select the longest valid EDP (assuming it's already sorted by length)
                longest_edp = valid_edps[-1]

                # Anzahl der Knoten prüfen
                if len(longest_edp) < 5:
                    raise ValueError("EDP must have at least 5 nodes for meaningful CP distribution.")

                # Divide the EDP into segments for CP selection
                num_segments = min(4, len(longest_edp) - 1)  # Ensure no out-of-bound segments
                segment_length = len(longest_edp) // num_segments

                # Calculate CP indices based on segment midpoints
                cp_indices = []
                for i in range(1, num_segments):
                    cp_idx = i * segment_length + segment_length // 2  # Midpoint of each segment
                    cp_idx = min(cp_idx, len(longest_edp) - 2)  # Ensure not too close to destination
                    cp_indices.append(cp_idx)

                # Select CPs based on calculated indices
                cps = [longest_edp[idx] for idx in cp_indices]

                # Validate CPs
                if len(set(cps)) < len(cps):
                    raise ValueError("Control points overlap. Adjust segmentation logic.")

                cp1 = cps[0]
                cp2 = cps[1]
                cp3 = cps[2]
                # Extract sub-paths
                
                edps_cp1_to_s = all_edps(cp1,source,graph).sort(key=len)
                
                #Special Case if Nodes are directly connected
                if edps_cp1_to_s == None:
                    edps_cp1_to_s = [[cp1,source]]
                
                edps_cp1_to_cp2 = all_edps(cp1,cp2,graph).sort(key=len)

                if edps_cp1_to_cp2 == None:
                    edps_cp1_to_cp2 = [[cp1,cp2]]

                edps_cp3_to_cp2 = all_edps(cp3,cp2,graph).sort(key=len)

                if edps_cp3_to_cp2 == None:
                    edps_cp3_to_cp2 = [[cp3,cp2]]
                
                edps_cp3_to_d = all_edps(cp3,destination,graph).sort(key=len)

                if edps_cp3_to_d == None:
                    edps_cp3_to_d = [[cp3,destination]]

                #print(f"EDPs from CP1 ({cp1}) to Source ({source}): {edps_cp1_to_s}")

                edps_cp1_to_cp2 = all_edps(cp1, cp2, graph)
                #print(f"EDPs from CP1 ({cp1}) to CP2 ({cp2}): {edps_cp1_to_cp2}")

                edps_cp3_to_cp2 = all_edps(cp3, cp2, graph)
                #print(f"EDPs from CP3 ({cp3}) to CP2 ({cp2}): {edps_cp3_to_cp2}")

                edps_cp3_to_d = all_edps(cp3, destination, graph)
                #print(f"EDPs from CP3 ({cp3}) to Destination ({destination}): {edps_cp3_to_d}")

                #draw_tree_with_highlights(graph,[source,cp1,cp2,cp3,destination])
                # Build trees for each sub-path
                tree_cp1_to_s = one_tree_with_checkpoint_for_faces(cp1, source, graph, edps_cp1_to_s[-1])    
                #print("[triple_checkpoint_pre] tree_cp1_to_s:", tree_cp1_to_s.nodes)
            
                tree_cp1_to_cp2 = one_tree_with_checkpoint(cp1, cp2, graph, edps_cp1_to_cp2[-1])
                #print("[triple_checkpoint_pre] tree_cp1_to_cp2:", tree_cp1_to_cp2.nodes)

                tree_cp3_to_cp2 = one_tree_with_checkpoint_for_faces(cp3, cp2, graph, edps_cp3_to_cp2[-1])
                #print("[triple_checkpoint_pre] tree_cp3_to_cp2:", tree_cp3_to_cp2.nodes)

                tree_cp3_to_d = one_tree_with_checkpoint(cp3, destination, graph, edps_cp3_to_d[-1])
                #print("[triple_checkpoint_pre] tree_cp3_to_d:", tree_cp3_to_d.nodes)

                # Save the paths and checkpoints
                paths[source][destination] = {
                    'cps': [cp1, cp2, cp3],
                    'edps_cp1_to_s': edps_cp1_to_s,
                    'edps_s_to_d': edps,
                    'edps_cp1_to_cp2': edps_cp1_to_cp2,
                    'edps_cp3_to_cp2': edps_cp3_to_cp2,
                    'edps_cp3_to_d': edps_cp3_to_d,
                    'tree_cp1_to_s': tree_cp1_to_s,
                    'tree_cp1_to_cp2': tree_cp1_to_cp2,
                    'tree_cp3_to_cp2': tree_cp3_to_cp2,
                    'tree_cp3_to_d': tree_cp3_to_d
                }
                #plot_paths_element(paths[source][destination],graph,source,destination)
                # if( len(tree_cp1_to_s.nodes)>14): 
                #     print_cut_structure(highlighted_nodes=[cp1,source],structure=tree_cp1_to_s,source=source,destination=cp1,save_plot=True,filename=f"graphen/OneTreeTriple_{cp1}_{source}.png")
                # if( len(tree_cp3_to_cp2.nodes)>14): 
                #     print_cut_structure(highlighted_nodes=[cp3,cp2],structure=tree_cp3_to_cp2,source=cp3,destination=cp2,save_plot=True,filename=f"graphen/OneTreeTriple_{cp3}_{cp2}.png")
    
        
    return paths

############################################################################################################################

############################################## MULTIPLETREES TRIPLE CHECKPOINT #############################################

def multiple_trees_triple_checkpooint_pre(graph):

    paths = {}
    print("[multipletrees triple pre] graph edges:", graph.edges)
    for source in graph.nodes:
        for destination in graph.nodes:
            if source != destination:

                if source not in paths:
                    paths[source] = {}

                # Compute all EDPs between source and destination
                edps = all_edps(source, destination, graph)
                edps.sort(key=len)

                # Filter EDPs to ensure they are at least 5 nodes long s-cp1-cp2-cp3-d
                valid_edps = [edp for edp in edps if len(edp) >= 5]

                # Handle special case where no valid EDP >= 5 is found
                if not valid_edps:
                    # Handle the case where source and destination are directly connected (len == 2)
                    

                    tree_from_s = nx.DiGraph()
                    #print("[triple_checkpoint] edps:", edps)
                    for i in range(len(edps[0])):
                        #print("i:", i)
                        tree_from_s.add_node(edps[0][i])
                        if i > 0:
                            # Füge eine Kante zwischen zwei aufeinanderfolgenden Knoten hinzu
                            #print(f"Edge: ({edps[0][i-1]}, {edps[0][i]})")
                            tree_from_s.add_edge(edps[0][i-1], edps[0][i])



                    paths[source][destination] = {
                        'cps': [destination],
                        'edps_s_to_d': edps,
                        'edps_cp1_to_s':edps,
                        'edps_cp1_to_cp2':edps,
                        'edps_cp3_to_cp2':edps,
                        'edps_cp3_to_d': edps,
                        'trees_cp1_to_s':[tree_from_s],
                        'trees_cp1_to_cp2':[tree_from_s],
                        'trees_cp3_to_cp2':[tree_from_s],
                        'trees_cp3_to_d':[tree_from_s]
                    }
                    #plot_paths_element(paths[source][destination],graph,source,destination)
                    continue

                # Select the longest valid EDP (assuming it's already sorted by length)
                longest_edp = valid_edps[-1]

                # Anzahl der Knoten prüfen
                if len(longest_edp) < 5:
                    raise ValueError("EDP must have at least 5 nodes for meaningful CP distribution.")

                # Divide the EDP into segments for CP selection
                num_segments = min(4, len(longest_edp) - 1)  # Ensure no out-of-bound segments
                segment_length = len(longest_edp) // num_segments

                # Calculate CP indices based on segment midpoints
                cp_indices = []
                for i in range(1, num_segments):
                    cp_idx = i * segment_length + segment_length // 2  # Midpoint of each segment
                    cp_idx = min(cp_idx, len(longest_edp) - 2)  # Ensure not too close to destination
                    cp_indices.append(cp_idx)

                # Select CPs based on calculated indices
                cps = [longest_edp[idx] for idx in cp_indices]

                # Validate CPs
                if len(set(cps)) < len(cps):
                    raise ValueError("Control points overlap. Adjust segmentation logic.")

                cp1 = cps[0]
                cp2 = cps[1]
                cp3 = cps[2]
                # Extract sub-paths
                
                edps_cp1_to_s = all_edps(cp1,source,graph).sort(key=len)
                
                #Special Case if Nodes are directly connected
                if edps_cp1_to_s == None:
                    edps_cp1_to_s = [[cp1,source]]
                
                edps_cp1_to_cp2 = all_edps(cp1,cp2,graph).sort(key=len)

                if edps_cp1_to_cp2 == None:
                    edps_cp1_to_cp2 = [[cp1,cp2]]

                edps_cp3_to_cp2 = all_edps(cp3,cp2,graph).sort(key=len)

                if edps_cp3_to_cp2 == None:
                    edps_cp3_to_cp2 = [[cp3,cp2]]
                
                edps_cp3_to_d = all_edps(cp3,destination,graph).sort(key=len)

                if edps_cp3_to_d == None:
                    edps_cp3_to_d = [[cp3,destination]]

                #print(f"EDPs from CP1 ({cp1}) to Source ({source}): {edps_cp1_to_s}")

                edps_cp1_to_cp2 = all_edps(cp1, cp2, graph)
                #print(f"EDPs from CP1 ({cp1}) to CP2 ({cp2}): {edps_cp1_to_cp2}")

                edps_cp3_to_cp2 = all_edps(cp3, cp2, graph)
                #print(f"EDPs from CP3 ({cp3}) to CP2 ({cp2}): {edps_cp3_to_cp2}")

                edps_cp3_to_d = all_edps(cp3, destination, graph)
                #print(f"EDPs from CP3 ({cp3}) to Destination ({destination}): {edps_cp3_to_d}")



                ####################################################################

                trees_cp1_to_s = multiple_trees_with_checkpoint_for_faces(cp1,source,graph,edps_cp1_to_s)
                

                ########################################################################

                trees_cp1_to_cp2 = multiple_trees_with_checkpoint(cp1,cp2,graph,edps_cp1_to_cp2)
                
                for tree in trees_cp1_to_cp2:
                    for node in tree:
                        tree.nodes[node]['pos'] = graph.nodes[node]['pos']

                #EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben, führen nicht zum Ziel und müssen gelöscht werden
                trees_cp1_to_cp2 = remove_single_node_trees(trees_cp1_to_cp2)

                ##########################################################################

                trees_cp3_to_cp2 = multiple_trees_with_checkpoint_for_faces(cp3,cp2,graph,edps_cp3_to_cp2)
                
                ##########################################################################

                trees_cp3_to_d = multiple_trees_with_checkpoint(cp3,destination,graph,edps_cp3_to_d)
                
                for tree in trees_cp3_to_d:
                    for node in tree:
                        tree.nodes[node]['pos'] = graph.nodes[node]['pos']

                #EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben, führen nicht zum Ziel und müssen gelöscht werden
                trees_cp3_to_d = remove_single_node_trees(trees_cp3_to_d)

                # Save the paths and checkpoints
                paths[source][destination] = {
                    'cps': [cp1, cp2, cp3],
                    'edps_cp1_to_s': edps_cp1_to_s,
                    'edps_s_to_d': edps,
                    'edps_cp1_to_cp2': edps_cp1_to_cp2,
                    'edps_cp3_to_cp2': edps_cp3_to_cp2,
                    'edps_cp3_to_d': edps_cp3_to_d,
                    'trees_cp1_to_s': trees_cp1_to_s,
                    'trees_cp1_to_cp2': trees_cp1_to_cp2,
                    'trees_cp3_to_cp2': trees_cp3_to_cp2,
                    'trees_cp3_to_d': trees_cp3_to_d
                }
                # #plot_paths_element(paths[source][destination],graph,source,destination)
                # if( len(trees_cp1_to_s.nodes)>14): 
                #     print_cut_structure(highlighted_nodes=[cp1,source],structure=trees_cp1_to_s,source=source,destination=cp1,save_plot=True,filename=f"graphen/MultipleTreesTriple_{cp1}_{source}.png")
                # if( len(trees_cp3_to_cp2.nodes)>14):
                #     print_cut_structure(highlighted_nodes=[cp3,cp2],structure=trees_cp3_to_cp2,source=cp3,destination=cp2,save_plot=True,filename=f"graphen/MultipleTreesTriple_{cp3}_{cp2}.png")
    
    
    
          
    return paths

#################################################### MULTIPLETREES FOR FACE ROUTING ################################################

##########################################################################################################################################

def multiple_trees_for_faces_pre(graph):
    paths = {}
    
    #draw_tree_with_highlights(graph)
    print("[MultipleTreesForFacesPre] Start Precomputation")
    for source in graph.nodes:
       
        for destination in graph.nodes:
            
            if source != destination:
                
                edps = all_edps(source, destination, graph) #Bildung der EDPs
                
                edps.sort(key=len, reverse=True) #Sortierung der EDPs
                
                longest_edp = edps[len(edps)-1]

                #special case if the s,d pair is connected and this is the only edp
                if(len(longest_edp) == 2):

                    if source not in paths:
                        paths[source] = {}
                    #print("Special case for : ", source, "-", destination)


                    tree_from_s = nx.DiGraph()
                    tree_from_s.add_node(source)
                    tree_from_s.add_node(destination)
                    
                    tree_from_s.nodes[source]['pos'] = graph.nodes[source]['pos']
                    tree_from_s.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    tree_from_s.add_edge(source,destination)

                    paths[source][destination] = {
                                                'structure': tree_from_s,
                                            }
                    continue
                
                edps_s_to_d = all_edps(source, destination, graph)
                edps_s_to_d.sort(key=len)
                
                trees_s_to_d = multiple_trees_with_checkpoint_for_faces(source,destination,graph,edps_s_to_d)
                                                        
                if source in paths:
                    paths[source][destination] = { 
                                                'structure': trees_s_to_d,
                                                }
                else:
                    paths[source] = {}
                    paths[source][destination] = {
                                                'structure': trees_s_to_d,
                    }
                #print(f"Structure for {source} to {destination} has been computed")
                #draw_graph_with_highlighted_edge(paths[source][destination]['structure'], source, destination, ())
                #if( len(combined_tree.nodes)>14): 
                #    print_cut_structure(highlighted_nodes=[biggest_source,biggest_destination],structure=combined_tree,source=source,destination=destination,save_plot=True,filename=f"graphen/MultipleTreesForFaces_{source}_{destination}.png")
    
        
       
    return paths


################################ Hilfsfunktionen #####################################################################

def find_faces(G):
    """
    Findet alle Flächen eines planaren Graphen.
    Nutzt die vorhandenen Positionsattribute der Knoten.
    """
    face_nodes = ()
    half_edges_in_faces = set()
    faces = []
    #print("Checkpoint 5.1")
    pos = {node: G.nodes[node]['pos'] for node in G.nodes}  # Verwende die Positionen aus dem Graphen

    for node in G.nodes:
        for dest in nx.neighbors(G, node):
            if (node, dest) not in half_edges_in_faces:
                found_half_edges = set()
                try:
                    face_nodes = G.traverse_face(node, dest, found_half_edges)
                except Exception as e:
                    #print("Checkpoint 5.1 ERROR")
                    nx.draw(G, pos, with_labels=True, node_size=700, node_color="red", font_size=8)
                    plt.show()
                    traceback.print_exc()
                   #print(f"An unexpected error occurred: {e}")

                half_edges_in_faces.update(found_half_edges)
                face_graph = G.subgraph(face_nodes).copy()

                # Nutze die Positionen für die Knoten im Face-Subgraphen
                for face_node in face_graph.nodes:
                    face_graph.nodes[face_node]['pos'] = pos[face_node]

                faces.append(face_graph)
    #print("Checkpoint 5.2")
    # Den gesamten Graphen als letztes Face hinzufügen
    graph_last = G.copy()
    for node in graph_last:
        graph_last.nodes[node]['pos'] = pos[node]

    faces.append(graph_last)
    return faces

def plot_tree_with_highlighted_nodes(tree, source, destination, highlighted_nodes):
    """
    Zeichnet den Baum mit hervorgehobenen Knoten. Verwendet vorhandene Positionen aus den Knotenattributen.
    """
    # Verwende die vorhandenen Positionen der Knoten im Baum
    pos = {node: tree.nodes[node]['pos'] for node in tree.nodes()}

    # Farben für die Knoten je nach Rolle (Quelle, Ziel, hervorgehoben)
    node_colors = []
    for node in tree.nodes():
        if node == source:
            node_colors.append('red')
        elif node == destination:
            node_colors.append('green')
        elif node in highlighted_nodes:
            node_colors.append('yellow')
        else:
            node_colors.append('skyblue')

    # Erstelle die Legende
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='Source'),
        Patch(facecolor='green', edgecolor='black', label='Destination'),
        Patch(facecolor='yellow', edgecolor='black', label='Highlighted'),
        Patch(facecolor='skyblue', edgecolor='black', label='Other Nodes')
    ]

    # Zeichne den Graphen mit den vorhandenen Positionen
    plt.figure(figsize=(10, 10))
    nx.draw(tree, pos=pos, with_labels=True, node_color=node_colors)

    # Titel und Legende hinzufügen
    plt.title(f"{source} to {destination}")
    plt.legend(handles=legend_elements, loc='upper left')
    plt.show()

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

def plot_faces(G, faces, title="Faces Plot"):
    """
    Visualisiert die Flächen eines Graphen.
    
    Args:
    - G: Der Graph (Graph oder PlanarEmbedding), aus dem die Faces stammen.
    - faces: Liste der Faces, entweder als Knotenlisten oder Subgraphen.
    - title: Titel für den Plot.
    """
    # Extrahiere die Positionen der Knoten
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        raise ValueError("Der Graph enthält keine 'pos'-Attribute für die Knoten.")
    
    # Zeichne den Graphen
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue", edge_color="gray")
    
    # Zeichne die Flächen
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'pink']
    for i, face in enumerate(faces):
        if i == len(faces)-1:
            continue
        # Wenn die Flächen als Listen von Knoten gegeben sind
        if isinstance(face, list):
            face_edges = [(face[j], face[(j + 1) % len(face)]) for j in range(len(face))]
        # Wenn die Flächen als Subgraphen gegeben sind
        elif isinstance(face, nx.Graph):
            face_edges = list(face.edges)
        else:
            raise ValueError("Unbekanntes Format der Face-Daten.")

        # Zeichne die Face-Kanten
        nx.draw_networkx_edges(G, pos, edgelist=face_edges, edge_color=colors[i % len(colors)], width=2)
    
    plt.title(title)
    plt.show()

def draw_tree_with_highlighted_nodes(tree, nodes):


    """
    Zeichnet einen Baum-Graphen und hebt bestimmte Knoten hervor.

    Parameter:
    - tree: NetworkX-Graph-Objekt, das den Baum darstellt.
    - nodes: Liste von Knoten, die hervorgehoben werden sollen.
    """
    # Verwende bereits vorhandene Positionen der Knoten
    pos = nx.get_node_attributes(tree, 'pos')

    # Zeichne alle Knoten im Baum
    plt.figure(figsize=(8, 6))
    nx.draw(tree, pos, with_labels=True, node_size=500, node_color="lightblue", font_weight="bold")

    # Hervorheben der speziellen Knoten
    if nodes:
        nx.draw_networkx_nodes(tree, pos, nodelist=nodes, node_color="orange", node_size=700)
        print(f"Hervorgehobene Knoten: {nodes}")

    # Zeichne den Baum
    plt.title("Baum mit hervorgehobenen Knoten")
    plt.show()

def draw_tree_with_highlights(tree, nodes=None, fails=None, current_edge=None):
    """
    Zeichnet einen Baum-Graphen und hebt bestimmte Knoten, fehlerhafte Kanten und die aktuelle Kante hervor.

    Parameter:
    - tree: NetworkX-Graph-Objekt, das den Baum darstellt.
    - nodes: Liste von Knoten, die hervorgehoben werden sollen (optional).
    - fails: Liste von fehlerhaften Kanten, die hervorgehoben werden sollen (optional).
    - current_edge: Aktuelle Kante, die hervorgehoben werden soll (optional).
    """
    pos = {node: tree.nodes[node]['pos'] for node in tree.nodes}  # Positionen der Knoten

    plt.figure(figsize=(10, 8))

    # Zeichne alle Kanten in Grau
    nx.draw_networkx_edges(tree, pos, edge_color='gray')

    # Zeichne fehlerhafte Kanten in Rot, falls vorhanden
    if fails:
        failed_edges = [(u, v) for u, v in fails if tree.has_edge(u, v)]
        nx.draw_networkx_edges(tree, pos, edgelist=failed_edges, edge_color='red', width=2)
        #print(f"Hervorgehobene Kanten (Fails): {fails}")

    # Highlight aktuelle Kante in Blau, falls vorhanden
    if current_edge:
        if tree.has_edge(*current_edge):
            nx.draw_networkx_edges(tree, pos, edgelist=[current_edge], edge_color='blue', width=2)
            #print(f"Aktuelle Kante hervorgehoben: {current_edge}")

    # Zeichne alle Knoten
    nx.draw_networkx_nodes(tree, pos, node_color='lightgray', node_size=500)
    nx.draw_networkx_labels(tree, pos)

    # Hervorheben spezieller Knoten in Orange, falls vorhanden
    if nodes:
        nx.draw_networkx_nodes(tree, pos, nodelist=nodes, node_color="orange", node_size=700)
        #print(f"Hervorgehobene Knoten: {nodes}")

    #plt.title("Baum mit hervorgehobenen Knoten, Kanten und aktueller Kante")
    plt.show()

def plot_paths_element(paths_element, tree, source, destination):
    """
    Plots a graph based on a single paths[source][destination] element, with node positions.

    Parameters:
        paths_element (dict): A single paths[source][destination] element containing cps and tree information.
        tree (nx.Graph): A NetworkX graph containing node positions.
    """
    # Initialisiere den Graphen
    G = nx.DiGraph()

    # Verarbeite die Struktur
    cps = paths_element.get('cps', [])  # Sicherstellen, dass 'cps' existiert

    # Füge Kontrollpunkte und Quelle/Ziel als spezielle Knoten hinzu
    special_nodes = {
        source: "yellow",
        destination: "green",
    }

    # Weisen Sie Farben dynamisch zu, wenn CPS vorhanden sind
    cp_colors = ["orange", "purple", "pink"]
    for idx, cp in enumerate(cps):
        if idx < len(cp_colors):
            special_nodes[cp] = cp_colors[idx]

    # Füge Knoten hinzu
    for cp in cps:
        G.add_node(cp)

    # Füge Kanten aus den Bäumen hinzu
    for key, edges in paths_element.items():
        if key.startswith('tree_'):
            if isinstance(edges, list):
                if all(isinstance(edge, tuple) for edge in edges):
                    G.add_edges_from(edges)
                else:
                    raise ValueError(f"Invalid edges format for {key}. Expected a list of tuples, got: {edges}")
            elif isinstance(edges, nx.Graph):
                G.add_edges_from(edges.edges())
            else:
                raise ValueError(f"Invalid edges format for {key}. Expected a list or a NetworkX Graph, got: {type(edges)}")

    # Bestimme Knotenfarben
    node_colors = []
    for node in G.nodes():
        if node in special_nodes:
            node_colors.append(special_nodes[node])
        else:
            node_colors.append("gray")  # Alle anderen Knoten in Grau

    # Positionen der Knoten aus dem Baum holen
    pos = {node: tree.nodes[node]['pos'] for node in tree.nodes() if node in G.nodes()}

    # Zeichne den Graphen
    nx.draw(G, pos, with_labels=True, node_color=node_colors, node_size=700, font_size=10, font_weight="bold")
    plt.show()

def print_cut_structure(highlighted_nodes, structure, source, destination,cut_edges=[], fails=[], current_edge=None, save_plot=False, filename="failedgraphs/graph.png"):
    pos = nx.get_node_attributes(structure, 'pos')
    
    plt.figure(figsize=(10, 10))
    
    # Draw the entire structure
    nx.draw(structure, pos, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    
    # Highlight the specified nodes
    nx.draw_networkx_nodes(structure, pos, nodelist=highlighted_nodes, node_color='red')
    
    # Highlight the cut edges
    nx.draw_networkx_edges(structure, pos, edgelist=cut_edges, edge_color='blue', width=2)
    
    # Highlight the source and destination nodes
    nx.draw_networkx_nodes(structure, pos, nodelist=[source], node_color='green')
    nx.draw_networkx_nodes(structure, pos, nodelist=[destination], node_color='purple')
    
    # Highlight the current edge if provided
    if current_edge:
        nx.draw_networkx_edges(structure, pos, edgelist=[current_edge], edge_color='green', width=2, style='dashed')
    
    # Highlight the failed edges if provided
    valid_fails = [edge for edge in fails if edge[0] in pos and edge[1] in pos]
    if valid_fails:
        nx.draw_networkx_edges(structure, pos, edgelist=valid_fails, edge_color='black', width=2, style='dotted')
    
    if save_plot:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        # Check if the filename already exists and append a number if it does
        base, ext = os.path.splitext(filename)
        counter = 1
        new_filename = filename
        while os.path.exists(new_filename):
            new_filename = f"{base}_{counter}{ext}"
            counter += 1
        
        plt.savefig(new_filename)
        plt.close()
    else:
        plt.show()

def build_face_graph(faces):
    """Erstellt einen Graphen, in dem jedes Face ein Knoten ist und Kanten existieren, wenn Faces gemeinsame Kanten haben."""
    face_graph = nx.Graph()
  
    face_edges = []  # Speichert alle Kanten eines Faces
   
    for i, face in enumerate(faces):
        face_graph.add_node(i)  # Face als Knoten hinzufügen
        edges = {tuple(sorted((face[j], face[(j + 1) % len(face)]))) for j in range(len(face))}
        face_edges.append(edges)
    # Verbinde Faces, die gemeinsame Kanten haben
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
             if face_edges[i] & face_edges[j]:  # Falls gemeinsame Kanten existieren
                face_graph.add_edge(i, j)
   
    return face_graph

def greedy_coloring(graph):
    """Weist den Knoten (Faces) im Graphen Farben zu, sodass keine benachbarten Faces dieselbe Farbe haben."""
    colors = {}
    sorted_nodes = sorted(graph.nodes(), key=lambda x: graph.degree(x), reverse=True)  # Welsh-Powell Heuristik
    
    available_colors = ["red", "blue", "green", "purple", "orange", "cyan", "pink", "yellow", "brown"]
    
    for node in sorted_nodes:
        neighbor_colors = {colors[neighbor] for neighbor in graph.neighbors(node) if neighbor in colors}
        for color in available_colors:
            if color not in neighbor_colors:
                colors[node] = color
                break
    
    return colors

def print_faces(G, faces):
    """Gibt die Faces mit ihren Knoten im Terminal aus."""
    
    print("\n===== Gefundene Faces =====")
    
    for i, face in enumerate(faces):
        if len(face) < 3:
            continue  # Überspringe ungültige Faces
        
        # Sortiere die Knoten zur besseren Übersicht
        sorted_face = sorted(face)
        
        # Falls Knotenkoordinaten mit ausgegeben werden sollen:
        coords = [G.nodes[node]['pos'] for node in sorted_face if 'pos' in G.nodes[node]]
        
        print(f"Face {i+1}: {sorted_face}")
        # Falls du auch die Koordinaten möchtest, entkommentiere die nächste Zeile:
        # print(f"   -> Koordinaten: {coords}")

    print("==========================\n")

def draw_graph_with_colored_faces(G, faces, source, destination):
    """Zeichnet den Graphen und füllt die Faces mit unterschiedlichen Farben (keine zwei benachbarten Faces haben dieselbe Farbe)."""
    print("\n===== Zeichne Graph mit gefärbten Faces =====")
    print(f"source: {source}, destination: {destination}")
    print(f"G.nodes[source]['pos']: {G.nodes[source]['pos']}")
    print(f"G.nodes[destination]['pos']: {G.nodes[destination]['pos']}")
    # Holen der Knotenpositionen aus den Attributen von G
    pos = {node: G.nodes[node]['pos'] for node in G.nodes if 'pos' in G.nodes[node]}

    plt.figure(figsize=(12, 6))

    # Zeichne alle Knoten und Kanten
    nx.draw(G, pos, with_labels=True, node_color="lightgray", edge_color="gray")

    # Erstelle den Face-Graphen und weise Farben zu
    face_graph = build_face_graph(faces)
    face_colors = greedy_coloring(face_graph)

    patches = []
    
    # Zeichne jedes Face als gefülltes Polygon mit der zugewiesenen Farbe
    for i, face in enumerate(faces):
        if len(face) < 3:
            continue  # Kein gültiges Polygon
        
        # Prüfe, ob alle Knoten eine Position haben
        if not all(node in pos for node in face):
            continue  # Falls eine Position fehlt, wird dieses Face übersprungen
        
        polygon_coords = np.array([pos[node] for node in face])
        polygon = Polygon(polygon_coords, closed=True, edgecolor="black", facecolor=face_colors[i], alpha=0.6)
        patches.append(polygon)

    # Zeichne die Faces
    p = PatchCollection(patches, match_original=True)
    plt.gca().add_collection(p)

    # Zeichne Source und Destination extra hervor
    nx.draw_networkx_nodes(G, pos, nodelist=[source], node_color="yellow", node_size=500, label="Source")
    nx.draw_networkx_nodes(G, pos, nodelist=[destination], node_color="red", node_size=500, label="Destination")

    plt.title("Graph mit gefärbten Faces (keine zwei benachbarten Faces haben dieselbe Farbe)")
    plt.show()

def draw_graph(tree, source, destination, graph):
    plt.figure(figsize=(8, 6))
    
    # Holen der gespeicherten Positionen aus dem ursprünglichen Graphen
    pos = {node: graph.nodes[node]['pos'] for node in graph.nodes if 'pos' in graph.nodes[node]}
    if not pos:
        pos = nx.spring_layout(tree)
    
    # Zeichne alle Kanten in Schwarz
    nx.draw(tree, pos, with_labels=True, node_color='lightblue', edge_color='black', font_weight='bold')
    nx.draw_networkx_nodes(tree, pos, nodelist=[source], node_color='yellow', node_size=500)
    nx.draw_networkx_nodes(tree, pos, nodelist=[destination], node_color='yellow', node_size=500)
    
    plt.show()

def should_debug(source, destination):

    return False
    return source == 49 and destination == 34

def draw_graph_with_highlighted_edge2(tree, source, destination, edge_list, current_edge):
    """Zeichnet den Graph mit speziellen Farben für Source, Destination, Paths und die aktuell eingefügte Kante."""
    
    pos = nx.get_node_attributes(tree, 'pos')
    #adding the reverse edges to the tree
    for edge in tree.edges:
        if (edge[1], edge[0]) not in tree.edges:
            tree.add_edge(edge[1], edge[0])

    if not pos:
        print("WARNING: No position attributes found! Graph might not be displayed correctly.")
    
    plt.figure(figsize=(10, 8))
    
    # Zeichne alle Kanten standardmäßig in Grau
    nx.draw(tree, pos, with_labels=True, edge_color='gray', node_color='lightgray', node_size=500, font_size=10)
    
    # Zeichne Source und Destination in speziellen Farben
    nx.draw_networkx_nodes(tree, pos, nodelist=[source], node_color='red', node_size=700, label='Source')
    nx.draw_networkx_nodes(tree, pos, nodelist=[destination], node_color='green', node_size=700, label='Destination')
    

    #print("Edge List: ", edge_list)
    

    colors = ['black', 'red', 'blue', 'green', 'purple', 'orange', 'pink', 'yellow', 'brown', 'cyan']
    color_index = 0
    for edges in edge_list:
        for edge in edges:
            edge = tuple(edge)
            if edge  in tree.edges or (edge[1], edge[0]) in tree.edges:
                
                edge_new = tuple(edge)
                try:
                    nx.draw_networkx_edges(tree, pos=pos, edgelist=[edge_new], edge_color=colors[color_index % len(colors)], width=2, alpha=0.5)
                except: 
                    flipped_edge = (edge_new[1], edge_new[0])
                    nx.draw_networkx_edges(tree, pos=pos, edgelist=[flipped_edge], edge_color=colors[color_index % len(colors)], width=2, alpha=0.5)
        
        color_index += 1
    
    # Zeichne die aktuelle Kante in Blau
    if current_edge:
        nx.draw_networkx_edges(tree, pos, edgelist=[current_edge], edge_color='blue', width=3, alpha=1.0)
    
    plt.legend()
    plt.show()
