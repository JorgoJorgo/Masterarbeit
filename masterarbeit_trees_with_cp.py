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

import matplotlib.pyplot as plt
import networkx as nx

def draw_complete_structure(faces, tree, graph,s,cp,d, with_labels=True, figsize=(10, 8)):
    pos = {node: graph.nodes[node]['pos'] for node in graph.nodes if 'pos' in graph.nodes[node]}
    print("[draw complete] s:",s)
    print("[draw complete] cp:",cp)
    print("[draw complete] d:",d)
    plt.figure(figsize=figsize)
    print("[draw complete] face edges:",faces.edges)
    print("[draw complete] tree edges:",tree.edges)
    # --- Draw base graph ---
    nx.draw_networkx_edges(graph, pos, edge_color='lightgray', width=1, alpha=0.5)
    nx.draw_networkx_nodes(graph, pos, node_size=100, node_color='lightgray', alpha=0.7)
    
    # --- Draw face structure (as a subgraph, in light blue) ---
    nx.draw_networkx_edges(faces, pos, edge_color='dodgerblue', width=2, alpha=0.4)
    nx.draw_networkx_nodes(faces, pos, node_size=120, node_color='dodgerblue', alpha=0.3)

    # --- Draw tree structure (as a subgraph, in dark green) ---
    nx.draw_networkx_edges(tree, pos, edge_color='forestgreen', width=2.5, alpha=0.9)
    nx.draw_networkx_nodes(tree, pos, node_size=140, node_color='forestgreen', alpha=0.6)

    # --- Optional: Draw labels ---
    if with_labels:
        nx.draw_networkx_labels(graph, pos, font_size=10)

    plt.axis('off')
    plt.title("combined structure: Faces, Tree und Originalgraph")
    plt.tight_layout()
    plt.show()



#################################################### BASE ALGORITHMS FOR TREE-BUILDING ################################################

#ein großer baum für alle edps, längster EDP wird als erstes erweitert, für normale baumstrukturen mit tree routing
def multiple_trees_with_checkpoint(source, destination, graph, all_edps):
    #print(f"[MultipleTreesWithCheckpoint] Start for {source} -> {destination}")
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

# a large tree for all edps, the longest EDP is extended first, ONLY FOR STRUCTURES WHERE FACE ROUTING IS USED
def multiple_trees_with_checkpoint_for_faces(source, destination, graph, all_edps):

    """Creates multiple trees with a checkpoint considering the faces."""

    debug = False
    tree = nx.Graph()
    tree.add_node(source)
    
    debug = should_debug(source, destination)

    # print(f"Start building Trees for {source} -> {destination}")

    # each tree initially consists of the edp
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


    # draw_graph_with_highlighted_edge(tree, source, destination, ())
    # go through each edp and extend it
    for i in range(len(all_edps)):

        # pathToExtend always contains all nodes of the trees so that it can be checked which neighbors can still be added

        pathToExtend = all_edps[i]
        
        nodes = pathToExtend

        # print(f"[Tree {i} Start] Building Tree with {len(nodes)} nodes.")

        # print(f"Treenodes Nodes: {tree.nodes}")
        previous_edge = None

        for j in range(len(pathToExtend)):

            it = 0
            
            while it < len(nodes):
                # print("Nodes: ", nodes)
                # print("Nodes[it]", nodes[it])

                neighbors = list(nx.neighbors(graph, nodes[it]))

                for k in range(len(neighbors)):
                    if ((nodes[it], neighbors[k])in tree.edges) or ((neighbors[k], nodes[it]) in tree.edges):
                         continue
                    # endif
                    if (nodes[it], neighbors[k]) == previous_edge or (neighbors[k],nodes[it]) == previous_edge:
                        continue

                    previous_edge = (nodes[it], neighbors[k])
                    fake_tree = tree.copy()
                    
                    neighbor_accepted = False 
                    
                    # add edge (nodes[it], neighbors[k]) to the fake_tree to check if the face condition is violated
                    
                    fake_tree.add_node(neighbors[k])
                    
                    fake_tree.nodes[neighbors[k]]['pos'] = graph.nodes[neighbors[k]]['pos']
                    
                    fake_tree.add_edge(nodes[it], neighbors[k])
                    
                    fake_tree.nodes[nodes[it]]['pos'] = graph.nodes[nodes[it]]['pos']
                    # draw_graph_with_highlighted_edge(fake_tree, source, destination, (nodes[it], neighbors[k]))
                    
                    extra_edge = False
                    
                    if graph.has_edge(destination, neighbors[k]) or graph.has_edge(neighbors[k], destination):
                        extra_edge = True
                        fake_tree.add_edge(neighbors[k], destination)
                        fake_tree.nodes[destination]['pos'] = graph.nodes[destination]['pos']
                    # endif
                    
                    # for node in fake_tree.nodes:
                    #    fake_tree.nodes[node]['pos'] = graph.nodes[node]['pos']

                    faces  = find_faces_pre(fake_tree,source,destination)
                    # print("Faces: ", faces)
                    if(debug):
                        draw_graph_with_highlighted_edge(fake_tree, source, destination, (nodes[it], neighbors[k]))
                    if len(faces) > 0:
                        for face in faces:
                            if source in face and destination in face and smallest_face in faces:
                                neighbor_accepted = True
                                break
                            # endif
                        # endfor
                    # endif

                    # if the neighbor can be accepted, then add the edge to the original tree
                    if neighbor_accepted:

                        
                        tree.add_node(neighbors[k])
                        tree.add_edge(nodes[it], neighbors[k])
                        if extra_edge:
                            tree.add_edge(neighbors[k], destination)
                            nodes.append(destination)
                        # endif
                        
                        # for node in tree.nodes:
                        #    tree.nodes[node]['pos'] = graph.nodes[node]['pos']
                        nodes.append(neighbors[k])
                        tree.nodes[neighbors[k]]['pos'] = graph.nodes[neighbors[k]]['pos']
                        tree.nodes[nodes[it]]['pos'] = graph.nodes[nodes[it]]['pos']
                    # endif
                # endfor   

                it += 1
            # endwhile
            # draw_graph_with_highlighted_edge(tree, source, destination, ())

            
            # pruning the tree by removing redundant paths, which do not lead to the destination
            
            changed = True
            # print("Starting Pruning")
            while changed:
                old_edges = len(tree.edges)
                nodes_to_check = list(tree.nodes)  # Create a list of nodes to iterate over

                for node in nodes_to_check:
                    # print("checking node: ", node)
                    
                    if node != source and node != destination and tree.degree(node) == 1:
                        accept_removal = False
                        fake_tree = tree.copy()
                        neighbor = list(nx.neighbors(fake_tree, node))[0]
                        # print("Neighbor: ", neighbor)
                        
                        if (node, neighbor) in fake_tree.edges:
                            # draw_graph_with_highlighted_edge(fake_tree, source, destination, (node, neighbor))
                            fake_tree.remove_edge(node, neighbor)
                        else:
                            # draw_graph_with_highlighted_edge(fake_tree, source, destination, (neighbor, node))
                            fake_tree.remove_edge(neighbor, node)
                        
                        fake_tree.remove_node(node)
                        
                        # for node2 in fake_tree.nodes:
                        #    fake_tree.nodes[node2]['pos'] = graph.nodes[node2]['pos']
                        
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
        # endfor
    # draw_graph_with_highlighted_edge(tree, source, destination, ())
    return tree

def multiple_trees_parallel_cp(source, destination, graph, all_edps):
    """Creates multiple trees with a checkpoint considering the faces in parallel."""
    
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
        #print("Starting a new iteration of expansion")
        
        for i in range(len(paths_to_extend)):
            checked_all = False
            checked_number = 0
            #print(f"Expanding path {i}")
            
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
                        #for edge in edge_lists[i]:
                        #    assert edge in tree.edges, f"Edge list {edge_lists[i]} not in tree edges!"  
                        #    assert edge in fake_tree.edges, f"Edge list {edge_lists[i]} not in fake_tree edges!"  

                        # Add position of the new node
                        tree.nodes[neighbor]['pos'] = graph.nodes[neighbor]['pos']
                        
                        
                        paths_to_extend[i].append(neighbor)
                        changed = True
                        added_edge = True
                        #print(f"Added edge ({node}, {neighbor})")
                        
                    if added_edge:
                        break
                
                #if no edge was added then we need to increase the iterator and check the next node of the current path
                if not added_edge:
                    it_list[i] += 1
                
                #if an edge was added we can switch to the next path and look for the next node to add
                else:
                    break
                
                #if we checked all nodes of the current path we can switch to the next path
                if checked_number >= len(paths_to_extend[i]):
                    #print(f"Checked all nodes of {paths_to_extend[i]}")
                    checked_all = True

                #reset the iterator if we checked to the end of the path, in order to start at the beginning of the path again
                if it_list[i] >= len(paths_to_extend[i]):
                    it_list[i] = 0
    

    #now the tree needs to be pruned
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
                
                #for node2 in fake_tree.nodes:
                #    fake_tree.nodes[node2]['pos'] = graph.nodes[node2]['pos']
                
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

# A large tree, created by extending the given EDP "longest_edp", for normal tree structures with tree routing
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
                if (not tree.has_node(j)) and (j!= destination): #not part of tree already and not the destination
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
    for node in tree.nodes:
        tree.nodes[node]['pos']=graph.nodes[node]['pos']

    #draw_tree_with_highlighted_nodes(tree,[source])

    while changed == True: #keep trying to shorten until no more can be shortened 
        
        old_tree = tree.copy()
        remove_redundant_paths(source, destination, tree, graph)
        changed = tree.order() != old_tree.order() # order returns the number of nodes in the graph.

    
    
    rank_tree(tree , source,longest_edp)

    connect_leaf_to_destination(tree, source, destination)

    tree.add_edge(longest_edp[len(longest_edp)-2],destination)

    #add 'rank' property to the added destination, -1 for highest priority in routing
    tree.nodes[destination]["rank"] = -1

    for node in tree.nodes:
        tree.nodes[node]['pos'] = graph.nodes[node]['pos']
        #node['pos'] = list(graph.nodes)[node]['pos']

    #draw_tree_with_highlighted_nodes(tree,[source])
    return tree    
        
    #end if

# a large tree, created by extending the given EDP "longest_edp", ONLY FOR STRUCTURES WHERE FACE ROUTING IS USED
def one_tree_with_checkpoint_for_faces(source, destination, graph, longest_edp):
    """Creates a tree from the EDP for face routing"""
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
                
                #for node2 in fake_tree.nodes:
                #    fake_tree.nodes[node2]['pos'] = graph.nodes[node2]['pos']
                
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
  
# goes through all edges of the graph again after an algorithm is finished and adds neighbors if S-D-face exists ONLY FOR STRUCTURES WHERE FACE ROUTING IS USED
def expand_face_structure(source, destination, graph, face_structure, tree_structure):

    #print("[ExpandFaceStructure] Starting the Expansion")

    if isinstance(tree_structure, list) and all(isinstance(item, nx.DiGraph) for item in tree_structure):
        #print("Check True")
        combined_tree_structure = nx.Graph()
        # for the MultipleTrees algorithms, all trees must first be combined into one
        for tree in tree_structure:
            combined_tree_structure.add_edges_from(tree.edges)


        tree_structure = combined_tree_structure

    debug= False
    added_edges = list()
    faces  = find_faces_pre(face_structure,source,destination)
    smallest_face = None
    smallest_face_size = 1000000
    for face in faces:
        if len(face) < smallest_face_size:
            smallest_face = face
            smallest_face_size = len(face)

    #draw_graph_with_highlighted_edge(face_structure, source, destination, ())

    # Output original edge lists
    #print("Original all_edges_graph:", list(graph.edges))
    #print("Original all_edges_face_structure:", list(face_structure.edges))
    #print("Original all_edges_tree_structure:", list(tree_structure.edges))

    # Convert lists to sets and store each edge symmetrically
    all_edges_graph = set((min(u, v), max(u, v)) for u, v in graph.edges)
    all_edges_face_structure = set((min(u, v), max(u, v)) for u, v in face_structure.edges)

    #print(f"Tree Structure check: {len(tree_structure)>0}")
    #print(f"Tree Structure: {tree_structure}")
    
    all_edges_tree_structure = set((min(u, v), max(u, v)) for u, v in tree_structure.edges)

    # Output after conversion to sets
    #print("\nSet all_edges_graph:", all_edges_graph)
    #print("Set all_edges_face_structure:", all_edges_face_structure)
    #print("Set all_edges_tree_structure:", all_edges_tree_structure)

    # Remove edges
    all_edges_graph -= all_edges_face_structure
    #print("\nall_edges_graph after removing face_structure edges:", all_edges_graph)
    
    
    all_edges_graph -= all_edges_tree_structure
    #print("\nall_edges_graph after removing tree_structure edges:", all_edges_graph)

    # If you need a list again
    all_edges_graph = list(all_edges_graph)

    # Final result output
    #print("\nFinal all_edges_graph list:", all_edges_graph)


    changed = True
    while changed:
        changed = False
        # Capture all nodes in face_structure
        face_nodes = face_nodes = set(face_structure.nodes)

        # Filter edges in all_edges_graph where at least one node is in face_nodes
        potential_edges = [edge for edge in all_edges_graph if edge[0] in face_nodes or edge[1] in face_nodes]

        # Output result
        #print("\nNodes in face_structure:", face_nodes)
        #print("\nPotential edges:", potential_edges)

        # Go through each potential edge to check if it still provides an s-d-face after adding
        for edge in potential_edges:
            fake_face_structure = face_structure.copy()
            fake_face_structure.add_edge(*edge)
            fake_face_structure.nodes[edge[0]]['pos'] = graph.nodes[edge[0]]['pos']
            fake_face_structure.nodes[edge[1]]['pos'] = graph.nodes[edge[1]]['pos']
            edge_accepted = False
            faces = find_faces_pre(fake_face_structure,source,destination)
            if smallest_face in faces:
                edge_accepted = True
            
            # As soon as the smallest s-d-face still exists, the edge can be added and it can be checked whether the new node also has a direct connection to d
            if edge_accepted:
                face_structure.add_edge(*edge)
                added_edges.append(edge)
                face_structure.nodes[edge[0]]['pos'] = graph.nodes[edge[0]]['pos']
                face_structure.nodes[edge[1]]['pos'] = graph.nodes[edge[1]]['pos']
    #endwhile

    changed = True
    # Here the structure is then shortened to avoid parts that are useless anyway
    while changed:
        old_edges = len(face_structure.edges)
        nodes_to_check = list(face_structure.nodes)  # Create a list of nodes to iterate over

        for node in nodes_to_check:
            #print("checking node: ", node)
            
            if node != source and node != destination and face_structure.degree(node) == 1:
                accept_removal = False
                fake_tree = face_structure.copy()
                neighbor = list(nx.neighbors(fake_tree, node))[0]
                #print("Neighbor: ", neighbor)
                
                if (node, neighbor) in fake_tree.edges:
                    #draw_graph_with_highlighted_edge(fake_tree, source, destination, (node, neighbor))
                    fake_tree.remove_edge(node, neighbor)
                else:
                    #draw_graph_with_highlighted_edge(fake_tree, source, destination, (neighbor, node))
                    fake_tree.remove_edge(neighbor, node)
                
                fake_tree.remove_node(node)
                
                #for node2 in fake_tree.nodes:
                #    fake_tree.nodes[node2]['pos'] = graph.nodes[node2]['pos']
                
                faces = find_faces_pre(fake_tree, source, destination)
                
                for face in faces:
                    if source in face and destination in face:
                        accept_removal = True
                        break
                
                if accept_removal:
                    if (node, neighbor) in face_structure.edges:
                        face_structure.remove_edge(node, neighbor)
                    else:
                        face_structure.remove_edge(neighbor, node)
                    face_structure.remove_node(node)
                    
        
        new_edges = len(face_structure.edges)
        changed = old_edges != new_edges

    #endwhile
    # Filter added_edges to keep only edges that are still in face_structure
    added_edges = [edge for edge in added_edges if face_structure.has_edge(*edge)]

    # Final output after cleanup
    #print("Final Added Edges:", added_edges)

    if debug:
        if len(added_edges) > 0:
            print_cut_structure([], face_structure,source,destination,added_edges,[],(),False,"test")
    
    return face_structure


#################################################### MULTIPLETREES WITH MIDDLE CHECKPOINT ################################################

##########################################################################################################################################

def multiple_trees_with_middle_checkpoint_parallel_pre(graph):
    paths = {}
    #draw_tree_with_highlights(graph)
    print("[MultipleTreesOneCheckpointParallelPre] Start Precomputation")
    for source in graph.nodes:
       
        for destination in graph.nodes:
            
            if source != destination:
                
                edps = all_edps(source, destination, graph) # Formation of EDPs
                
                edps.sort(key=len, reverse=True) # Sorting of EDPs
                
                longest_edp = edps[len(edps)-1]

                # Special case if the s,d pair is connected and this is the only edp
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

                # EDPs that could not be extended because other trees had already blocked the edges do not lead to the destination and must be deleted
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
                
                # Check if the graph is planar
                # print("Nodes of combined tree:", combined_tree.nodes)
                # print("Edges of combined tree:", combined_tree.edges)
                # is_planar, embedding = nx.check_planarity(combined_tree)

                # if is_planar:
                #     planar_embedding = create_planar_embedding_new(combined_tree, graph)
                    
                    
                    
                # else:
                #     print("The graph is not planar!")

                # # Now the trees cp->s must be added as planar embedding
                # # so that the faces can be found


               
                    
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
    #print("All Combinations: ", (len(graph.nodes) * len(graph.nodes)) - len(graph.nodes))
    combinations = 0
    for source in graph.nodes:
       
        for destination in graph.nodes:
            
            
            if source != destination:
                combinations += 1
                #print("Current Combination: ", combinations, " of ", (len(graph.nodes) * len(graph.nodes)) - len(graph.nodes))
                edps = all_edps(source, destination, graph) # Formation of EDPs
                
                edps.sort(key=len, reverse=True) # Sorting of EDPs
                
                longest_edp = edps[len(edps)-1]

                # Special case if the s,d pair is connected and this is the only edp
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

                # EDPs that could not be extended because other trees had already blocked the edges do not lead to the destination and must be deleted
                #trees_cp_to_s = remove_single_node_trees(trees_cp_to_s)
                
                #then build multiple trees cp->d
                
                trees_cp_to_d = multiple_trees_with_checkpoint(cp,destination,graph,edps_cp_to_d)
                
                for tree in trees_cp_to_d:
                    for node in tree:
                        tree.nodes[node]['pos'] = graph.nodes[node]['pos']

                # EDPs that could not be extended because other trees had already blocked the edges do not lead to the destination and must be deleted
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
                combined_edges=0
                for tree in trees_cp_to_d:
                    combined_edges += len(tree.edges)

                #print(f"[MultipleTrees] #Edges:{len(trees_cp_to_s.edges)+combined_edges}")
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
    #print("All Combinations: ", (len(graph.nodes) * len(graph.nodes)) - len(graph.nodes))
    combinations = 0
    for source in graph.nodes:
       
        for destination in graph.nodes:
            
            
            if source != destination:
                combinations += 1
                #print("Current Combination: ", combinations, " of ", (len(graph.nodes) * len(graph.nodes)) - len(graph.nodes))
                edps = all_edps(source, destination, graph) #Formation of EDPs
                
                edps.sort(key=len, reverse=False) #Sorting of EDPs
                
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

                #EDPs that could not be extended because other trees had already blocked the edges do not lead to the destination and must be deleted
                #trees_cp_to_s = remove_single_node_trees(trees_cp_to_s)
                
                #then build multiple trees cp->d
                
                trees_cp_to_d = multiple_trees_with_checkpoint(cp,destination,graph,edps_cp_to_d)
                
                for tree in trees_cp_to_d:
                    for node in tree:
                        tree.nodes[node]['pos'] = graph.nodes[node]['pos']

                #EDPs that could not be extended because other trees had already blocked the edges do not lead to the destination and must be deleted
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
    #print("All Combinations: ", (len(graph.nodes) * len(graph.nodes)) - len(graph.nodes))
    combinations = 0
    for source in graph.nodes:
       
        for destination in graph.nodes:
            
            
            if source != destination:
                combinations += 1
                #print("Current Combination: ", combinations, " of ", (len(graph.nodes) * len(graph.nodes)) - len(graph.nodes))
                edps = all_edps(source, destination, graph) #Formation of EDPs
                
                edps.sort(key=len, reverse=False) #Sorting of EDPs
                
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

                #EDPs that could not be extended because other trees had already blocked the edges do not lead to the destination and must be deleted
                #trees_cp_to_s = remove_single_node_trees(trees_cp_to_s)
                
                #then build multiple trees cp->d
                
                trees_cp_to_d = multiple_trees_with_checkpoint(cp,destination,graph,edps_cp_to_d)
                
                for tree in trees_cp_to_d:
                    for node in tree:
                        tree.nodes[node]['pos'] = graph.nodes[node]['pos']

                #EDPs that could not be extended because other trees had already blocked the edges do not lead to the destination and must be deleted
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
   #print("All Combinations: ", (len(graph.nodes) * len(graph.nodes)) - len(graph.nodes))
    combinations = 0
    for source in graph.nodes:
       
        for destination in graph.nodes:
            
            
            if source != destination:
                combinations += 1
                #print("Current Combination: ", combinations, " of ", (len(graph.nodes) * len(graph.nodes)) - len(graph.nodes))
                edps = all_edps(source, destination, graph) #Formation of EDPs
                
                edps.sort(key=len, reverse=False) #Sorting of EDPs
                
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

                #EDPs that could not be extended because other trees had already blocked the edges do not lead to the destination and must be deleted
                #trees_cp_to_s = remove_single_node_trees(trees_cp_to_s)
                
                #then build multiple trees cp->d
                
                trees_cp_to_d = multiple_trees_with_checkpoint(cp,destination,graph,edps_cp_to_d)
                
                for tree in trees_cp_to_d:
                    for node in tree:
                        tree.nodes[node]['pos'] = graph.nodes[node]['pos']

                #EDPs that could not be extended because other trees had already blocked the edges do not lead to the destination and must be deleted
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
   #print("All Combinations: ", (len(graph.nodes) * len(graph.nodes)) - len(graph.nodes))
    combinations = 0
    for source in graph.nodes:
       
        for destination in graph.nodes:
            
            
            if source != destination:
                combinations += 1
                #print("Current Combination: ", combinations, " of ", (len(graph.nodes) * len(graph.nodes)) - len(graph.nodes))
                edps = all_edps(source, destination, graph) #Formation of EDPs
                
                edps.sort(key=len, reverse=False) #Sorting of EDPs
                
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

                #EDPs that could not be extended because other trees had already blocked the edges do not lead to the destination and must be deleted
                #trees_cp_to_s = remove_single_node_trees(trees_cp_to_s)
                
                #then build multiple trees cp->d
                
                trees_cp_to_d = multiple_trees_with_checkpoint(cp,destination,graph,edps_cp_to_d)
                
                for tree in trees_cp_to_d:
                    for node in tree:
                        tree.nodes[node]['pos'] = graph.nodes[node]['pos']

                #EDPs that could not be extended because other trees had already blocked the edges do not lead to the destination and must be deleted
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

#################################################### MULTIPLETREES INVERS WITH DEGREE EXTENDED CHECKPOINT ################################################

##########################################################################################################################################
removed_edges_multtrees = []

def multiple_trees_invers_with_degree_checkpoint_extended_pre(graph):
    paths = {}
    #draw_tree_with_highlights(graph)
    print("[MultipleTreesOneCheckpointPre] Start Precomputation")
   #print("All Combinations: ", (len(graph.nodes) * len(graph.nodes)) - len(graph.nodes))
    combinations = 0
    for source in graph.nodes:
       
        for destination in graph.nodes:
            
            
            if source != destination:
                combinations += 1
                #print("Current Combination: ", combinations, " of ", (len(graph.nodes) * len(graph.nodes)) - len(graph.nodes))
                edps = all_edps(source, destination, graph) #Formation of EDPs
                
                edps.sort(key=len, reverse=False) #Sorting of EDPs
                
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

                

                #EDPs that could not be extended because other trees had already blocked the edges do not lead to the destination and must be deleted
                #trees_cp_to_s = remove_single_node_trees(trees_cp_to_s)
                
                #then build multiple trees cp->d
                
                trees_cp_to_d = multiple_trees_with_checkpoint(cp,destination,graph,edps_cp_to_d)
                trees_cp_to_s = expand_face_structure(source,cp,graph,trees_cp_to_s,trees_cp_to_d)
                for tree in trees_cp_to_d:
                    for node in tree:
                        tree.nodes[node]['pos'] = graph.nodes[node]['pos']

                #EDPs that could not be extended because other trees had already blocked the edges do not lead to the destination and must be deleted
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


#################################################### MULTIPLETREES WITH DEGREE EXTENDED CHECKPOINT ################################################

##########################################################################################################################################
removed_edges_multtrees = []

def multiple_trees_with_degree_checkpoint_extended_pre(graph):
    paths = {}
    #draw_tree_with_highlights(graph)
    print("[MultipleTreesOneCheckpointPre] Start Precomputation")
   #print("All Combinations: ", (len(graph.nodes) * len(graph.nodes)) - len(graph.nodes))
    combinations = 0
    for source in graph.nodes:
       
        for destination in graph.nodes:
            
            
            if source != destination:
                combinations += 1
                #print("Current Combination: ", combinations, " of ", (len(graph.nodes) * len(graph.nodes)) - len(graph.nodes))
                edps = all_edps(source, destination, graph) #Formation of EDPs
                
                edps.sort(key=len, reverse=True) #Sorting of EDPs
                
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

                

                #EDPs that could not be extended because other trees had already blocked the edges do not lead to the destination and must be deleted
                #trees_cp_to_s = remove_single_node_trees(trees_cp_to_s)
                
                #then build multiple trees cp->d
                
                trees_cp_to_d = multiple_trees_with_checkpoint(cp,destination,graph,edps_cp_to_d)
                trees_cp_to_s = expand_face_structure(source,cp,graph,trees_cp_to_s,trees_cp_to_d)
                for tree in trees_cp_to_d:
                    for node in tree:
                        tree.nodes[node]['pos'] = graph.nodes[node]['pos']

                #EDPs that could not be extended because other trees had already blocked the edges do not lead to the destination and must be deleted
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
                
                edps = all_edps(source, destination, graph) # Formation of EDPs
                
                edps.sort(key=len, reverse=True) # Sorting of EDPs
                
                longest_edp = edps[len(edps)-1]

                # Special case if the s,d pair is connected and this is the only edp
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
                                                'trees_cp_to_d': tree_from_d, 
                                                'edps_cp_to_d': [[source,destination]],
                                                'edps_s_to_d': [[source,destination]]
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
                                                'trees_cp_to_d': tree_from_d, 
                                                'edps_cp_to_d': [[source,destination]],
                                                'edps_s_to_d': [[source,destination]]
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

                # EDPs that could not be extended because other trees had already blocked the edges do not lead to the destination and must be deleted
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
                
                edps = all_edps(source, destination, graph) # Formation of EDPs
                
                edps.sort(key=len, reverse=True) # Sorting of EDPs
                
                longest_edp = edps[len(edps)-1]

                # Special case if the s,d pair is connected and this is the only edp
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
                                                'trees_cp_to_d': tree_from_d, 
                                                'edps_cp_to_d': [[source,destination]],
                                                'edps_s_to_d': [[source,destination]]
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
                                                'trees_cp_to_d': tree_from_d, 
                                                'edps_cp_to_d': [[source,destination]],
                                                'edps_s_to_d': [[source,destination]]
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

                # EDPs that could not be extended because other trees had already blocked the edges do not lead to the destination and must be deleted
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
                
                edps = all_edps(source, destination, graph) # Formation of EDPs
                
                edps.sort(key=len, reverse=True) # Sorting of EDPs
                
                longest_edp = edps[len(edps)-1]

                # Special case if the s,d pair is connected and this is the only edp
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
                # EDPs that could not be extended because other trees had already blocked the edges do not lead to the destination and must be deleted
                trees_cp_to_s = remove_single_node_trees(trees_cp_to_s)
           
                # Since no tree-routing s->cp takes place, but face-routing, all trees (cp->s) are combined into one large tree on which face-routing can be performed
                # Combine all trees into one large undirected tree
                combined_tree = nx.Graph()
                for tree in trees_cp_to_s:
                    tree = tree.to_undirected()  # Ensure the tree is undirected
                    for node in tree.nodes:
                            combined_tree.add_node(node)  # Add node without position
                    combined_tree.add_edges_from(tree.edges())  # Add edges

                for node in combined_tree.nodes:
                    combined_tree.nodes[node]['pos'] = graph.nodes[node]['pos']
         
                # Contains an nx.Graph planar, all trees in one graph with coordinates
                trees_cp_to_s = combined_tree
                
                # Then build multiple trees cp->d
                
                trees_cp_to_d = multiple_trees_with_checkpoint(cp,destination,graph,edps_cp_to_d)
                
                for tree in trees_cp_to_d:
                    for node in tree:
                        tree.nodes[node]['pos'] = graph.nodes[node]['pos']

                # EDPs that could not be extended because other trees had already blocked the edges do not lead to the destination and must be deleted
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

# Function to generate the 2 trees for each s->d pair (s->cp & cp->d)
# Each tree gets generated by expanding the longest edp of each pair
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
                
                # Now compute the chosen checkpoint  
                # First get the longest edp s->d    
                edps = all_edps(source, destination, graph)
                
                edps.sort(key=len)
                
                longest_edp = edps[len(edps)-1]
                
                # Special case if the s,d pair is connected and this is the only edp
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
                
                # Then select the middle node of the longest_edp
                
                cp = longest_edp[ int(len(longest_edp)/2)]
               
                # Then get the edps + longest_edps_cp_s and the longest_edps_cp_d
                edps_cp_to_s = all_edps(cp, source, graph)
                edps_cp_to_d = all_edps(cp, destination, graph)
                edps_cp_to_s.sort(key=len)
                edps_cp_to_d.sort(key=len)
                
                # And build trees out of the longest_edps_cp_s and the longest_edps_cp_d
                tree_cp_to_s = one_tree_with_checkpoint_for_faces(cp,source,graph,edps_cp_to_s[len(edps_cp_to_s)-1])

                #draw_tree_with_highlighted_nodes(tree_cp_to_s,[source,cp])

                tree_cp_to_d = one_tree_with_checkpoint(cp,destination,graph,edps_cp_to_d[len(edps_cp_to_d)-1])
                
                # Because the tree cp->s got built in reverse direction, the edges need to be reversed again
                # Data structure to give the needed information for the routing (edps, trees, checkpoint)
                
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
                #draw_tree_with_highlighted_nodes(tree_cp_to_s,[source,cp])
                #draw_tree_with_highlighted_nodes(graph,[destination,cp])
                #draw_tree_with_highlighted_nodes(tree_cp_to_d,[destination,cp])
                #if(source==0 and destination == 30):
                #print(f"[MiddleCheckpoint] #Edges:{len(tree_cp_to_s.edges)+len(tree_cp_to_d.edges)}")
                #draw_complete_structure(tree_cp_to_s,tree_cp_to_d,graph,s=source,cp=cp,d=destination)
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
                # fake_cp = longest_edp[ int(len(longest_edp)/2)]
                # if fake_cp != cp:
                #     print("Fake_cp != cp für :", source ,"-",destination)
                #     print(f"#Edges:{len(tree_cp_to_s.edges)+len(tree_cp_to_d.edges)}")

                #if(source==0 and destination == 30):
                #print(f"[Degree Checkpoint] #Edges:{len(tree_cp_to_s.edges)+len(tree_cp_to_d.edges)}")
                #draw_complete_structure(tree_cp_to_s,tree_cp_to_d,graph,s=source,cp=cp,d=destination)
                    
    return paths

########################################################################################################

#################################################### ONETREE WITH DEGREE CHECKPOINT ######################################################

##########################################################################################################################################


def one_tree_with_degree_checkpoint_extended_pre(graph):
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
                
                tree_cp_to_s = expand_face_structure(source,cp,graph,tree_cp_to_s,tree_cp_to_d)

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
                #print(f"[Betweenness Checkpoint] #Edges:{len(tree_cp_to_s.edges)+len(tree_cp_to_d.edges)}")
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
                #print(f"[Closeness Checkpoint] #Edges:{len(tree_cp_to_s.edges)+len(tree_cp_to_d.edges)}")
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

############################################### ONETREECHECKPOINT WITH SHORTEST EDP EXTENDED ##############################################

def one_tree_with_middle_checkpoint_shortest_edp_extended_pre(graph):
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

                tree_cp_to_s= expand_face_structure(source,cp,graph,tree_cp_to_s,tree_cp_to_d)

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
                #print(f"[Triple Checkpoint OneTree] #Edges:{len(tree_cp1_to_s.edges)+len(tree_cp1_to_cp2.edges)+len(tree_cp3_to_cp2.edges)+len(tree_cp3_to_d.edges)}")
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
    #print("[multipletrees triple pre] graph edges:", graph.edges)
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
                            # Add an edge between two consecutive nodes
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

                # Check the number of nodes
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
                
                # Special case if nodes are directly connected
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

                # EDPs that could not be extended because other trees had already blocked the edges do not lead to the destination and must be deleted
                trees_cp1_to_cp2 = remove_single_node_trees(trees_cp1_to_cp2)

                ##########################################################################

                trees_cp3_to_cp2 = multiple_trees_with_checkpoint_for_faces(cp3,cp2,graph,edps_cp3_to_cp2)
                
                ##########################################################################

                trees_cp3_to_d = multiple_trees_with_checkpoint(cp3,destination,graph,edps_cp3_to_d)
                
                for tree in trees_cp3_to_d:
                    for node in tree:
                        tree.nodes[node]['pos'] = graph.nodes[node]['pos']

                # EDPs that could not be extended because other trees had already blocked the edges do not lead to the destination and must be deleted
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
                # combined_edges1= 0
                # for tree in trees_cp1_to_cp2:
                #     combined_edges1 += len(tree.edges)
                # combined_edges2= 0
                # for tree in trees_cp3_to_d:
                #     combined_edges2 += len(tree.edges)
                # print(f"[Triple Checkpoint MultipleTrees] #Edges:{len(trees_cp1_to_s.edges)+combined_edges1+len(trees_cp3_to_cp2.edges)+combined_edges2}")
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

#################################################### MULTIPLETREES FOR FACE ROUTING EXTENDED ################################################

##########################################################################################################################################

def multiple_trees_for_faces_extended_pre(graph):
    paths = {}
    
    #draw_tree_with_highlights(graph)
    print("[MultipleTreesForFacesPre Extended] Start Precomputation")
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
                                                        
                fake_structure = nx.Graph()
                trees_s_to_d = expand_face_structure(source,destination,graph,trees_s_to_d,fake_structure)

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

################################ Helper Functions #####################################################################

def find_faces(G):
    """
    Finds all faces of a planar graph.
    Uses the existing position attributes of the nodes.
    """
    face_nodes = ()
    half_edges_in_faces = set()
    faces = []
    #print("Checkpoint 5.1")
    pos = {node: G.nodes[node]['pos'] for node in G.nodes}  # Use the positions from the graph

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

                # Use the positions for the nodes in the face subgraph
                for face_node in face_graph.nodes:
                    face_graph.nodes[face_node]['pos'] = pos[face_node]

                faces.append(face_graph)
    #print("Checkpoint 5.2")
    # Add the entire graph as the last face
    graph_last = G.copy()
    for node in graph_last:
        graph_last.nodes[node]['pos'] = pos[node]

    faces.append(graph_last)
    return faces

def plot_tree_with_highlighted_nodes(tree, source, destination, highlighted_nodes):
    """
    Draws the tree with highlighted nodes. Uses existing positions from the node attributes.
    """
    # Use the existing positions of the nodes in the tree
    pos = {node: tree.nodes[node]['pos'] for node in tree.nodes()}

    # Colors for the nodes depending on their role (source, destination, highlighted)
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

    # Create the legend
    legend_elements = [
        Patch(facecolor='red', edgecolor='black', label='Source'),
        Patch(facecolor='green', edgecolor='black', label='Destination'),
        Patch(facecolor='yellow', edgecolor='black', label='Highlighted'),
        Patch(facecolor='skyblue', edgecolor='black', label='Other Nodes')
    ]

    # Draw the graph with the existing positions
    plt.figure(figsize=(10, 10))
    nx.draw(tree, pos=pos, with_labels=True, node_color=node_colors)

    # Add title and legend
    plt.title(f"{source} to {destination}")
    plt.legend(handles=legend_elements, loc='upper left')
    plt.show()

def convert_to_planar_embedding(graph):
    """
    Converts a planar graph into a PlanarEmbedding structure.
    """
    is_planar, embedding = nx.check_planarity(graph)
    if not is_planar:
        raise ValueError("Graph is not planar and cannot be converted into a PlanarEmbedding.")
    # Transfer the node positions into the PlanarEmbedding object
    for node, data in graph.nodes(data=True):
        embedding.add_node(node, **data)
    return embedding

def plot_faces(G, faces, title="Faces Plot"):
    """
    Visualizes the faces of a graph.
    
    Args:
    - G: The graph (Graph or PlanarEmbedding) from which the faces originate.
    - faces: List of faces, either as node lists or subgraphs.
    - title: Title for the plot.
    """
    # Extract the positions of the nodes
    pos = nx.get_node_attributes(G, 'pos')
    if not pos:
        raise ValueError("The graph does not contain 'pos' attributes for the nodes.")
    
    # Draw the graph
    plt.figure(figsize=(8, 8))
    nx.draw(G, pos, with_labels=True, node_size=700, node_color="lightblue", edge_color="gray")
    
    # Draw the faces
    colors = ['blue', 'red', 'green', 'purple', 'orange', 'pink']
    for i, face in enumerate(faces):
        if i == len(faces)-1:
            continue
        # If the faces are given as lists of nodes
        if isinstance(face, list):
            face_edges = [(face[j], face[(j + 1) % len(face)]) for j in range(len(face))]
        # If the faces are given as subgraphs
        elif isinstance(face, nx.Graph):
            face_edges = list(face.edges)
        else:
            raise ValueError("Unknown format of face data.")

        # Draw the face edges
        nx.draw_networkx_edges(G, pos, edgelist=face_edges, edge_color=colors[i % len(colors)], width=2)
    
    plt.title(title)
    plt.show()

def draw_tree_with_highlighted_nodes(tree, nodes):


    """
    Draws a tree graph and highlights specific nodes.

    Parameters:
    - tree: NetworkX graph object representing the tree.
    - nodes: List of nodes to be highlighted.
    """
    # Use already existing positions of the nodes
    pos = nx.get_node_attributes(tree, 'pos')

    # Draw all nodes in the tree
    plt.figure(figsize=(8, 6))
    nx.draw(tree, pos, with_labels=True, node_size=500, node_color="lightblue", font_weight="bold")

    # Highlight the special nodes
    if nodes:
        nx.draw_networkx_nodes(tree, pos, nodelist=nodes, node_color="orange", node_size=700)
        print(f"Highlighted nodes: {nodes}")

    # Draw the tree
    plt.title("Tree with highlighted nodes")
    plt.show()

def draw_tree_with_highlights(tree, nodes=None, fails=None, current_edge=None):
    """
    Draws a tree graph and highlights specific nodes, failed edges, and the current edge.

    Parameters:
    - tree: NetworkX graph object representing the tree.
    - nodes: List of nodes to be highlighted (optional).
    - fails: List of failed edges to be highlighted (optional).
    - current_edge: Current edge to be highlighted (optional).
    """
    pos = {node: tree.nodes[node]['pos'] for node in tree.nodes}  # Positions of the nodes

    plt.figure(figsize=(10, 8))

    # Draw all edges in gray
    nx.draw_networkx_edges(tree, pos, edge_color='gray')

    # Draw failed edges in red, if present
    if fails:
        failed_edges = [(u, v) for u, v in fails if tree.has_edge(u, v)]
        nx.draw_networkx_edges(tree, pos, edgelist=failed_edges, edge_color='red', width=2)
        #print(f"Highlighted edges (Fails): {fails}")

    # Highlight current edge in blue, if present
    if current_edge:
        if tree.has_edge(*current_edge):
            nx.draw_networkx_edges(tree, pos, edgelist=[current_edge], edge_color='blue', width=2)
            #print(f"Current edge highlighted: {current_edge}")

    # Draw all nodes
    nx.draw_networkx_nodes(tree, pos, node_color='lightgray', node_size=500)
    nx.draw_networkx_labels(tree, pos)

    # Highlight special nodes in orange, if present
    if nodes:
        nx.draw_networkx_nodes(tree, pos, nodelist=nodes, node_color="orange", node_size=700)
        #print(f"Highlighted nodes: {nodes}")

    #plt.title("Tree with highlighted nodes, edges, and current edge")
    plt.show()

def plot_paths_element(paths_element, tree, source, destination):
    """
    Plots a graph based on a single paths[source][destination] element, with node positions.

    Parameters:
        paths_element (dict): A single paths[source][destination] element containing cps and tree information.
        tree (nx.Graph): A NetworkX graph containing node positions.
    """
    # Initialize the graph
    G = nx.DiGraph()

    # Process the structure
    cps = paths_element.get('cps', [])  # Ensure 'cps' exists

    # Add control points and source/destination as special nodes
    special_nodes = {
        source: "yellow",
        destination: "green",
    }

    # Dynamically assign colors if CPS are present
    cp_colors = ["orange", "purple", "pink"]
    for idx, cp in enumerate(cps):
        if idx < len(cp_colors):
            special_nodes[cp] = cp_colors[idx]

    # Add nodes
    for cp in cps:
        G.add_node(cp)

    # Add edges from the trees
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

    # Determine node colors
    node_colors = []
    for node in G.nodes():
        if node in special_nodes:
            node_colors.append(special_nodes[node])
        else:
            node_colors.append("gray")  # All other nodes in gray

    # Get node positions from the tree
    pos = {node: tree.nodes[node]['pos'] for node in tree.nodes() if node in G.nodes()}

    # Draw the graph
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
    """Creates a graph where each face is a node and edges exist if faces share common edges."""
    face_graph = nx.Graph()
  
    face_edges = []  # Stores all edges of a face
   
    for i, face in enumerate(faces):
        face_graph.add_node(i)  # Add face as a node
        edges = {tuple(sorted((face[j], face[(j + 1) % len(face)]))) for j in range(len(face))}
        face_edges.append(edges)
    # Connect faces that share common edges
    for i in range(len(faces)):
        for j in range(i + 1, len(faces)):
             if face_edges[i] & face_edges[j]:  # If common edges exist
                face_graph.add_edge(i, j)
   
    return face_graph

def greedy_coloring(graph):
    """Assigns colors to the nodes (faces) in the graph such that no adjacent faces have the same color."""
    colors = {}
    sorted_nodes = sorted(graph.nodes(), key=lambda x: graph.degree(x), reverse=True)  # Welsh-Powell heuristic
    
    available_colors = ["red", "blue", "green", "purple", "orange", "cyan", "pink", "yellow", "brown"]
    
    for node in sorted_nodes:
        neighbor_colors = {colors[neighbor] for neighbor in graph.neighbors(node) if neighbor in colors}
        for color in available_colors:
            if color not in neighbor_colors:
                colors[node] = color
                break
    
    return colors

def print_faces(G, faces):
    """Prints the faces with their nodes in the terminal."""
    
    print("\n===== Found Faces =====")
    
    for i, face in enumerate(faces):
        if len(face) < 3:
            continue  # Skip invalid faces
        
        # Sort the nodes for better readability
        sorted_face = sorted(face)
        
        # If node coordinates should also be printed:
        coords = [G.nodes[node]['pos'] for node in sorted_face if 'pos' in G.nodes[node]]
        
        print(f"Face {i+1}: {sorted_face}")
        # If you also want the coordinates, uncomment the next line:
        # print(f"   -> Coordinates: {coords}")

    print("==========================\n")

def draw_graph_with_colored_faces(G, faces, source, destination):
    """Draws the graph and fills the faces with different colors (no two adjacent faces have the same color)."""
    print("\n===== Drawing Graph with Colored Faces =====")
    print(f"source: {source}, destination: {destination}")
    print(f"G.nodes[source]['pos']: {G.nodes[source]['pos']}")
    print(f"G.nodes[destination]['pos']: {G.nodes[destination]['pos']}")
    # Retrieve node positions from the attributes of G
    pos = {node: G.nodes[node]['pos'] for node in G.nodes if 'pos' in G.nodes[node]}

    plt.figure(figsize=(12, 6))

    # Draw all nodes and edges
    nx.draw(G, pos, with_labels=True, node_color="lightgray", edge_color="gray")

    # Create the face graph and assign colors
    face_graph = build_face_graph(faces)
    face_colors = greedy_coloring(face_graph)

    patches = []
    
    # Draw each face as a filled polygon with the assigned color
    for i, face in enumerate(faces):
        if len(face) < 3:
            continue  # Not a valid polygon
        
        # Check if all nodes have a position
        if not all(node in pos for node in face):
            continue  # Skip this face if a position is missing
        
        polygon_coords = np.array([pos[node] for node in face])
        polygon = Polygon(polygon_coords, closed=True, edgecolor="black", facecolor=face_colors[i], alpha=0.6)
        patches.append(polygon)

    # Draw the faces
    p = PatchCollection(patches, match_original=True)
    plt.gca().add_collection(p)

    # Highlight source and destination separately
    nx.draw_networkx_nodes(G, pos, nodelist=[source], node_color="yellow", node_size=500, label="Source")
    nx.draw_networkx_nodes(G, pos, nodelist=[destination], node_color="red", node_size=500, label="Destination")

    plt.title("Graph with Colored Faces (no two adjacent faces have the same color)")
    plt.show()

def draw_graph(tree, source, destination, graph):
    plt.figure(figsize=(8, 6))
    
    # Retrieve stored positions from the original graph
    pos = {node: graph.nodes[node]['pos'] for node in graph.nodes if 'pos' in graph.nodes[node]}
    if not pos:
        pos = nx.spring_layout(tree)
    
    # Draw all edges in black
    nx.draw(tree, pos, with_labels=True, node_color='lightblue', edge_color='black', font_weight='bold')
    nx.draw_networkx_nodes(tree, pos, nodelist=[source], node_color='yellow', node_size=500)
    nx.draw_networkx_nodes(tree, pos, nodelist=[destination], node_color='yellow', node_size=500)
    
    plt.show()

def should_debug(source, destination):

    return False
    return source == 49 and destination == 34

def draw_graph_with_highlighted_edge2(tree, source, destination, edge_list, current_edge):
    """Draws the graph with special colors for source, destination, paths, and the currently added edge."""
    
    pos = nx.get_node_attributes(tree, 'pos')
    # Adding the reverse edges to the tree
    for edge in tree.edges:
        if (edge[1], edge[0]) not in tree.edges:
            tree.add_edge(edge[1], edge[0])

    if not pos:
        print("WARNING: No position attributes found! Graph might not be displayed correctly.")
    
    plt.figure(figsize=(10, 8))
    
    # Draw all edges in gray by default
    nx.draw(tree, pos, with_labels=True, edge_color='gray', node_color='lightgray', node_size=500, font_size=10)
    
    # Draw source and destination in special colors
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
    
    # Draw the current edge in blue
    if current_edge:
        nx.draw_networkx_edges(tree, pos, edgelist=[current_edge], edge_color='blue', width=3, alpha=1.0)
    
    plt.legend()
    plt.show()
