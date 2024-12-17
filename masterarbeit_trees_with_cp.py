from platform import node
import sys
import time
from traceback import print_list
import traceback
from typing import List, Any, Union
import random
from matplotlib.patches import Patch
import networkx as nx
from networkx import PlanarEmbedding
import numpy as np
import itertools
from itertools import combinations, permutations
from arborescences import *
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import networkx as nx
import traceback

#################################################### MULTIPLETREES WITH MIDDLE CHECKPOINT ################################################

##########################################################################################################################################
removed_edges_multtrees = []

def multiple_trees_with_middle_checkpoint_pre(graph):
    paths = {}
    count = 1
    all_graph_edge_number = len(graph.edges)
    all_tree_edge_number = 0
    for source in graph.nodes:
       
        for destination in graph.nodes:
            
            if source != destination:
                
                edps = all_edps(source, destination, graph) #Bildung der EDPs
                
                edps.sort(key=len, reverse=True) #Sortierung der EDPs
                
                longest_edp = edps[len(edps)-1]

                cp = longest_edp[ int(len(longest_edp)/2)]

                edps_cp_to_s = all_edps(cp, source, graph)
                
                edps_cp_to_d = all_edps(cp, destination, graph)
                
                edps_cp_to_s.sort(key=len)
                
                edps_cp_to_d.sort(key=len)

                
                            
                
                #first build multiple trees cp->s and unite them in a face structure
                
                trees_cp_to_s = multiple_trees_with_checkpoint(cp,source,graph,edps_cp_to_s).copy()

                trees_cp_to_s = remove_single_node_trees(trees_cp_to_s)#EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben,
                                                        #führen nicht zum Ziel und müssen gelöscht werden
                
                input("[MultipleTreesRandomCheckpoint] Trees CP->S gebaut")

                trees_cp_to_s = convert_to_undirected_multiple_trees(trees_cp_to_s)

                faces_cp_to_s = find_faces_multiple_trees(trees_cp_to_s)
                
                #then build multiple trees cp->d
                
                trees_cp_to_d = multiple_trees_with_checkpoint(cp,destination,graph,edps_cp_to_d).copy()
                
                trees_cp_to_d = remove_single_node_trees(trees_cp_to_s)#EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben,
                                                        #führen nicht zum Ziel und müssen gelöscht werden
                
                # i= 0
                # for tree in trees:    
                #     i = i + 1
                
                # #print(" ")
                # edges_of_this_run = 0 
                # for tree in trees:
                #     all_tree_edge_number = all_tree_edge_number + len(tree.edges)
                #     edges_of_this_run = edges_of_this_run + len(tree.edges)
                # count = count + 1
                #print("Die Kanten dieses Laufs (normal) : " , edges_of_this_run)
                #print(" ")
                if source in paths:
                    paths[source][destination] = { 
                                                    'cp': cp,
                                                'faces_cp_to_s': faces_cp_to_s, 
                                                'edps_cp_to_s': edps_cp_to_s,
                                                'tree_cp_to_d': trees_cp_to_d, 
                                                'edps_cp_to_d': edps_cp_to_d,
                                                'edps_s_to_d': edps
                                                  }
                else:
                    paths[source] = {}
                    paths[source][destination] = {
                                                'cp': cp,
                                                'faces_cp_to_s': faces_cp_to_s, 
                                                'edps_cp_to_s': edps_cp_to_s,
                                                'tree_cp_to_d': trees_cp_to_d, 
                                                'edps_cp_to_d': edps_cp_to_d,
                                                'edps_s_to_d': edps
                    }

    #print("Bei einem count von " , count , " und insgesamt Graph Kanten " , all_graph_edge_number, " ergeben sich " , all_tree_edge_number , " Baumkanten bei der normalen Variante")
    #print("Normal durchschnittliche Truncation : ", (sum(removed_edges_multtrees)/(len(removed_edges_multtrees))))
    #input("...")           
    return paths

#gibt für ein source-destination paar alle trees zurück
def multiple_trees_with_checkpoint(source, destination, graph, all_edps):
    removed_edges = 0
    trees = [] 

    #für jeden tree muss hier sein edp eingefügt werden in den jeweiligen graph des trees 

    for i in range(0,len(all_edps)):

        current_edp = all_edps[i]

        tree = nx.DiGraph()
        tree.add_node(source)
        for j in range(1,len(current_edp)-1):
            tree.add_node(current_edp[j])
            tree.add_edge(current_edp[j-1], current_edp[j])
        #endfor
        trees.append(tree)
    #endfor

    assert len(trees) == len(all_edps), 'Not every edp got a tree!'

    for i in range(0,len(all_edps)): #jeden edp einmal durchgehen
                                      #um zu versuchen aus jedem edp einen Baum zu bauen
        
        tree = trees[i] # Baum der zuvor mit dem edp gefüllt wurde
        pathToExtend = all_edps[i]

        nodes = pathToExtend[:len(pathToExtend) -1]#in nodes stehen dann alle knoten drin die wir besuchen wollen um deren nachbarn auch reinzupacken
                                                   # am anfang ganzer edp drin und -2 damit die destination nicht mit drin steht
        
        for j in range(0,len(pathToExtend)-1): #alle knoten aus nodes[] durchgehen und deren nachbarn suchen, angefangen mit den knoten aus dem edp
            
                       
            it = 0
            while it < len(nodes):
                
                neighbors = list(nx.neighbors(graph, nodes[it])) #für jeden knoten aus nodes die nachbarn finden und gucken ob sie in den tree eingefügt werden dürfen

                for k in range(0,len(neighbors)): #jeden der nachbarn durchgehen

                    if(neighbors[k] != nodes[it] and neighbors[k] != destination): #kanten zu sich selbst dürfen nicht rein da dann baum zu kreis wird und kanten zur destination auch nicht    
 

                        #prüfen ob kante von nodes[it] nach neighbors[k] schon in anderen trees verbaut ist
                        is_in_other_tree = False
                        if(len(trees)>0):#wenn es schon andere trees gibt muss man alle anderen durchsuchen
                            for tree_to_check in trees: 
                               
                                if (tree_to_check.has_edge(nodes[it],neighbors[k]) or tree_to_check.has_edge(neighbors[k],nodes[it])): #wenn ein tree die edge schon drin hat dann darf man die edge nicht mehr benutzen
                                    is_in_other_tree = True
                                    break
                                #endif
                            #endfor
                        
                            if not ( is_in_other_tree or (tree.has_node(neighbors[k])) ):#wenn die kante weder in einem anderen baum noch in diesen baum drin ist, dann füge sie ein
                                
                                nodes.append(neighbors[k]) 
                                tree.add_node(neighbors[k])
                                tree.add_edge(nodes[it],neighbors[k])
                            #endif
                        #endif

                        else: #das ist der fall wenn es noch keine anderen trees zum checken gibt, ob die kante schon verbaut ist
                              #dann darf die kante nicht zur destination sein & der knoten darf nicht im jetzigen tree drin sein
                                                                                                    
                            if not((neighbors[k] == destination) or (tree.has_node(neighbors[k]))): 
                                tree.add_node(neighbors[k])
                                tree.add_edge(nodes[it],neighbors[k])
                            #endif

                            #wenn der node der grad in den tree eingefügt wurde schon in nodes war dann soll er nicht nochmal eingefügt werden
                            if not (neighbors[k]in nodes): #damit knoten nicht doppelt in nodes eingefügt werden
                                nodes.append(neighbors[k]) 
                            #endif
                        #endelse
                    #endif
                #endfor
                it = it + 1                
            #endwhile
        #endfor

        changed = True 

        
        old_edges = len(tree.edges)
        
        while changed == True: #solange versuchen zu kürzen bis nicht mehr gekürzt werden kann 
            old_tree = tree.copy()
            remove_redundant_paths(source, destination, tree, graph) 
            changed = tree.order() != old_tree.order() # order gibt die Anzahl an Knoten zurück
        #endwhile
        
        new_edges = len(tree.edges)
        
        removed_edges =  removed_edges + (old_edges - new_edges)
        

        #man muss prüfen ob nur die source im baum ist , da man im nächsten schritt der destination einen Rang geben muss
        #nur die source im baum (tree.order == 1) bedeutet, dass es im graphen die kante source -> destination gibt
        if( tree.order() > 1 ):
            

            rank_tree(tree , source,all_edps[i])
            connect_leaf_to_destination(tree, source,destination)

            tree.add_edge(all_edps[i][len(all_edps[i])-2],destination)
            
            tree.nodes[destination]["rank"] = -1
        #endif
        
        #edps direkt von s->d kommen müssen gesondert betrachtet werden
        if(tree.order() == 1 and len(all_edps[i]) == 2):
            tree.add_edge(source,destination)
            tree.nodes[destination]["rank"] = -1
        #endif
        
           
    #endfor

    removed_edges_multtrees.append(removed_edges)
    
    return trees


def find_faces_multiple_trees(trees_cp_to_s):
    faces = []
    
    return faces

def convert_to_undirected_multiple_trees(trees_cp_to_s):
    trees = []
    for tree in trees_cp_to_s:
        trees.append(tree.copy().to_undirected())
    return trees

#################################################### ONETREE WITH MIDDLE CHECKPOINT ######################################################

##########################################################################################################################################

#function to generate the 2 trees for each s->d pair (s->cp & cp->d)
#each tree gets generated by expanding the longest edp of each pair
from trees import all_edps, connect_leaf_to_destination, rank_tree, remove_redundant_paths, remove_single_node_trees


def one_tree_with_middle_checkpoint_pre(graph):
    debug = False
    paths = {}
    
    for source in graph.nodes:
        #print("[OTC Random Pre] check")
        for destination in graph.nodes:
            
            if source != destination:
                
                if source not in paths:
                    paths[source] = {}
                
                #now compute the chosen checkpoint  
                #first get the longest edp s->d    
                edps = all_edps(source, destination, graph)
                
                edps.sort(key=len)
                
                longest_edp = edps[len(edps)-1]
                
                #special case if the s,d pair is connected and this is the only edp
                if(len(longest_edp) == 2):
                    #print("Special case for : ", source, "-", destination)
                    paths[source][destination] = {
                                                'cp': destination,
                                                'faces_cp_to_s': [], 
                                                'edps_cp_to_s': [[source,destination]],
                                                'tree_cp_to_d':[], 
                                                'edps_cp_to_d': [[source,destination]],
                                                'edps_s_to_d':[[source,destination]]
                                            }
                    continue
                
                #then select the middle node of the longest_edp
               #p("[OneTreeRandomCheckpoint] longestEDP: ", longest_edp)
                cp = longest_edp[ int(len(longest_edp)/2)]
               #p("[OneTreeRandomCheckpoint] Checkpoint: ",cp)
                #then get the edps + longest_edps_cp_s and the longest_edps_cp_d
                
                edps_cp_to_s = all_edps(cp, source, graph)
                
                edps_cp_to_d = all_edps(cp, destination, graph)
                
                edps_cp_to_s.sort(key=len)
                
                edps_cp_to_d.sort(key=len)
                
                #print(" ")
                
                #print("(CP PRE, MIDDLE 2) EDPS CP -> D for : (", source ,",", cp ,"," , destination , ") : ", edps_cp_to_d)   
                
                #and build trees out of the longest_edps_cp_s and the longest_edps_cp_d
                
                #faces_cp_to_s, tree_planar_embedding_cp_to_s, tree_cp_to_s = one_tree_with_checkpoint(cp,source,graph,edps_cp_to_s[len(edps_cp_to_s)-1], True)
                faces_cp_to_s, tree_cp_to_s = one_tree_with_checkpoint(cp,source,graph,edps_cp_to_s[len(edps_cp_to_s)-1], True)

                #draw_tree_with_highlighted_nodes(tree_cp_to_s,[source,cp])
                
                #print("[OneTreeCheckpoint] Typ von tree_planar_embedding:", type(tree_planar_embedding_cp_to_s))
                #print("[OneTreeCheckpoint] Typ von graph:", type(tree_cp_to_s))

                #neighbors = list(tree_cp_to_s.neighbors(source))
                #input(f"[OTCP Pre] Untersuche die Nachbarn: {neighbors}")

                tree_cp_to_d = one_tree_with_checkpoint(cp,destination,graph,edps_cp_to_d[len(edps_cp_to_d)-1], False).copy()
                
                #bc the tree cp->s got build reverse direction the edges need to be reversed again
                #data structure to give the needed information for the routing (edps, trees, checkpoint)
                
                paths[source][destination] = {
                                                'cp': cp,
                                                'faces_cp_to_s': faces_cp_to_s, 
                                                'edps_cp_to_s': edps_cp_to_s,
                                                'tree_cp_to_d': tree_cp_to_d, 
                                                'edps_cp_to_d': edps_cp_to_d,
                                                'edps_s_to_d': edps,
                                                #'tree_planar_embedding_cp_to_s':tree_planar_embedding_cp_to_s,
                                                'tree_cp_to_s':tree_cp_to_s
                                            }
                                    
    return paths


#this algorithm builds a tree for the one_tree_with_checpoint function
#the tree has the source as root of the tree and every leaf is connected with the destination at the end
#the tree is build by expanding the longest edp as much as possible and only keeping the paths that lead to the destination

#special: because the second tree that is required to build by the one_tree_with_random_checkpoint_pre is the tree cp->s
#its directed edges need to flipped (arg: reverse)
def one_tree_with_checkpoint(source, destination, graph, longest_edp, reverse):
    
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
    

    #print("[OneTreeCheckpoint] Plotting before truncation Tree...")
    #pos = {node: tree.nodes[node]['pos'] for node in tree.nodes if 'pos' in tree.nodes[node]}  # Positionen für den Tree

    # plt.figure(figsize=(8, 8))
    # nx.draw(tree, pos, with_labels=True, node_color='lightblue', edge_color='orange')
    # plt.title("Tree before Truncation Tree")
    # plt.show()

    changed = True 
    

    while changed == True: #keep trying to shorten until no more can be shortened 
        
        old_tree = tree.copy()
        remove_redundant_paths(source, destination, tree, graph)
        changed = tree.order() != old_tree.order() # order returns the number of nodes in the graph.
    
    #print("[OneTreeCheckpoint] Plotting after truncation Tree...")
    #pos = {node: tree.nodes[node]['pos'] for node in tree.nodes if 'pos' in tree.nodes[node]}  # Positionen für den Tree

    # plt.figure(figsize=(8, 8))
    # nx.draw(tree, pos, with_labels=True, node_color='lightblue', edge_color='blue')
    # plt.title("Tree after Truncation Tree")
    # plt.show()

    #before ranking the tree, if the the is build for cp->s the edges need to be flipped
    if(reverse):

        #the current tree has:
        # source = cp (from the global graph)
        # destination = source (from the global graph)

        connect_leaf_to_destination(tree,source,destination)

        #in order to find and traverse faces the tree need to be an undirected graph
        undirected_tree = tree.to_undirected()

        tree_planar_embedding = PlanarEmbedding()

        for node in graph.nodes:
            if node in undirected_tree.nodes:
                undirected_tree.nodes[node]['pos']=graph.nodes[node]['pos']
                tree_planar_embedding.add_node(node)
                tree_planar_embedding.nodes[node]['pos']=graph.nodes[node]['pos']

        # Kopiere die Kanten aus dem tree
        for u, v in tree.edges:
            tree_planar_embedding.add_edge(u, v)

        faces_graph = find_faces(graph)

        faces = find_faces(graph) #ich hole mir alle faces des graphen

        tree_nodes = set(tree.nodes)

        # faces_graph ist eine Liste der Faces
        last_face = faces_graph[-1]
        filtered_faces = [
            face for face in faces_graph[:-1]  # Alle Faces außer dem letzten
            if set(face.nodes).issubset(tree_nodes)
        ]

        filtered_faces.append(last_face) # dann schmeisse ich alle faces raus die nicht nur aus knoten des baums bestehen

        #neighbors = list(undirected_tree.neighbors(source))

        #input(f"[OTCP] Untersuche die Nachbarn: {neighbors}")

        #return faces, tree_planar_embedding, undirected_tree
        return faces, undirected_tree




    else: #if the tree build is for cp->d nothing is changed
    
        rank_tree(tree , source,longest_edp)
    
        connect_leaf_to_destination(tree, source, destination)
    
        tree.add_edge(longest_edp[len(longest_edp)-2],destination)
        
        #add 'rank' property to the added destinaton, -1 for highest priority in routing
        tree.nodes[destination]["rank"] = -1
        
        return tree    
        
    #end if
    
########################################################################################################

#################################################### ONETREE WITH DEGREE CHECKPOINT ######################################################

##########################################################################################################################################


def one_tree_with_degree_checkpoint_pre(graph):
    debug = False
    paths = {}
    
    for source in graph.nodes:
        #print("[OTC Degree Pre] check")
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
                if len(longest_edp) == 2:
                    paths[source][destination] = {
                        'cp': destination,
                        'faces_cp_to_s': [], 
                        'edps_cp_to_s': [[source, destination]],
                        'tree_cp_to_d': [], 
                        'edps_cp_to_d': [[source, destination]],
                        'edps_s_to_d': [[source, destination]]
                    }
                    continue
                
                # Calculate Degree Centrality for nodes in the graph
                degree_centrality = nx.degree_centrality(graph)
                #print("[OneTreeDegreeCheckpoint] longestEDP:", longest_edp)
                ##print the Degree Centrality of all nodes in the longest EDP
                #print(f"[OneTreeDegreeCheckpoint] Degree Centralities for longest EDP (Source: {source}, Destination: {destination}):")
                #for node in longest_edp:
                #   #print(f"Node {node}: Centrality {degree_centrality[node]:.4f}")
                
                # Filter out source and destination from the longest EDP
                filtered_edp = [node for node in longest_edp if node != source and node != destination]
                
                # Handle the case where no valid cp is available after filtering
                if not filtered_edp:
                   #print(f"[OneTreeDegreeCheckpoint] No valid checkpoint for Source: {source}, Destination: {destination}")
                    paths[source][destination] = {
                        'cp': None,
                        'faces_cp_to_s': [],
                        'edps_cp_to_s': [],
                        'tree_cp_to_d': [],
                        'edps_cp_to_d': [],
                        'edps_s_to_d': edps
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
                faces_cp_to_s = one_tree_with_checkpoint(
                    cp, source, graph, edps_cp_to_s[-1], True
                ).copy()
                
                tree_cp_to_d = one_tree_with_checkpoint(
                    cp, destination, graph, edps_cp_to_d[-1], False
                ).copy()
                
                # Store the result in the paths dictionary
                paths[source][destination] = {
                    'cp': cp,
                    'faces_cp_to_s': faces_cp_to_s, 
                    'edps_cp_to_s': edps_cp_to_s,
                    'tree_cp_to_d': tree_cp_to_d, 
                    'edps_cp_to_d': edps_cp_to_d,
                    'edps_s_to_d': edps
                }
                                    
    return paths



######################################################################################################################################################

#################################################### ONETREE WITH BETWEENNESS CHECKPOINT ######################################################

##########################################################################################################################################


def one_tree_with_betweenness_checkpoint_pre(graph):
    debug = False
    paths = {}
    
    for source in graph.nodes:
        #print("[OTC Betweenness Pre] check")
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
                if len(longest_edp) == 2:
                    paths[source][destination] = {
                        'cp': destination,
                        'faces_cp_to_s': [], 
                        'edps_cp_to_s': [[source, destination]],
                        'tree_cp_to_d': [], 
                        'edps_cp_to_d': [[source, destination]],
                        'edps_s_to_d': [[source, destination]]
                    }
                    continue
                
                # Calculate Betweenness Centrality for nodes in the graph
                betweenness_centrality = nx.betweenness_centrality(graph, normalized=True)
               #print("[OneTreeBetweennessCheckpoint] longestEDP:", longest_edp)
                
                ##print the Betweenness Centrality of all nodes in the longest EDP
                #print(f"[OneTreeBetweennessCheckpoint] Betweenness Centralities for longest EDP (Source: {source}, Destination: {destination}):")
                #for node in longest_edp:
                #   #print(f"Node {node}: Centrality {betweenness_centrality[node]:.4f}")
                
                # Filter out source and destination from the longest EDP
                filtered_edp = [node for node in longest_edp if node != source and node != destination]
                
                # Handle the case where no valid cp is available after filtering
                if not filtered_edp:
                   #print(f"[OneTreeBetweennessCheckpoint] No valid checkpoint for Source: {source}, Destination: {destination}")
                    paths[source][destination] = {
                        'cp': None,
                        'faces_cp_to_s': [],
                        'edps_cp_to_s': [],
                        'tree_cp_to_d': [],
                        'edps_cp_to_d': [],
                        'edps_s_to_d': edps
                    }
                    continue
                
                # Select the node with the highest Betweenness Centrality in the filtered EDP as the checkpoint
                cp = max(filtered_edp, key=lambda node: betweenness_centrality[node])
               #print(f"[OneTreeBetweennessCheckpoint] Selected Checkpoint (cp): Node {cp} with Centrality {betweenness_centrality[cp]:.4f}\n")
                
                # Get EDPs from the checkpoint to the source and destination
                edps_cp_to_s = all_edps(cp, source, graph)
                edps_cp_to_d = all_edps(cp, destination, graph)
                
                edps_cp_to_s.sort(key=len)
                edps_cp_to_d.sort(key=len)
                
                # Build trees and faces
                faces_cp_to_s = one_tree_with_checkpoint(
                    cp, source, graph, edps_cp_to_s[-1], True
                ).copy()
                
                tree_cp_to_d = one_tree_with_checkpoint(
                    cp, destination, graph, edps_cp_to_d[-1], False
                ).copy()
                
                # Store the result in the paths dictionary
                paths[source][destination] = {
                    'cp': cp,
                    'faces_cp_to_s': faces_cp_to_s, 
                    'edps_cp_to_s': edps_cp_to_s,
                    'tree_cp_to_d': tree_cp_to_d, 
                    'edps_cp_to_d': edps_cp_to_d,
                    'edps_s_to_d': edps
                }
                                    
    return paths



######################################################################################################################################################
#################################################### ONETREE WITH BETWEENNESS CHECKPOINT ######################################################

##########################################################################################################################################


def one_tree_with_closeness_checkpoint_pre(graph):
    debug = False
    paths = {}
    
    for source in graph.nodes:
        #print("[OTC Closeness Pre] check")
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
                if len(longest_edp) == 2:
                    paths[source][destination] = {
                        'cp': destination,
                        'faces_cp_to_s': [], 
                        'edps_cp_to_s': [[source, destination]],
                        'tree_cp_to_d': [], 
                        'edps_cp_to_d': [[source, destination]],
                        'edps_s_to_d': [[source, destination]]
                    }
                    continue
                
                # Calculate Closeness Centrality for nodes in the graph
                closeness_centrality = nx.closeness_centrality(graph)
               #print("[OneTreeClosenessCheckpoint] longestEDP:", longest_edp)
                
                ##print the Closeness Centrality of all nodes in the longest EDP
                #print(f"[OneTreeClosenessCheckpoint] Closeness Centralities for longest EDP (Source: {source}, Destination: {destination}):")
                #for node in longest_edp:
                #   #print(f"Node {node}: Centrality {closeness_centrality[node]:.4f}")
                
                # Filter out source and destination from the longest EDP
                filtered_edp = [node for node in longest_edp if node != source and node != destination]
                
                # Handle the case where no valid cp is available after filtering
                if not filtered_edp:
                   #print(f"[OneTreeClosenessCheckpoint] No valid checkpoint for Source: {source}, Destination: {destination}")
                    paths[source][destination] = {
                        'cp': None,
                        'faces_cp_to_s': [],
                        'edps_cp_to_s': [],
                        'tree_cp_to_d': [],
                        'edps_cp_to_d': [],
                        'edps_s_to_d': edps
                    }
                    continue
                
                # Select the node with the highest Closeness Centrality in the filtered EDP as the checkpoint
                cp = max(filtered_edp, key=lambda node: closeness_centrality[node])
               #print(f"[OneTreeClosenessCheckpoint] Selected Checkpoint (cp): Node {cp} with Centrality {closeness_centrality[cp]:.4f}\n")
                
                # Get EDPs from the checkpoint to the source and destination
                edps_cp_to_s = all_edps(cp, source, graph)
                edps_cp_to_d = all_edps(cp, destination, graph)
                
                edps_cp_to_s.sort(key=len)
                edps_cp_to_d.sort(key=len)
                
                # Build trees and faces
                faces_cp_to_s = one_tree_with_checkpoint(
                    cp, source, graph, edps_cp_to_s[-1], True
                ).copy()
                
                tree_cp_to_d = one_tree_with_checkpoint(
                    cp, destination, graph, edps_cp_to_d[-1], False
                ).copy()
                
                # Store the result in the paths dictionary
                paths[source][destination] = {
                    'cp': cp,
                    'faces_cp_to_s': faces_cp_to_s, 
                    'edps_cp_to_s': edps_cp_to_s,
                    'tree_cp_to_d': tree_cp_to_d, 
                    'edps_cp_to_d': edps_cp_to_d,
                    'edps_s_to_d': edps
                }
                                    
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
                        fitting_faces_special = []

                        for face in all_faces_special:
                            if source in face.nodes() and destination in face.nodes():
                                fitting_faces_special.append(face)

                        paths[source][destination] = {
                            'cp': destination,
                            'faces_cp_to_s': fitting_faces_special,
                            'edps_cp_to_s': [[source, destination]],
                            'tree_cp_to_d': [],
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
                faces_cp_to_s = one_tree_with_checkpoint(
                    cp, source, graph, edps_cp_to_s[-1], True
                ).copy()

                tree_cp_to_d = one_tree_with_checkpoint(
                    cp, destination, graph, edps_cp_to_d[-1], False
                ).copy()

                # Store the result in the paths dictionary
                paths[source][destination] = {
                    'cp': cp,
                    'faces_cp_to_s': faces_cp_to_s,
                    'edps_cp_to_s': edps_cp_to_s,
                    'tree_cp_to_d': tree_cp_to_d,
                    'edps_cp_to_d': edps_cp_to_d,
                    'edps_s_to_d': edps
                }

    return paths



############################################################################################################################

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