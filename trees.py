from platform import node
import sys
import time
from traceback import print_list
from typing import List, Any, Union
import random
from matplotlib.patches import Patch
import networkx as nx
import numpy as np
import itertools

from arborescences import *
DEBUG = False



#die struktur von paths : 
#Für jede Kombination aus Source-Destination gibt es einen Eintrag
#paths[source][destination] = {
#                               'trees': hier Ist dann ein Array drin, welches aus weiteren Arrays besteht in denen die Trees drin stehen
#                               ,
#                               'edps': hier Ist dann ein Array drin, welches aus weiteren Arrays besteht in denen die EDPs drin stehen
#                              }


#der Algorithmus der die Baumbildung aufruft
removed_edges_multtrees = []
def multiple_trees_pre(graph):
    paths = {}
    count = 1
    all_graph_edge_number = len(graph.edges)
    all_tree_edge_number = 0
    for source in graph.nodes:
       
        for destination in graph.nodes:
            
            if source != destination:
                
                edps = all_edps(source, destination, graph) #Bildung der EDPs
                
                edps.sort(key=len, reverse=True) #Sortierung der EDPs
                
                #print("Start building trees with MultipleTrees Base for ", source , " to ", destination)
                
                trees = multiple_trees(source,destination,graph,edps)
                
                trees = remove_single_node_trees(trees)#EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben,
                                                        #führen nicht zum Ziel und müssen gelöscht werden
                i= 0
                for tree in trees:    
                    i = i + 1
                
                #print(" ")
                edges_of_this_run = 0 
                for tree in trees:
                    all_tree_edge_number = all_tree_edge_number + len(tree.edges)
                    edges_of_this_run = edges_of_this_run + len(tree.edges)
                count = count + 1
                #print("Die Kanten dieses Laufs (normal) : " , edges_of_this_run)
                #print(" ")
                if source in paths:
                    paths[source][destination] = { 'trees': trees, 'edps': edps}
                else:
                    paths[source] = {}
                    paths[source][destination] = {'trees': trees, 'edps': edps}
                
    #print("Bei einem count von " , count , " und insgesamt Graph Kanten " , all_graph_edge_number, " ergeben sich " , all_tree_edge_number , " Baumkanten bei der normalen Variante")
    #print("Normal durchschnittliche Truncation : ", (sum(removed_edges_multtrees)/(len(removed_edges_multtrees))))
    #input("...")           
    return paths

#gibt für ein source-destination paar alle trees zurück
def multiple_trees(source, destination, graph, all_edps):
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



########################################## MultTrees Änderung Reihenfolge der EDPs ###############################################################


#Funktion die für jeden des Source-Destination-Paar die Baumbildung aufruft und in paths speichert
def multiple_trees_pre_order_of_edps_mod(graph):
    paths = {}

    
    for source in graph.nodes:

        for destination in graph.nodes:
            
            if source != destination:
                
                edps = all_edps(source, destination, graph) #Bildung der EDPs
                
                edps.sort(key=len, reverse=True) #Sortierung der EDPs in absteigender Folge, dh. der längste edp ist in index 0
                
                print("Start building trees with MultipleTrees Mod Order for ", source , " to ", destination)
                trees = multiple_trees_order_of_edps_mod(source,destination,graph,edps)
                
                trees = remove_single_node_trees(trees)#EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben,
                                                        #führen nicht zum Ziel und müssen gelöscht werden
                
                print_trees(source,destination,trees)
                
                print(" ")
                if source in paths:
                    paths[source][destination] = { 'trees': trees, 'edps': edps}
                else:
                    paths[source] = {}
                    paths[source][destination] = {'trees': trees, 'edps': edps}

                
    return paths

#gibt für ein source-destination paar alle trees zurück
def multiple_trees_order_of_edps_mod(source, destination, graph, all_edps):
    trees = [] #hier werden alle trees gespeichert 


    #############################################################################
    # mode nach belieben verändern 
    #############################################################################
    mode = "random" # random / invert

   
    #reihenfolge wird zufällig gewählt
    if(mode == "random" ):
        print("Randomizinig EDPs")
        print("EDPs before shuffle : " , all_edps)
        random.shuffle(all_edps)
        print("EDPs after shuffle : " , all_edps)
    
    #reihenfolge wird invertiert
    if(mode == "invert" ):
        print("Inverting EDPs")
        print("EDPs before inverting : ", all_edps)
        all_edps.sort(key=len, reverse=False)
        print("EDPs after inverting : ", all_edps)

    #für jeden tree muss hier sein edp eingefügt werden in den jeweiligen graph des trees 
    for i in range(0,len(all_edps)):

        current_edp = all_edps[i]
        tree = nx.DiGraph()
        tree.add_node(source)
        for j in range(1,len(current_edp)-1):
            tree.add_node(current_edp[j])
            tree.add_edge(current_edp[j-1], current_edp[j])

        trees.append(tree)

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
                        #prüfen ob kante von nodes[j] nach neighbors[k] schon in anderen trees verbaut ist
                        is_in_other_tree = False
                        if(len(trees)>0):#wenn es schon andere trees gibt muss man alle anderen durchsuchen
                            for tree_to_check in trees: 
        
                                if (tree_to_check.has_edge(nodes[it],neighbors[k]) or tree_to_check.has_edge(neighbors[k],nodes[it])): #wenn ein tree die edge schon drin hat dann darf man die edge nicht mehr benutzen
                                    is_in_other_tree = True
                                    break
                                #endif
                            #endfor
                        
                            if not ( is_in_other_tree or (tree.has_node(neighbors[k])) ):

                                nodes.append(neighbors[k]) 
                                tree.add_node(neighbors[k])
                                tree.add_edge(nodes[it],neighbors[k])

                            #endif
                        #endif
                        else: #das ist der fall wenn es noch keine anderen trees zum checken gibt, ob die kante schon verbaut ist
                            if not((neighbors[k] == destination) or (tree.has_node(neighbors[k]))): #dann darf die kante nicht zur destination sein
                                                                                                    #der knoten darf nicht im jetzigen tree drin sein
                                
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

        while changed == True: #solange versuchen zu kürzen bis nicht mehr gekürzt werden kann 
            old_tree = tree.copy()
            remove_redundant_paths(source, destination, tree, graph) 
            changed = tree.order() != old_tree.order() # order returns the number of nodes in the graph.
        #endwhile

        #man muss prüfen ob nur die source im baum ist , da man im nächsten schritt der destination einen Rang geben muss
        #nur die source im baum (tree.order == 1) bedeutet, dass es im graphen die kante source -> destination gibt
        if( tree.order() > 1 ):
            rank_tree(tree , source, all_edps[i])
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
    return trees



######################################## MultTrees Reihenfolge invert  ##################################################################

#Funktion die für jeden des Source-Destination-Paar die Baumbildung aufruft und in paths speichert
def multiple_trees_pre_invert_order_of_edps_mod(graph):
    paths = {}
    count = 1
    all_graph_edge_number = len(graph.edges)
    all_tree_edge_number = 0
    
    for source in graph.nodes:

        for destination in graph.nodes:
            
            if source != destination:
                
                edps = all_edps(source, destination, graph) #Bildung der EDPs
                
                edps.sort(key=len, reverse=True) #Sortierung der EDPs in absteigender Folge, dh. der längste edp ist in index 0
                
                print("Start building trees with MultipleTrees Mod Order for ", source , " to ", destination)
                trees = multiple_trees_invert_order_of_edps_mod(source,destination,graph,edps)
                
                trees = remove_single_node_trees(trees)#EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben,
                                                        #führen nicht zum Ziel und müssen gelöscht werden
                
                print(" ")
                edges_of_this_run = 0 
                for tree in trees:
                    all_tree_edge_number = all_tree_edge_number + len(tree.edges)
                    edges_of_this_run = edges_of_this_run + len(tree.edges)
                count = count + 1
                print("Die Kanten dieses Laufs (mod) : " , edges_of_this_run)
                print(" ")

                print(" ")
                if source in paths:
                    paths[source][destination] = { 'trees': trees, 'edps': edps}
                else:
                    paths[source] = {}
                    paths[source][destination] = {'trees': trees, 'edps': edps}

    print("Bei einem count von " , count , " und insgesamt Graph Kanten " , all_graph_edge_number, " ergeben sich " , all_tree_edge_number , " Baumkanten bei der modifizierten (invert) Variante")
                
    return paths

#gibt für ein source-destination paar alle trees zurück
def multiple_trees_invert_order_of_edps_mod(source, destination, graph, all_edps):
    trees = [] #hier werden alle trees gespeichert 


    #das ist der Algorithmus für die invertierte Reihenfolge

    mode = "invert" # random / invert

   
    #reihenfolge wird zufällig gewählt
    if(mode == "random" ):
        print("Randomizinig EDPs")
        print("EDPs before shuffle : " , all_edps)
        random.shuffle(all_edps)
        print("EDPs after shuffle : " , all_edps)
    
    #reihenfolge wird invertiert
    if(mode == "invert" ):
        print("Inverting EDPs")
        print("EDPs before inverting : ", all_edps)
        all_edps.sort(key=len, reverse=False)
        print("EDPs after inverting : ", all_edps)

    #für jeden tree muss hier sein edp eingefügt werden in den jeweiligen graph des trees 
    for i in range(0,len(all_edps)):

        current_edp = all_edps[i]
        tree = nx.DiGraph()
        tree.add_node(source)
        for j in range(1,len(current_edp)-1):
            tree.add_node(current_edp[j])
            tree.add_edge(current_edp[j-1], current_edp[j])

        trees.append(tree)

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
                        #prüfen ob kante von nodes[j] nach neighbors[k] schon in anderen trees verbaut ist
                        is_in_other_tree = False
                        if(len(trees)>0):#wenn es schon andere trees gibt muss man alle anderen durchsuchen
                            for tree_to_check in trees: 
        
                                if (tree_to_check.has_edge(nodes[it],neighbors[k]) or tree_to_check.has_edge(neighbors[k],nodes[it])): #wenn ein tree die edge schon drin hat dann darf man die edge nicht mehr benutzen
                                    is_in_other_tree = True
                                    break
                                #endif
                            #endfor
                        
                            if not ( is_in_other_tree or (tree.has_node(neighbors[k])) ):

                                nodes.append(neighbors[k]) 
                                tree.add_node(neighbors[k])
                                tree.add_edge(nodes[it],neighbors[k])

                            #endif
                        #endif
                        else: #das ist der fall wenn es noch keine anderen trees zum checken gibt, ob die kante schon verbaut ist
                            if not((neighbors[k] == destination) or (tree.has_node(neighbors[k]))): #dann darf die kante nicht zur destination sein
                                                                                                    #der knoten darf nicht im jetzigen tree drin sein
                                
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

        while changed == True: #solange versuchen zu kürzen bis nicht mehr gekürzt werden kann 
            old_tree = tree.copy()
            remove_redundant_paths(source, destination, tree, graph) 
            changed = tree.order() != old_tree.order() # order returns the number of nodes in the graph.
        #endwhile

        #man muss prüfen ob nur die source im baum ist , da man im nächsten schritt der destination einen Rang geben muss
        if( tree.order() > 1 ):
            rank_tree(tree , source, all_edps[i])
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
    return trees

######################################### MultTrees Reihenfolge random ##################################################################

#Funktion die für jeden des Source-Destination-Paar die Baumbildung aufruft und in paths speichert
def multiple_trees_pre_random_order_of_edps_mod(graph):
    paths = {}

    #PG = nx.nx_pydot.write_dot(graph, "./multiple_trees_graphen/graph")
    
    for source in graph.nodes:

        for destination in graph.nodes:
            
            if source != destination:
                
                edps = all_edps(source, destination, graph) #Bildung der EDPs
                
                edps.sort(key=len, reverse=True) #Sortierung der EDPs in absteigender Folge, dh. der längste edp ist in index 0
                
                print("Start building trees with MultipleTrees Mod Order for ", source , " to ", destination)
                trees = multiple_trees_random_order_of_edps_mod(source,destination,graph,edps)
                
                trees = remove_single_node_trees(trees)#EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben,
                                                        #führen nicht zum Ziel und müssen gelöscht werden
                
                print_trees(source,destination,trees)
                
                print(" ")
                if source in paths:
                    paths[source][destination] = { 'trees': trees, 'edps': edps}
                else:
                    paths[source] = {}
                    paths[source][destination] = {'trees': trees, 'edps': edps}

                
    return paths

#gibt für ein source-destination paar alle trees zurück
def multiple_trees_random_order_of_edps_mod(source, destination, graph, all_edps):
    trees = [] #hier werden alle trees gespeichert 


    mode = "random" # random / invert

   
    #reihenfolge wird zufällig gewählt
    if(mode == "random" ):
        print("Randomizinig EDPs")
        print("EDPs before shuffle : " , all_edps)
        random.shuffle(all_edps)
        print("EDPs after shuffle : " , all_edps)
    
    #reihenfolge wird invertiert
    if(mode == "invert" ):
        print("Inverting EDPs")
        print("EDPs before inverting : ", all_edps)
        all_edps.sort(key=len, reverse=False)
        print("EDPs after inverting : ", all_edps)

    #für jeden tree muss hier sein edp eingefügt werden in den jeweiligen graph des trees 
    for i in range(0,len(all_edps)):

        current_edp = all_edps[i]
        tree = nx.DiGraph()
        tree.add_node(source)
        for j in range(1,len(current_edp)-1):
            tree.add_node(current_edp[j])
            tree.add_edge(current_edp[j-1], current_edp[j])

        trees.append(tree)

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
                        #prüfen ob kante von nodes[j] nach neighbors[k] schon in anderen trees verbaut ist
                        is_in_other_tree = False
                        if(len(trees)>0):#wenn es schon andere trees gibt muss man alle anderen durchsuchen
                            for tree_to_check in trees: 
        
                                if (tree_to_check.has_edge(nodes[it],neighbors[k]) or tree_to_check.has_edge(neighbors[k], nodes[it]) ): #wenn ein tree die edge schon drin hat dann darf man die edge nicht mehr benutzen
                                    is_in_other_tree = True
                                    break
                                #endif
                            #endfor
                        
                            if not ( is_in_other_tree or (tree.has_node(neighbors[k])) ):

                                nodes.append(neighbors[k]) 
                                tree.add_node(neighbors[k])
                                tree.add_edge(nodes[it],neighbors[k])

                            #endif
                        #endif
                        else: #das ist der fall wenn es noch keine anderen trees zum checken gibt, ob die kante schon verbaut ist
                            if not((neighbors[k] == destination) or (tree.has_node(neighbors[k]))): #dann darf die kante nicht zur destination sein
                                                                                                    #der knoten darf nicht im jetzigen tree drin sein
                                
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

        while changed == True: #solange versuchen zu kürzen bis nicht mehr gekürzt werden kann 
            old_tree = tree.copy()
            remove_redundant_paths(source, destination, tree, graph) 
            changed = tree.order() != old_tree.order() # order returns the number of nodes in the graph.
        #endwhile

        #man muss prüfen ob nur die source im baum ist , da man im nächsten schritt der destination einen Rang geben muss
        if( tree.order() > 1 ):
            rank_tree(tree , source, all_edps[i])
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
    return trees


########################################## MultTrees Änderung Anzahl Bäume ###############################################################


#Funktion die für jeden des Source-Destination-Paar die Baumbildung aufruft und in paths speichert
def multiple_trees_pre_num_of_trees_mod(graph):
    paths = {}
    count = 1
    all_graph_edge_number = len(graph.edges)
    all_tree_edge_number = 0
    for source in graph.nodes:

        for destination in graph.nodes:
            
            if source != destination:
                
                edps = all_edps(source, destination, graph) #Bildung der EDPs
                
                edps.sort(key=len, reverse=True) #Sortierung der EDPs
                
                #print("Start building trees for ", source , " to ", destination)
                trees = multiple_trees_num_of_trees_mod(source,destination,graph,edps)
                
                trees = remove_single_node_trees(trees)#EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben,
                                                        #führen nicht zum Ziel und müssen gelöscht werden
                
                
                
                edges_of_this_run = 0 
                for tree in trees:
                    all_tree_edge_number = all_tree_edge_number + len(tree.edges)
                    edges_of_this_run = edges_of_this_run + len(tree.edges)
                count = count + 1
                print("Die Kanten dieses Laufs (modifiziert) : " , edges_of_this_run)
                print(" ")
                if source in paths:
                    paths[source][destination] = { 'trees': trees, 'edps': edps}
                else:
                    paths[source] = {}
                    paths[source][destination] = {'trees': trees, 'edps': edps}
    print("Bei einem count von " , count , " und insgesamt Graph Kanten " , all_graph_edge_number, " ergeben sich " , all_tree_edge_number , " Baumkanten bei der modifizierten (num of trees) Variante")
                
    return paths

#gibt für ein source-destination paar alle trees zurück
def multiple_trees_num_of_trees_mod(source, destination, graph, all_edps):
    trees = [] #hier werden alle trees gespeichert 
    debug = False

    ###################################################################################################################

    number_of_wanted_trees = 3 #diese Zahl muss geändert werden, damit man die Anzahl an zu bauenden Bäumen einschränkt
    
    ###################################################################################################################
    number_of_edps = len(all_edps)
    
    print("Versuche " , number_of_wanted_trees , " aus " , number_of_edps , " zu bilden ")


    #dann so (zufällige Zahl)-viele Elemente aus dem all_edps[] in einer subliste speichern 
    #dann die all_edps = subliste setzen
    sublist = []
    try:
        indexes_for_sublist = random.sample(range(0, number_of_edps), number_of_wanted_trees) # auswgewählt-viele zufällige Zahlen zwischen 1 - Anzahl an EDPs

        print("Die zufällig gewählten Indezes : ", indexes_for_sublist)

        for i in range(0, len(indexes_for_sublist)): #die zufällig gewählten edps in die sublist einfügen
            index = indexes_for_sublist[i]
            sublist.append(all_edps[index])  
        #endfor
        print("all_edps vor der Änderung : " , all_edps)
        all_edps = sublist
        print("all_edps nach der Änderung : " , all_edps)
    #endtry
    except ValueError:
        print('Zu viele EDPs ausgewählt, es werden alle EDPs genutzt') #wenn man versucht zu viele edps zu wählen 
    #endexcept

    #für jeden tree muss hier sein edp eingefügt werden in den jeweiligen graph des trees 
    for i in range(0,len(all_edps)):

        current_edp = all_edps[i]
        #print("Current EDP : ", current_edp)
        tree = nx.DiGraph()
        tree.add_node(source)
        for j in range(1,len(current_edp)-1):
            tree.add_node(current_edp[j])
            tree.add_edge(current_edp[j-1], current_edp[j])
        #endfor
        trees.append(tree)
    #endfor
    
    if(source == 1 and destination == 21 and debug == True):
        print("Hab passendes Paar gefunden")
        i = 0
        for tree in trees:
        #    PG = nx.nx_pydot.write_dot(tree , "./graphen/custom_onlyEdp"+ str(source) + "_" +  str(destination) + "_" + str(i))
            i = i + 1
        #endfor
    #endif

    assert len(trees) == len(all_edps), 'Not every edp got a tree!'

    for i in range(0,len(all_edps)): #jeden edp einmal durchgehen
                                      #um zu versuchen aus jedem edp einen Baum zu bauen

        tree = trees[i] # Baum der zuvor mit dem edp gefüllt wurde
        pathToExtend = all_edps[i]

        nodes = pathToExtend[:len(pathToExtend) -1]#in nodes stehen dann alle knoten drin die wir besuchen wollen um deren nachbarn auch reinzupacken
                                                   # am anfang ganzer edp drin und -2 damit die destination nicht mit drin steht
        
        if(source == 1 and destination == 21 and debug == True ):
            print("Building Tree for EDP : ", all_edps)
        #endif

        for j in range(0,len(pathToExtend)-1): #alle knoten aus nodes[] durchgehen und deren nachbarn suchen, angefangen mit den knoten aus dem edp
            
                       
            it = 0
            while it < len(nodes):
                
                neighbors = list(nx.neighbors(graph, nodes[it])) #für jeden knoten aus nodes die nachbarn finden und gucken ob sie in den tree eingefügt werden dürfen

                for k in range(0,len(neighbors)): #jeden der nachbarn durchgehen
                    
                    if(neighbors[k] != nodes[it] and neighbors[k] != destination):
                        if(source == 1 and destination == 21 and debug == True):
                            print("Versuche die Kante" , ((nodes[it],neighbors[k])) , " einzufügen" )

                        #prüfen ob kante von nodes[it] nach neighbors[k] schon in anderen trees verbaut ist
                        is_in_other_tree = False
                        if(len(trees)>0):#wenn es schon andere trees gibt muss man alle anderen durchsuchen

                            if(source == 1 and destination == 21 and debug == True):
                                print("Bin in dem Fall, dass es andere Trees gibt")

                            for tree_to_check in trees: 
                                if (tree_to_check.has_edge(nodes[it],neighbors[k]) or tree_to_check.has_edge(neighbors[k],nodes[it]) ):
                                    if(source == 1 and destination == 21 and debug == True):
                                        print("Kante ist in anderem Tree")
                                    is_in_other_tree = True
                                    break
                                
                                else:
                                    if(source == 1 and destination == 21 and debug == True ):
                                        print("Kante ist NICHT in tree :", tree.nodes)
                                #endif
                            #endfor
                        
                            if not ( is_in_other_tree or (tree.has_node(neighbors[k])) ):
                                if(source == 1 and destination == 21 and debug == True):
                                    print("Kante ist NICHT in einem anderen Tree")
                                    print("Füge die Kante : ", nodes[j] , " - " , neighbors[k] , " ein bei len(trees) > 0")
                                nodes.append(neighbors[k]) 
                                tree.add_node(neighbors[k])
                                tree.add_edge(nodes[it],neighbors[k])
                            #endif
                        #endif

                        else: #das ist der fall wenn es noch keine anderen trees zum checken gibt, ob die kante schon verbaut ist
                            

                            if not((neighbors[k] == destination) or (tree.has_node(neighbors[k]))): #dann darf die kante nicht zur destination sein
                                                                                                    #der knoten darf nicht im jetzigen tree drin sein
                                
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

        while changed == True: #solange versuchen zu kürzen bis nicht mehr gekürzt werden kann 
            old_tree = tree.copy()
            remove_redundant_paths(source, destination, tree, graph) 
            changed = tree.order() != old_tree.order() # order returns the number of nodes in the graph.
        #endwhile

        #man muss prüfen ob nur die source im baum ist , da man im nächsten schritt der destination einen Rang geben muss
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
    

    return trees

########################################## MultTrees Änderung Breite ###############################################################
#der Algorithmus der die Baumbildung aufruft
def multiple_trees_pre_breite_mod(graph):
    paths = {}
    
    for source in graph.nodes:
        for destination in graph.nodes:
            if source != destination:
                
                edps = all_edps(source, destination, graph) #Bildung der EDPs
                
                edps.sort(key=len, reverse=True) #Sortierung der EDPs
                
                print("Start building trees with MultipleTrees Breite Mod for ", source , " to ", destination)
                trees = multiple_trees_breite_mod(source,destination,graph,edps, 2) #HIER KANN DER LETZTE FUNKTIONSPARAMETER GEÄNDERT WERDEN JE NACH GEWÜNSCHTER BREITE
                
                trees = remove_single_node_trees(trees)#EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben,
                                                        #führen nicht zum Ziel und müssen gelöscht werden
                

                print(" ")
                if source in paths:
                    paths[source][destination] = { 'trees': trees, 'edps': edps}
                else:
                    paths[source] = {}
                    paths[source][destination] = {'trees': trees, 'edps': edps}
         
    return paths

def multiple_trees_breite_mod(source, destination, graph, all_edps ,limitX):
    trees = [] #hier werden alle trees gespeichert 

    #für jeden tree muss hier sein edp eingefügt werden in den jeweiligen graph des trees 
    for i in range(0,len(all_edps)):

        current_edp = all_edps[i]
        tree = nx.DiGraph()
        tree.add_node(source)
        for j in range(1,len(current_edp)-1):
            tree.add_node(current_edp[j])
            tree.add_edge(current_edp[j-1], current_edp[j])

        trees.append(tree)

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

                    #hier muss dann zusätzlich geprüft werden ob der jetzige node noch weitere Kinder aufnehmen kann, da die Breite beschränkt wird in dieser Änderung
                    int_node = int(nodes[it])
                    outgoing_edges = list(tree.edges(int_node))
                    number_out_edges = len(outgoing_edges)                        
                    limit = limitX
                    
                    if(neighbors[k] != nodes[it] and neighbors[k] != destination and number_out_edges < limit): #kanten zu sich selbst dürfen nicht rein da dann baum zu kreis wird und kanten zur destination auch nicht
                        
                        
                        #prüfen ob kante von nodes[j] nach neighbors[k] schon in anderen trees verbaut ist
                        is_in_other_tree = False
                        if(len(trees)>0):#wenn es schon andere trees gibt muss man alle anderen durchsuchen
                            for tree_to_check in trees: 
                                if (tree_to_check.has_edge(nodes[it],neighbors[k]) or tree_to_check.has_edge(neighbors[k],nodes[it])): #wenn ein tree die edge schon drin hat dann darf man die edge nicht mehr benutzen
                                    is_in_other_tree = True
                                    break
                                #endif
                            #endfor
                        
                            if not ( is_in_other_tree or (tree.has_node(neighbors[k])) ):
                                nodes.append(neighbors[k]) 
                                tree.add_node(neighbors[k])
                                tree.add_edge(nodes[it],neighbors[k])
                            #endif
                        #endif
                        else: #das ist der fall wenn es noch keine anderen trees zum checken gibt, ob die kante schon verbaut ist
                            if not((neighbors[k] == destination) or (tree.has_node(neighbors[k]))): #dann darf die kante nicht zur destination sein
                                                                                                    #der knoten darf nicht im jetzigen tree drin sein
                                
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

        while changed == True: #solange versuchen zu kürzen bis nicht mehr gekürzt werden kann 
            old_tree = tree.copy()
            remove_redundant_paths(source, destination, tree, graph) 
            #removed_edgeList.extend(    list(   set(    list(   old_tree.edges  )    ) - set(    list(   tree.edges  )    )   )   )
            changed = tree.order() != old_tree.order() # order returns the number of nodes in the graph.
        #endwhile

        #man muss prüfen ob nur die source im baum ist , da man im nächsten schritt der destination einen Rang geben muss
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
    
    return trees

#################################################################################################################################
############################################# MultTrees mit Änderung der Baumbildung ###################################################################


removed_edges_parallel = []

#der Algorithmus der die Baumbildung aufruft
def multiple_trees_pre_parallel(graph):
    paths = {}
    count = 1
    all_graph_edge_number = len(graph.edges)
    all_tree_edge_number = 0
    
    for source in graph.nodes:
        for destination in graph.nodes:
            if source != destination:
                
                edps = all_edps(source, destination, graph) #Bildung der EDPs
                
                edps.sort(key=len, reverse=True) #Sortierung der EDPs
                
                print("Start building trees with MultipleTrees Parallel for ", source , " to ", destination)
                trees = multiple_trees_parallel(source,destination,graph,edps)
                
                trees = remove_single_node_trees(trees)#EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben,
                                                        #führen nicht zum Ziel und müssen gelöscht werden
                edges_of_this_run = 0 
                for tree in trees:
                    all_tree_edge_number = all_tree_edge_number + len(tree.edges)
                    edges_of_this_run = edges_of_this_run + len(tree.edges)
                
                count = count + 1
                print("Die Kanten dieses Laufs (modifiziert) : " , edges_of_this_run)
                print(" ")
                if source in paths:
                    paths[source][destination] = { 'trees': trees, 'edps': edps}
                else:
                    paths[source] = {}
                    paths[source][destination] = {'trees': trees, 'edps': edps}

    print("Bei einem count von " , count , " und insgesamt Graph Kanten " , all_graph_edge_number, " ergeben sich " , all_tree_edge_number , " Baumkanten bei der modifizierten (parallel) Variante")            
    print("Parallel durchschnittliche Truncation : ", (sum(removed_edges_parallel)/(len(removed_edges_parallel)-1)))
    return paths

#in dieser funktion werden die trees parallel gebaut, das bedeutet, dass pro tree jeweils 1 Kante eingebaut wird
#und dann im nächsten Tree eine Kante eingebaut wird


def multiple_trees_parallel(source, destination, graph, all_edps):
    removed_edges = 0
    trees = []
    nodes_in_tree = []
    #für jeden tree muss hier sein edp eingefügt werden in den jeweiligen graph des trees 
    for i in range(0,len(all_edps)):

        current_edp = all_edps[i]
        tree = nx.DiGraph()
        tree.add_node(source)
        for j in range(1,len(current_edp)-1):
            tree.add_node(current_edp[j])
            tree.add_edge(current_edp[j-1], current_edp[j])

        trees.append(tree)

    for i in range(0, len(all_edps)):
        nodes_in_tree.append( all_edps[i][:len(all_edps[i]) -1] ) #in nodes stehen dann alle knoten drin die wir besuchen wollen um deren nachbarn auch reinzupacken
                                                    # am anfang ganzer edp drin und -1 damit die destination nicht mit drin steht
                                                    
    assert len(trees) == len(all_edps) == len(nodes_in_tree), 'Not every edp got a tree!'

    changed = True
    j = 0
    while (changed) :
        changed = False

        for i in range(0,len(trees)): #jeden tree einmal durchgehen
                                        #um zu versuchen aus jedem edp einen Baum zu bauen
                                        
            tree = trees[i] # Baum aus vorheriger interation

            
            if j < len(nodes_in_tree[i]):
                changed = True # node_in_tree[i] array got elements left to work with

                        
                it = 0
                while it < len(nodes_in_tree[i]):
                    skip_while = False #die skip_while und break sind dafür da dass man genau 1 kante pro iteration einfügt
                    
                    neighbors = list(nx.neighbors(graph, nodes_in_tree[i][it])) #für jeden knoten aus nodes die nachbarn finden und gucken ob sie in den tree eingefügt werden dürfen
                    
                    for k in range(0,len(neighbors)): #jeden der nachbarn durchgehen
                        if(neighbors[k] != nodes_in_tree[i][it] and neighbors[k] != destination): #kanten zu sich selbst dürfen nicht rein da dann baum zu kreis wird und kanten zur destination auch nicht    
                            

                            #prüfen ob kante von nodes[j] nach neighbors[k] schon in anderen trees verbaut ist
                            is_in_other_tree = False
                            if(len(trees)>0):#wenn es schon andere trees gibt muss man alle anderen durchsuchen
                                for tree_to_check in trees: 
                                    if (tree_to_check.has_edge(nodes_in_tree[i][it],neighbors[k]) or tree_to_check.has_edge(neighbors[k],nodes_in_tree[i][it])): #wenn ein tree die edge schon drin hat dann darf man die edge nicht mehr benutzen
                                        is_in_other_tree = True
                                        break
                                    #endif
                                #endfor
                            
                                if not ( is_in_other_tree or (tree.has_node(neighbors[k])) ):
                                    nodes_in_tree[i].append(neighbors[k]) 
                                    tree.add_node(neighbors[k])
                                    tree.add_edge(nodes_in_tree[i][it],neighbors[k])
                                    skip_while = True
                                    break
                                #endif
                            #endif
                            else: #das ist der fall wenn es noch keine anderen trees zum checken gibt, ob die kante schon verbaut ist
                                if not((neighbors[k] == destination) or (tree.has_node(neighbors[k]))): #dann darf die kante nicht zur destination sein
                                                                                                        #der knoten darf nicht im jetzigen tree drin sein
                                    print("Füge die Kante : " , nodes_in_tree[i][it] , " - " , neighbors[k] , " ein bei len(trees) = 0")
                                    tree.add_node(neighbors[k])
                                    tree.add_edge(nodes_in_tree[i][it],neighbors[k])
                                #endif
                                #wenn der node der grad in den tree eingefügt wurde schon in nodes war dann soll er nicht nochmal eingefügt werden
                                if not (neighbors[k]in nodes_in_tree[i]): #damit knoten nicht doppelt in nodes eingefügt werden
                                    nodes_in_tree[i].append(neighbors[k]) 
                                #endif
                                skip_while = True
                                break
                            #endelse
                        #endif
                    #endfor
                    if skip_while:
                        break
                    it = it + 1                
                #endwhile
            #endif
        #endfor
        j = j+1 # next node in nodes array for new itteration
    #endwhile
    edpIndex = 0 
    i = 0
    for tree in trees:
        changed = True 
        old_edges = len(tree.edges)
        while changed == True: #solange versuchen zu kürzen bis nicht mehr gekürzt werden kann 
            old_tree = tree.copy()
            remove_redundant_paths(source, destination, tree, graph) 
            changed = tree.order() != old_tree.order() # order returns the number of nodes in the graph.
        #endwhile
        new_edges = len(tree.edges)
        removed_edges = removed_edges + (old_edges - new_edges)
        #man muss prüfen ob nur die source im baum ist , da man im nächsten schritt der destination einen Rang geben muss
        if( tree.order() > 1 ):
            rank_tree(tree , source,all_edps[edpIndex])
            connect_leaf_to_destination(tree, source,destination)
            tree.add_edge(all_edps[i][len(all_edps[i])-2],destination)
            tree.nodes[destination]["rank"] = -1
            edpIndex = edpIndex+1
        #endif

        #edps direkt von s->d kommen müssen gesondert betrachtet werden
        if(tree.order() == 1 and len(all_edps[i]) == 2):
            tree.add_edge(source,destination)
            tree.nodes[destination]["rank"] = -1
        #endif

    #endfor
    #print("In parallel removed_edges : ", removed_edges)
    removed_edges_parallel.append(removed_edges)
    return trees


####################################################################################################################################

########################################## MultTrees Parallel + Recycling ###############################################################


#Algorithmus der anfags wie die parallele Modifikation den Baum baut und nahcdem die Bäume gekürzt wurden, werden die Bäume nochmal
#der Reihenfolge nach versucht erweitert zu werden. Recycling im Sinne von : Kanten werden "weggeschmissen" in der Truncatio eines Baumes,
#aber ein anderer Baum könnte diese Kanten wieder aufnehmen
def multiple_trees_pre_recycling(graph):
    paths = {}
    count = 1
    all_graph_edge_number = len(graph.edges)
    all_tree_edge_number = 0
    for source in graph.nodes:
        for destination in graph.nodes:
            if source != destination:
                
                edps = all_edps(source, destination, graph) #Bildung der EDPs
                
                edps.sort(key=len, reverse=True) #Sortierung der EDPs
                
                print("Start building trees with MultipleTrees Recycling for ", source , " to ", destination)
                trees = multiple_trees_recycling(source,destination,graph,edps)
                
                trees = remove_single_node_trees(trees)#EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben,
                                                        #führen nicht zum Ziel und müssen gelöscht werden
                edges_of_this_run = 0 
                for tree in trees:
                    all_tree_edge_number = all_tree_edge_number + len(tree.edges)
                    edges_of_this_run = edges_of_this_run + len(tree.edges)
                
                
                count = count + 1
                print("Die Kanten dieses Laufs (modifiziert) : " , edges_of_this_run)
                print(" ")
                #print_trees(source,destination,trees)
                print(" ")
                if source in paths:
                    paths[source][destination] = { 'trees': trees, 'edps': edps}
                else:
                    paths[source] = {}
                    paths[source][destination] = {'trees': trees, 'edps': edps}

    print("Bei einem count von " , count , " und insgesamt Graph Kanten " , all_graph_edge_number, " ergeben sich " , all_tree_edge_number , " Baumkanten bei der modifizierten (recycling) Variante")            
    return paths

def multiple_trees_recycling(source, destination, graph, all_edps):
    removed_edges = 0
    trees = []
    nodes_in_tree = []
    print("All Edps : ", all_edps)
    #für jeden tree muss hier sein edp eingefügt werden in den jeweiligen graph des trees 
    for i in range(0,len(all_edps)):

        current_edp = all_edps[i]
        tree = nx.DiGraph()
        tree.add_node(source)
        for j in range(1,len(current_edp)-1):
            tree.add_node(current_edp[j])
            tree.add_edge(current_edp[j-1], current_edp[j])

        trees.append(tree)

    for i in range(0, len(all_edps)):
        nodes_in_tree.append( all_edps[i][:len(all_edps[i]) -1] ) #in nodes stehen dann alle knoten drin die wir besuchen wollen um deren nachbarn auch reinzupacken
                                                    # am anfang ganzer edp drin und -1 damit die destination nicht mit drin steht
                                                    
    assert len(trees) == len(all_edps) == len(nodes_in_tree), 'Not every edp got a tree!'

    changed = True
    j = 0
    while (changed) :
        changed = False

        for i in range(0,len(trees)): #jeden tree einmal durchgehen
                                        #um zu versuchen aus jedem edp einen Baum zu bauen
                                        
            tree = trees[i] # Baum aus vorheriger interation

            
            if j < len(nodes_in_tree[i]):
                changed = True # node_in_tree[i] array got elements left to work with

                        
                it = 0
                while it < len(nodes_in_tree[i]):
                    skip_while = False #die skip_while und break sind dafür da dass man genau 1 kante pro iteration einfügt
                    
                    neighbors = list(nx.neighbors(graph, nodes_in_tree[i][it])) #für jeden knoten aus nodes die nachbarn finden und gucken ob sie in den tree eingefügt werden dürfen
                    
                    for k in range(0,len(neighbors)): #jeden der nachbarn durchgehen
                        if(neighbors[k] != nodes_in_tree[i][it] and neighbors[k] != destination): #kanten zu sich selbst dürfen nicht rein da dann baum zu kreis wird und kanten zur destination auch nicht    
                            

                            #prüfen ob kante von nodes[j] nach neighbors[k] schon in anderen trees verbaut ist
                            is_in_other_tree = False
                            if(len(trees)>0):#wenn es schon andere trees gibt muss man alle anderen durchsuchen
                                for tree_to_check in trees: 
                                    if (tree_to_check.has_edge(nodes_in_tree[i][it],neighbors[k]) or tree_to_check.has_edge(neighbors[k],nodes_in_tree[i][it])): #wenn ein tree die edge schon drin hat dann darf man die edge nicht mehr benutzen
                                        is_in_other_tree = True
                                        break
                                    #endif
                                #endfor
                            
                                if not ( is_in_other_tree or (tree.has_node(neighbors[k])) ):
                                    nodes_in_tree[i].append(neighbors[k]) 
                                    tree.add_node(neighbors[k])
                                    tree.add_edge(nodes_in_tree[i][it],neighbors[k])
                                    skip_while = True
                                    break
                                #endif
                            #endif
                            else: #das ist der fall wenn es noch keine anderen trees zum checken gibt, ob die kante schon verbaut ist
                                if not((neighbors[k] == destination) or (tree.has_node(neighbors[k]))): #dann darf die kante nicht zur destination sein
                                                                                                        #der knoten darf nicht im jetzigen tree drin sein
                                    print("Füge die Kante : " , nodes_in_tree[i][it] , " - " , neighbors[k] , " ein bei len(trees) = 0")
                                    tree.add_node(neighbors[k])
                                    tree.add_edge(nodes_in_tree[i][it],neighbors[k])
                                #endif
                                #wenn der node der grad in den tree eingefügt wurde schon in nodes war dann soll er nicht nochmal eingefügt werden
                                if not (neighbors[k]in nodes_in_tree[i]): #damit knoten nicht doppelt in nodes eingefügt werden
                                    nodes_in_tree[i].append(neighbors[k]) 
                                #endif
                                skip_while = True
                                break
                            #endelse
                        #endif
                    #endfor
                    if skip_while:
                        break
                    it = it + 1                
                #endwhile
            #endif
        #endfor
        j = j+1 # next node in nodes array for new itteration
    #endwhile
    edpIndex = 0 
    i = 0
    removed_edgeList = []
    for tree in trees:
        changed = True 
        old_edges = len(tree.edges)

        while changed == True: #solange versuchen zu kürzen bis nicht mehr gekürzt werden kann 

            old_tree = tree.copy()

            remove_redundant_paths(source, destination, tree, graph) 
            
            removed_edgeList.extend(    list(   set(    list(   old_tree.edges  )    ) - set(    list(   tree.edges  )    )   )   )

            changed = tree.order() != old_tree.order() # order returns the number of nodes in the graph.

        #endwhile
        
        new_edges = len(tree.edges)

        removed_edges = removed_edges + (old_edges - new_edges)
        
        
        i = i +1
    #endfor
    #print("In parallel removed_edges : ", removed_edges)
    removed_edges_parallel.append(removed_edges)

    
    #hier würde der parallele Algorithmus eigentlich die Bäume zurückgeben und die Baumbildung beenden
    #aber in der Recycling Variante wird hier die normale Baumbildung von MultipleTrees eingestzt
    trees = recycleTrees(trees,source,destination,graph,all_edps)

    countTrees = 0
    all_edges = []
    for tree in trees:
        all_edges.extend(list(tree.edges))
        countTrees = countTrees +1 
        
    for edge in all_edges:
        if edge in removed_edgeList:
            print("Kante : ", edge , " wurde recyclet ")
    

    return trees


#Baumbildung wie in MultipleTrees aber hier werden Bäume erweitert und nicht EDPs
def recycleTrees(trees,source,destination,graph,all_edps):
    for i in range(0,len(trees)): #jeden baum einmal durchgehen
                                      #um zu versuchen jeden baum zu erweitern
        
        tree = trees[i] # Baum der zuvor gekürzt wurde
        pathToExtend = list(tree.nodes)

        ################# DEBUG ONLY ################
        tree_before_recycling = tree.copy()
        if( tree_before_recycling.order() > 1 ):
            rank_tree(tree_before_recycling , source,all_edps[i])
            connect_leaf_to_destination(tree_before_recycling, source,destination)
            tree_before_recycling.add_edge(all_edps[i][len(all_edps[i])-2],destination)
            tree_before_recycling.nodes[destination]["rank"] = -1
            
        #endif
        #edps direkt von s->d kommen müssen gesondert betrachtet werden
        if(tree_before_recycling.order() == 1 and len(all_edps[i]) == 2):
            tree_before_recycling.add_edge(source,destination)
            tree_before_recycling.nodes[destination]["rank"] = -1
        #endif
        ##############################################


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
                               
                                if (tree_to_check.has_edge(nodes[it],neighbors[k])or tree_to_check.has_edge(neighbors[k],nodes[it]) ): #wenn ein tree die edge schon drin hat dann darf man die edge nicht mehr benutzen
                                    is_in_other_tree = True
                                    break
                                #endif
                            #endfor
                        
                            if not ( is_in_other_tree or (tree.has_node(neighbors[k])) ):
                                
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

        counting = 0
        while changed == True: #solange versuchen zu kürzen bis nicht mehr gekürzt werden kann 
            counting = counting + 1
            old_tree = tree.copy()
            remove_redundant_paths(source, destination, tree, graph) 
            changed = tree.order() != old_tree.order() # order returns the number of nodes in the graph
        #endwhile
        
        #man muss prüfen ob nur die source im baum ist , da man im nächsten schritt der destination einen Rang geben muss
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
    return trees
####################################################################################################################################






####################################################################################################################################

# Funktion zur Überprüfung, ob sich die Positionen der Knoten überlappen
def check_overlap(node, pos, other_nodes, other_pos):
    for other_node, other_position in zip(other_nodes, other_pos):
        if node != other_node:
            if abs(pos[0] - other_position[0]) < 0.5 and abs(pos[1] - other_position[1]) < 0.5:
                return True
    return False





#methode um für jedes source destination paar einen baum zu bauen
def one_tree_pre(graph):
    #die paths struktur besteht daraus : für jeden source (1. index) zu jeder destination (2. index) gibt es 1 Objekt dass den Baum drin hat (Attribut 'tree') und alle EDPs (Attribut 'edps')
    # und alle weglängen zur destination in 'distance'

    paths = {}

    for source in graph.nodes:

        for destination in graph.nodes:
            
            if source != destination: #and source == 28 and destination == 13:                
                edps = all_edps(source, destination, graph)
                edps.sort(key=len)
                longest_edp = edps[len(edps)-1]

                
                tree = one_tree(source,destination,graph,longest_edp)
                if source in paths:
                    paths[source][destination] = { 'tree': tree, 'edps': edps}
                else:
                    paths[source] = {}
                    paths[source][destination] = {'tree': tree, 'edps': edps}
                
                edps.sort(key=len)
                
    return paths

#hilfsfunktion damit man die weglänge von jedem node zur distance hat , das braucht man um die reihenfolge festzulegen die man bei den verzweigungen nimmt 
def compute_distance_to_dest(tree, destination):
    return dict(nx.single_target_shortest_path_length(tree, destination))

#den baum bauen indem man jeden knoten von der source aus mitnimmt der mit einem knoten aus dem baum benachbart ist
#dabei guckt man sich die nachbarn im ursprungsgraphen  an und fügt die dann in einem anderen graphen (tree) ein
# am ende löscht man noch die pfade die nicht zum destination führen
# der baum ist ein gerichteter graph , damit man im tree die struktur zwischen parent/children erkennen kann anhand eingehender/ausgehender kanten
def one_tree(source, destination, graph, longest_edp):
    tree = nx.DiGraph()
    assert source == longest_edp[0] , 'Source is not start of edp'
    tree.add_node(source) # source = longest_edp[0]

    #hier muss noch hin dass wir den edp an sich reinmachen
    for i in range(1,len(longest_edp)-1): # -2 da wir die destination ncht einfügen wollen
        tree.add_node(longest_edp[i])
        tree.add_edge(longest_edp[i-1],longest_edp[i])

    pathToExtend = longest_edp
    
    for i in range(0,len(pathToExtend)-1): # i max 7
        
        nodes = pathToExtend[:len(pathToExtend) -2]
        it = 0 # um die nachbarn der nachbarn zu bekommen
        while it < len(nodes):

            neighbors = list(nx.neighbors(graph, nodes[it]))
            for j in neighbors:
                if (not tree.has_node(j)) and (j!= destination): #not part of tree already and not the destiantion
                    nodes.append(j)
                    tree.add_node(j) #add neighbors[j] to tree
                    tree.add_edge(nodes[it], j) # add edge to new node
                #end if
            #end for
            it = it+1
        #end while
    #end for
    

    #PG1 = nx.nx_pydot.write_dot(tree , "./breite_mod_trees/treeNBT"+str(source)+"-"+str(destination))

    changed = True 
    while changed == True: #solange versuchen zu kürzen bis nicht mehr gekürzt werden kann 
        old_tree = tree.copy()
        remove_redundant_paths(source, destination, tree, graph)
        changed = tree.order() != old_tree.order() # order returns the number of nodes in the graph.

    
    
    rank_tree(tree , source,longest_edp)
    
    connect_leaf_to_destination(tree, source, destination)
    tree.add_edge(longest_edp[len(longest_edp)-2],destination)
    #add 'rank' property to the added destinaton, -1 for highest priorty in routing
    tree.nodes[destination]["rank"] = -1


    return tree


#################################### OneTree Breite Mod ############################################################################################

#methode um für jedes source destination paar einen baum zu bauen
def one_tree_pre_breite_mod(graph):
    #die paths struktur besteht daraus : für jeden source (1. index) zu jeder destination (2. index) gibt es 1 Objekt dass den Baum drin hat (Attribut 'tree') und alle EDPs (Attribut 'edps')
    # und alle weglängen zur destination in 'distance'

    paths = {}
    
    for source in graph.nodes:

        for destination in graph.nodes:
            
            if source != destination: #and source == 28 and destination == 13:                
                edps = all_edps(source, destination, graph)
                edps.sort(key=len)
                longest_edp = edps[len(edps)-1]

                #################################################################

                #den letzten Parameter nach der gewünschten Breite verändern

                tree = one_tree_breite_mod(source,destination,graph,longest_edp,2) 
                
                #################################################################
                if source in paths:
                    paths[source][destination] = { 'tree': tree, 'edps': edps}
                else:
                    paths[source] = {}
                    paths[source][destination] = {'tree': tree, 'edps': edps}

                
    return paths

#hilfsfunktion damit man die weglänge von jedem node zur distance hat , das braucht man um die reihenfolge festzulegen die man bei den verzweigungen nimmt 
def compute_distance_to_dest(tree, destination):
    return dict(nx.single_target_shortest_path_length(tree, destination))

#den baum bauen indem man jeden knoten von der source aus mitnimmt der mit einem knoten aus dem baum benachbart ist
#dabei guckt man sich die nachbarn im ursprungsgraphen  an und fügt die dann in einem anderen graphen (tree) ein
# am ende löscht man noch die pfade die nicht zum destination führen
# der baum ist ein gerichteter graph , damit man im tree die struktur zwischen parent/children erkennen kann anhand eingehender/ausgehender kanten

def one_tree_breite_mod(source, destination, graph, longest_edp,limitX):

    tree = nx.DiGraph()

    assert source == longest_edp[0] , 'Source is not start of edp'

    tree.add_node(source) # source = longest_edp[0]

    #hier muss noch hin dass wir den edp an sich reinmachen
    for i in range(1,len(longest_edp)-1): # -2 da wir die destination ncht einfügen wollen
        tree.add_node(longest_edp[i])
        tree.add_edge(longest_edp[i-1],longest_edp[i])

    pathToExtend = longest_edp
    
    for i in range(0,len(pathToExtend)-1): # i max 7
        


        nodes = pathToExtend[:len(pathToExtend) -2]
        it = 0 # um die nachbarn der nachbarn zu bekommen
        while it < len(nodes):

            neighbors = list(nx.neighbors(graph, nodes[it]))

            for j in neighbors:

                #hier muss dann zusätzlich geprüft werden ob der jetzige node noch weitere Kinder aufnehmen kann, da die Breite beschränkt wird in dieser Änderung
                int_node = int(nodes[it])
                outgoing_edges = list(tree.edges(int_node))
                number_out_edges = len(outgoing_edges)                        
                limit = limitX


                if(number_out_edges < limit): 

                    
                    if (not tree.has_node(j)) and (j!= destination): #not part of tree already and not the destiantion
                        nodes.append(j)
                        
                        tree.add_node(j) #add neighbors[j] to tree
                        tree.add_edge(nodes[it], j) # add edge to new node
                    #end if

                else:
                    
                    break;
                #endif
            #end for
            
            it = it+1
        #end while
    #end for
    
    changed = True 
    while changed == True: #solange versuchen zu kürzen bis nicht mehr gekürzt werden kann 
        old_tree = tree.copy()
        remove_redundant_paths(source, destination, tree, graph)
        changed = tree.order() != old_tree.order() # order returns the number of nodes in the graph.
    #endwhile 

    rank_tree(tree , source , longest_edp)
    connect_leaf_to_destination(tree, source, destination)
    tree.add_edge(longest_edp[len(longest_edp)-2],destination)
    # Destination hat den kleinsten möglichen Rang
    tree.nodes[destination]["rank"] = -1

    return tree

####################################################################################################################################################
#################################### OneTree Inverse Mod ############################################################################################
#methode um für jedes source destination paar einen baum zu bauen
#der baum wird aus dem kürzesten EDP erstellt
def one_tree_pre_mod_inverse(graph):
    #die paths struktur besteht daraus : für jeden source (1. index) zu jeder destination (2. index) gibt es 1 Objekt dass den Baum drin hat (Attribut 'tree') und alle EDPs (Attribut 'edps')
    # und alle weglängen zur destination in 'distance'

    paths = {}

    for source in graph.nodes:

        for destination in graph.nodes:
            
            if source != destination: #and source == 28 and destination == 13:                
                edps = all_edps(source, destination, graph)
                edps.sort(key=len)
                shortest_edp = edps[0]

                
                tree = one_tree(source,destination,graph,shortest_edp)
                if source in paths:
                    paths[source][destination] = { 'tree': tree, 'edps': edps}
                else:
                    paths[source] = {}
                    paths[source][destination] = {'tree': tree, 'edps': edps}
               
                
    return paths


################################# Kombinationen ####################################################################################

######################### Parallel & Invertiert #######################################################################
def multiple_trees_pre_parallel_and_inverse(graph):
    paths = {}
    count = 1
    all_graph_edge_number = len(graph.edges)
    all_tree_edge_number = 0
    
    for source in graph.nodes:
        for destination in graph.nodes:
            if source != destination:
                
                edps = all_edps(source, destination, graph) #Bildung der EDPs
                
                edps.sort(key=len, reverse=True) #Sortierung der EDPs
                
                print("Start building trees with MultipleTrees Parallel and Inverse for ", source , " to ", destination)
                trees = multiple_trees_parallel_and_inverse(source,destination,graph,edps)
                
                trees = remove_single_node_trees(trees)#EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben,
                                                        #führen nicht zum Ziel und müssen gelöscht werden
                edges_of_this_run = 0 
                for tree in trees:
                    all_tree_edge_number = all_tree_edge_number + len(tree.edges)
                    edges_of_this_run = edges_of_this_run + len(tree.edges)
                
                count = count + 1
                print("Die Kanten dieses Laufs (modifiziert) : " , edges_of_this_run)
                print(" ")
                #print_trees(source,destination,trees)
                print(" ")
                if source in paths:
                    paths[source][destination] = { 'trees': trees, 'edps': edps}
                else:
                    paths[source] = {}
                    paths[source][destination] = {'trees': trees, 'edps': edps}

    print("Bei einem count von " , count , " und insgesamt Graph Kanten " , all_graph_edge_number, " ergeben sich " , all_tree_edge_number , " Baumkanten bei der modifizierten (parallel + inverse) Variante")            
    return paths

#in dieser funktion werden die trees parallel gebaut, das bedeutet, dass pro tree jeweils 1 Kante eingebaut wird
#und dann im nächsten Tree eine Kante eingebaut wird


def multiple_trees_parallel_and_inverse(source, destination, graph, all_edps):
 
    trees = []
    nodes_in_tree = []


    mode = "invert" # random / invert

   
    #reihenfolge wird zufällig gewählt
    if(mode == "random" ):
        print("Randomizinig EDPs")
        print("EDPs before shuffle : " , all_edps)
        random.shuffle(all_edps)
        print("EDPs after shuffle : " , all_edps)
    
    #reihenfolge wird invertiert
    if(mode == "invert" ):
        print("Inverting EDPs")
        print("EDPs before inverting : ", all_edps)
        all_edps.sort(key=len, reverse=False)
        print("EDPs after inverting : ", all_edps)

    #für jeden tree muss hier sein edp eingefügt werden in den jeweiligen graph des trees 
    for i in range(0,len(all_edps)):

        current_edp = all_edps[i]
        tree = nx.DiGraph()
        tree.add_node(source)
        for j in range(1,len(current_edp)-1):
            tree.add_node(current_edp[j])
            tree.add_edge(current_edp[j-1], current_edp[j])

        trees.append(tree)

    for i in range(0, len(all_edps)):
        nodes_in_tree.append( all_edps[i][:len(all_edps[i]) -1] ) #in nodes stehen dann alle knoten drin die wir besuchen wollen um deren nachbarn auch reinzupacken
                                                    # am anfang ganzer edp drin und -1 damit die destination nicht mit drin steht
                                                    
    assert len(trees) == len(all_edps) == len(nodes_in_tree), 'Not every edp got a tree!'

    changed = True
    j = 0
    while (changed) :
        changed = False

        for i in range(0,len(trees)): #jeden tree einmal durchgehen
                                        #um zu versuchen aus jedem edp einen Baum zu bauen
                                        
            tree = trees[i] # Baum aus vorheriger interation

            
            if j < len(nodes_in_tree[i]):
                changed = True # node_in_tree[i] array got elements left to work with

                        
                it = 0
                while it < len(nodes_in_tree[i]):
                    skip_while = False #die skip_while und break sind dafür da dass man genau 1 kante pro iteration einfügt
                    
                    neighbors = list(nx.neighbors(graph, nodes_in_tree[i][it])) #für jeden knoten aus nodes die nachbarn finden und gucken ob sie in den tree eingefügt werden dürfen
                    
                    for k in range(0,len(neighbors)): #jeden der nachbarn durchgehen
                        if(neighbors[k] != nodes_in_tree[i][it] and neighbors[k] != destination): #kanten zu sich selbst dürfen nicht rein da dann baum zu kreis wird und kanten zur destination auch nicht    
                            

                            #prüfen ob kante von nodes[j] nach neighbors[k] schon in anderen trees verbaut ist
                            is_in_other_tree = False
                            if(len(trees)>0):#wenn es schon andere trees gibt muss man alle anderen durchsuchen
                                for tree_to_check in trees: 
                                    if (tree_to_check.has_edge(nodes_in_tree[i][it],neighbors[k]) or tree_to_check.has_edge(neighbors[k],nodes_in_tree[i][it])): #wenn ein tree die edge schon drin hat dann darf man die edge nicht mehr benutzen
                                        is_in_other_tree = True
                                        break
                                    #endif
                                #endfor
                            
                                if not ( is_in_other_tree or (tree.has_node(neighbors[k])) ):
                                    nodes_in_tree[i].append(neighbors[k]) 
                                    tree.add_node(neighbors[k])
                                    tree.add_edge(nodes_in_tree[i][it],neighbors[k])
                                    skip_while = True
                                    break
                                #endif
                            #endif
                            else: #das ist der fall wenn es noch keine anderen trees zum checken gibt, ob die kante schon verbaut ist
                                if not((neighbors[k] == destination) or (tree.has_node(neighbors[k]))): #dann darf die kante nicht zur destination sein
                                                                                                        #der knoten darf nicht im jetzigen tree drin sein
                                    tree.add_node(neighbors[k])
                                    tree.add_edge(nodes_in_tree[i][it],neighbors[k])
                                #endif
                                #wenn der node der grad in den tree eingefügt wurde schon in nodes war dann soll er nicht nochmal eingefügt werden
                                if not (neighbors[k]in nodes_in_tree[i]): #damit knoten nicht doppelt in nodes eingefügt werden
                                    nodes_in_tree[i].append(neighbors[k]) 
                                #endif
                                skip_while = True
                                break
                            #endelse
                        #endif
                    #endfor
                    if skip_while:
                        break
                    it = it + 1                
                #endwhile
            #endif
        #endfor
        j = j+1 # next node in nodes array for new itteration
    #endwhile
    edpIndex = 0 
    i = 0
    for tree in trees:
        changed = True 

        while changed == True: #solange versuchen zu kürzen bis nicht mehr gekürzt werden kann 
            old_tree = tree.copy()
            remove_redundant_paths(source, destination, tree, graph) 
            changed = tree.order() != old_tree.order() # order returns the number of nodes in the graph.
        #endwhile

        #man muss prüfen ob nur die source im baum ist , da man im nächsten schritt der destination einen Rang geben muss
        if( tree.order() > 1 ):
            rank_tree(tree , source,all_edps[edpIndex])
            connect_leaf_to_destination(tree, source,destination)
            tree.add_edge(all_edps[i][len(all_edps[i])-2],destination)
            tree.nodes[destination]["rank"] = -1
            edpIndex = edpIndex+1
        #endif

        #edps direkt von s->d kommen müssen gesondert betrachtet werden
        if(tree.order() == 1 and len(all_edps[i]) == 2):
            tree.add_edge(source,destination)
            tree.nodes[destination]["rank"] = -1
        #endif
        i = i+1
    return trees
#######################################################################################################################

##################################### KOMBINATIONEN #####################################################################


##################################### Breite & Invertiert ###############################################################

#der Algorithmus der die Baumbildung aufruft


#Kombination aus der veränderten Breite mit invertiertier Reihenfolge der EDPs
def multiple_trees_pre_breite_mod_and_inverse(graph):
    paths = {}
    
    for source in graph.nodes:
        for destination in graph.nodes:
            if source != destination:
                
                edps = all_edps(source, destination, graph) #Bildung der EDPs
                
                edps.sort(key=len, reverse=True) #Sortierung der EDPs
                
                print("Start building trees with MultipleTrees Breite Mod for ", source , " to ", destination)
                trees = multiple_trees_breite_mod_and_inverse(source,destination,graph,edps, 2) #HIER KANN DER LETZTE FUNKTIONSPARAMETER GEÄNDERT WERDEN JE NACH GEWÜNSCHTER BREITE
                
                trees = remove_single_node_trees(trees)#EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben,
                                                        #führen nicht zum Ziel und müssen gelöscht werden

                print(" ")
                if source in paths:
                    paths[source][destination] = { 'trees': trees, 'edps': edps}
                else:
                    paths[source] = {}
                    paths[source][destination] = {'trees': trees, 'edps': edps}
      
    return paths

#gibt die Bäume zurück für ein Source-Destination-Paar

#dabei wird erst die Reihenolge der EDPs invertiert und anschließend die Baumbildung mit "limitX"
#Breite limitiert
def multiple_trees_breite_mod_and_inverse(source, destination, graph, all_edps ,limitX):
    trees = [] #hier werden alle trees gespeichert 


    mode = "invert" # random / invert

   
    #reihenfolge wird zufällig gewählt
    if(mode == "random" ):
        print("Randomizinig EDPs")
        print("EDPs before shuffle : " , all_edps)
        random.shuffle(all_edps)
        print("EDPs after shuffle : " , all_edps)
    
    #reihenfolge wird invertiert
    if(mode == "invert" ):
        print("Inverting EDPs")
        print("EDPs before inverting : ", all_edps)
        all_edps.sort(key=len, reverse=False)
        print("EDPs after inverting : ", all_edps)

    #für jeden tree muss hier sein edp eingefügt werden in den jeweiligen graph des trees 
    for i in range(0,len(all_edps)):

        current_edp = all_edps[i]
        tree = nx.DiGraph()
        tree.add_node(source)
        for j in range(1,len(current_edp)-1):
            tree.add_node(current_edp[j])
            tree.add_edge(current_edp[j-1], current_edp[j])

        trees.append(tree)

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

                    #hier muss dann zusätzlich geprüft werden ob der jetzige node noch weitere Kinder aufnehmen kann, da die Breite beschränkt wird in dieser Änderung
                    int_node = int(nodes[it])
                    outgoing_edges = list(tree.edges(int_node))
                    number_out_edges = len(outgoing_edges)                        
                    limit = limitX
                    
                    if(neighbors[k] != nodes[it] and neighbors[k] != destination and number_out_edges < limit): #kanten zu sich selbst dürfen nicht rein da dann baum zu kreis wird und kanten zur destination auch nicht
                        
                        
                        #prüfen ob kante von nodes[j] nach neighbors[k] schon in anderen trees verbaut ist
                        is_in_other_tree = False
                        if(len(trees)>0):#wenn es schon andere trees gibt muss man alle anderen durchsuchen
                            for tree_to_check in trees: 
                                if (tree_to_check.has_edge(nodes[it],neighbors[k])or tree_to_check.has_edge(neighbors[k],nodes[it])): #wenn ein tree die edge schon drin hat dann darf man die edge nicht mehr benutzen
                                    is_in_other_tree = True
                                    break
                                #endif
                            #endfor
                        
                            if not ( is_in_other_tree or (tree.has_node(neighbors[k])) ):
                                nodes.append(neighbors[k]) 
                                tree.add_node(neighbors[k])
                                tree.add_edge(nodes[it],neighbors[k])
                            #endif
                        #endif
                        else: #das ist der fall wenn es noch keine anderen trees zum checken gibt, ob die kante schon verbaut ist
                            if not((neighbors[k] == destination) or (tree.has_node(neighbors[k]))): #dann darf die kante nicht zur destination sein
                                                                                                    #der knoten darf nicht im jetzigen tree drin sein
                                
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

        while changed == True: #solange versuchen zu kürzen bis nicht mehr gekürzt werden kann 
            old_tree = tree.copy()
            remove_redundant_paths(source, destination, tree, graph) 
            changed = tree.order() != old_tree.order() # order returns the number of nodes in the graph.
        #endwhile

        #man muss prüfen ob nur die source im baum ist , da man im nächsten schritt der destination einen Rang geben muss
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
    return trees







#######################################################################################################################

##################################### Anzahl & Random Reihenfolge ###############################################################


#Kombination aus der Randomisierten Anzahl und Reihenfolge
def multiple_trees_pre_num_of_trees_mod_and_random_order(graph):
    paths = {}
    count = 1
    all_graph_edge_number = len(graph.edges)
    all_tree_edge_number = 0
    
    for source in graph.nodes:

        for destination in graph.nodes:
            
            if source != destination:
                
                edps = all_edps(source, destination, graph) #Bildung der EDPs
                
                edps.sort(key=len, reverse=True) #Sortierung der EDPs
                
                #print("Start building trees for ", source , " to ", destination)
                trees = multiple_trees_num_of_trees_mod(source,destination,graph,edps)
                
                trees = remove_single_node_trees(trees)#EDPs die nicht erweitert werden konnten, da andere Bäume die Kanten schon vorher verbaut haben,
                                                        #führen nicht zum Ziel und müssen gelöscht werden
                
                
                
                edges_of_this_run = 0 
                for tree in trees:
                    all_tree_edge_number = all_tree_edge_number + len(tree.edges)
                    edges_of_this_run = edges_of_this_run + len(tree.edges)
                count = count + 1
                print("Die Kanten dieses Laufs (modifiziert) : " , edges_of_this_run)
                print(" ")
                if source in paths:
                    paths[source][destination] = { 'trees': trees, 'edps': edps}
                else:
                    paths[source] = {}
                    paths[source][destination] = {'trees': trees, 'edps': edps}
    print("Bei einem count von " , count , " und insgesamt Graph Kanten " , all_graph_edge_number, " ergeben sich " , all_tree_edge_number , " Baumkanten bei der modifizierten  (num + random order) Variante")
                
    return paths

#gibt für ein source-destination paar alle trees zurück
def multiple_trees_num_of_trees_mod_and_random_order(source, destination, graph, all_edps):
    trees = [] #hier werden alle trees gespeichert 

    
    debug = False

    #Zuerst wird die Reihenfolge randomisiert und anschließend werden randomisiert EDPs ausgewählt
    random.shuffle(all_edps)

    number_of_wanted_trees = 3 #diese Zahl muss geändert werden, damit man die Anzahl an zu bauenden Bäumen einschränkt

    number_of_edps = len(all_edps)
    
    print("Versuche " , number_of_wanted_trees , " aus " , number_of_edps , " zu bilden ")


    #dann so (zufällige Zahl)-viele Elemente aus dem all_edps[] in einer subliste speichern 
    #dann die all_edps = subliste setzen
    sublist = []
    try:
        indexes_for_sublist = random.sample(range(0, number_of_edps), number_of_wanted_trees) # auswgewählt-viele zufällige Zahlen zwischen 1 - Anzahl an EDPs

        print("Die zufällig gewählten Indezes : ", indexes_for_sublist)

        for i in range(0, len(indexes_for_sublist)): #die zufällig gewählten edps in die sublist einfügen
            index = indexes_for_sublist[i]
            sublist.append(all_edps[index])  
        #endfor
        print("all_edps vor der Änderung : " , all_edps)
        all_edps = sublist
        print("all_edps nach der Änderung : " , all_edps)
    #endtry
    except ValueError:
        print('Zu viele EDPs ausgewählt, es werden alle EDPs genutzt') #wenn man versucht zu viele edps zu wählen 
    #endexcept

   
    #reihenfolge wird zufällig gewählt
    random.shuffle(all_edps)
    
        

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
                    
                    if(neighbors[k] != nodes[it] and neighbors[k] != destination):
                        
                        #prüfen ob kante von nodes[it] nach neighbors[k] schon in anderen trees verbaut ist
                        is_in_other_tree = False
                        if(len(trees)>0):#wenn es schon andere trees gibt muss man alle anderen durchsuchen

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
                        #endif

                        else: #das ist der fall wenn es noch keine anderen trees zum checken gibt, ob die kante schon verbaut ist
                            if(source == 1 and destination == 21 and debug == True):
                                print("Es gibt noch keine anderen Trees zum checken")

                            if not((neighbors[k] == destination) or (tree.has_node(neighbors[k]))): #dann darf die kante nicht zur destination sein
                                                                                                    #der knoten darf nicht im jetzigen tree drin sein
                                
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

        while changed == True: #solange versuchen zu kürzen bis nicht mehr gekürzt werden kann 
            old_tree = tree.copy()
            remove_redundant_paths(source, destination, tree, graph) 
            changed = tree.order() != old_tree.order() # order returns the number of nodes in the graph.
        #endwhile

        #man muss prüfen ob nur die source im baum ist , da man im nächsten schritt der destination einen Rang geben muss
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

    
    return trees







#######################################################################################################################



############################### Hilfsfunktionen ####################################################################################

#hilfsfunktion, welche bäume aus der liste entfernt die nur aus der source bestehen
def remove_single_node_trees(trees):
    new_trees = []
    for tree in trees:
        if(tree.order() > 1):
            new_trees.append(tree)
    return new_trees

#hilfsfunktion mit der man die trees aus der multipletrees im ordner speichern kann 
def print_trees(source,destination,trees):
    index = 0
    for tree in trees:
    #    PG = nx.nx_pydot.write_dot(tree , "./multiple_trees_graphen/tree_"+ str(source) + "_" + str(destination)+ "_" + str(index))
        index = index + 1

def print_trees_with_redundant(source,destination,trees):
    index = 0
    for tree in trees:
    #    PG = nx.nx_pydot.write_dot(tree , "./multiple_trees_graphen/tree_ungekuerzt"+ str(source) + "_" + str(destination)+ "_" + str(index))
        index = index + 1


# den baum von den leafs aus ranken, dabei kriegen die leafs als ersten ihren rang 
# und parents nehmen den kleinsten rang ihrer kinder + 1 für die Kante zu ihrem kind

#damit der edp des baums als erstes durchlaufen wird müssen die ränge so verteilt werden dass
#die kanten des edps über alle anderen priorisiert werden
def rank_tree(tree , source, edp):
    nx.set_node_attributes(tree, sys.maxsize, name="rank")


    edp_edges = list()
    
    for i in range(1,len(edp)):
        edp_edges.append((edp[i-1],edp[i]))

    # initialize with all leafes
    done_nodes = [node for node in tree if len(list(nx.neighbors(tree, node))) == 0]


    for leaf in done_nodes: #initially we add rank 0 to all leafes
        tree.nodes[leaf]["rank"] = 0
        
    #es geht nicht darum dass jedes kind eines nodes einen rang hat , der erste rang ist dann auch der kleinste
    #weil dieser rang auch am schnellsten gebildet wurde
    while tree.order() != len(done_nodes):

        to_add = []
        for node in done_nodes:
            parent = get_parent_node(tree, node)
            if parent in done_nodes or parent in to_add:
                continue # eltern wurden schon gelabeled von kürzerem kind
            else: #elternknoten hat kein rang paths[source][destination] = { 'tree': tree, 'edps': edps}
                children= list(nx.neighbors(tree, parent)) #get ranks of children
                children_rank = []
                for child in children:
                    children_rank.append(tree.nodes[child]["rank"])
                tree.nodes[parent]["rank"] = min(children_rank) + 1
                to_add.append(parent)
        #endfor        
        done_nodes.extend(to_add)
    #endwihle

    #es folgt eine schleife über jeden node der die ränge der kinder so verschiebt dass die edp knoten den kleinsten rang haben
    for node in tree:

        children= list(nx.neighbors(tree, node))

        for child in children:

            if (node,child) in edp_edges:
                
                children.sort(key=lambda x: (getRank(tree, x)))

                min_rank = tree.nodes[children[0]]["rank"]

                #wenn der rang des edp pfads der kleinste ist dann muss nichts getan werden
                if min_rank == tree.nodes[child]["rank"]:
                    continue
                #wenn der edp nicht von anfang an den kleinsten rang hat
                else:
                    #der knoten des edp kriegt den kleinsten rang und alle anderen kinder kriegen ihren rang +1
                    children_without_edp_node = list()
                    
                    for element in children:

                        if element != child:
                            children_without_edp_node.append(element)
                        #endif
                    tree.nodes[child]["rank"]= min_rank

                    for element in children_without_edp_node:

                        old_rank = tree.nodes[element]["rank"]

                        tree.nodes[element]["rank"] = old_rank +1
                    #endfor
                #endif
            #endif
        #endfor
    #endfor
    

def getRank(tree, el):
    return tree.nodes[el]["rank"]

def get_parent_node(tree, node):
    pre = list(tree.predecessors(node))
    if len(pre) > 1:
        raise AssertionError("Node" + node + "has multiple  predecessors.")
    if len(pre) == 1:
        return pre[0]
    else:
        return node #Wurzel des Baumes
        
#löscht die blätter die keine direkte kante zum destination haben
#jeden knoten durchgehen  in tree der nur 1 Kante hat (also ein blatt ist) und prüfen ob dieser eine direkte kante hat zum destination in graph
def remove_redundant_paths(source, destination, tree, graph):
    nodes_to_remove = []
    for node in tree.nodes:
        #prüfen ob node den man hat ein blatt ist (genau 1 nachbarn hat)
        neighbors = list(nx.neighbors(tree, node))
        if len(neighbors) == 0 and node != source:
            #print("leaf:", node)
            #prüfen ob blatt aus dem tree im ursprungsgraphen eine direkte kante zum destination hat
            if not graph.has_edge(node,destination):
                # nur leaf mit verbindung zur destination werden
                nodes_to_remove.append(node)
                #print("adding " + str(node) + " to remove list")
    tree.remove_nodes_from(nodes_to_remove)
    #print("Remove Nodes : " , nodes_to_remove)

#beim start dieser funktion wurden alle redundanten pfade entfernt und die leaves haben im ursprungsgraph eine direkte verbindung zur destination
def connect_leaf_to_destination(tree, source, destination):
    #beinhaltet tupel aus nodes
    nodes_to_connect = []
    #nodes finden die blätter sind
    for node in tree.nodes:
        neighbors = list(nx.neighbors(tree, node))
        if len(neighbors) == 0 and node != source:
            nodes_to_connect.append((node, destination))
        elif (len(tree.nodes) == 1):
            nodes_to_connect.append((node,destination))
    #edges hinzufügen
    tree.add_edges_from(nodes_to_connect)

#hilfsfunktion die den Iterator einmal durchgeht und diesen als liste zurückgibt
def finish_iterator(neighbors_as_iterator):
    neighbors_as_list = []
    while True:
        try:
            # Iterate by calling next
            item = next(neighbors_as_iterator)
            neighbors_as_list.append(item)
            #print(item)
        except StopIteration:
            # exception will happen when iteration will over
            return neighbors_as_list

def all_edps(source, destination, graph):
    return list(nx.edge_disjoint_paths(graph, source , destination))
    #return list(nx.node_disjoint_paths(graph, source , destination))

def all_edps_greedy(source, destination, graph):
    return list(nx.edge_disjoint_paths(graph, source , destination))
    #return list(nx.node_disjoint_paths(graph, source , destination))


