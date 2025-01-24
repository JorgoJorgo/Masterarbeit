import math
import sys
import networkx as nx
import numpy as np
import itertools
import random
import time
from arborescences import *
from extra_links import *
import glob

from faces import route
from trees import get_parent_node

#global variables in this file
seed = 1
n = 10
rep = 1
k = 8
f_num = 40
samplesize=20
name = "experiment-routing"


#set global variables
def set_params(params):
    set_routing_params(params)

def set_routing_params(params):
    global seed, n, rep, k, samplesize, name, f_num
    [n, rep, k, samplesize, f_num, seed, name] = params

########################################################################################################################

def RouteWithTripleCheckpointOneTree(s,d,fails,paths):
    print()  
    
    detour_edges = []
    hops = 0
    switches = 0

    # paths[source][destination] = {
    #                     'cps': [destination],
    #                     'edps_s_to_d': [edps],
    #                     'edps_s_to_cp1':[edps],
    #                     'edps_cp1_to_cp2':[edps],
    #                     'edps_cp2_to_cp3':[edps],
    #                     'tree_cp1_to_s':tree_from_s,
    #                     'tree_cp1_to_cp2':tree_from_s,
    #                     'tree_cp3_to_cp2':tree_from_s,
    #                     'tree_cp3_to_d':tree_from_s
    #                 }
    edps_for_s_d = paths[s][d]['edps_s_to_d']

    cps = paths[s][d]['cps']

    edps_cp1_to_s = paths[s][d]['edps_cp1_to_s']
    tree_cp1_to_s  = paths[s][d]['tree_cp1_to_s']

    edps_cp1_to_cp2 = paths[s][d]['edps_cp1_to_cp2']
    tree_cp1_to_cp2  = paths[s][d]['tree_cp1_to_cp2']

    edps_cp3_to_cp2 = paths[s][d]['edps_cp3_to_cp2']
    tree_cp3_to_cp2  = paths[s][d]['tree_cp3_to_cp2']

    tree_cp3_to_d  = paths[s][d]['tree_cp3_to_d']
    edps_cp3_to_d   = paths[s][d]['edps_cp3_to_d']

    print("Routing with a checkpoint started for : ", s , " -> " , cps, " -> ",d)  
    currentNode = -1
    edpIndex = 0
    detour_edges = []
    hops = 0
    switches = 0
    
    

    print('Routing TripleCheckpointOneTree via EDPs started for ' , s , " to " , d )
    #input(" ")
    #print('EDPs:', edps_for_s_d)

    for edp in edps_for_s_d:
        #print("[Debug] Start processing EDP:", edp)

        currentNode = s
        last_node = s
        #print("[Debug] Initial currentNode:", currentNode, "last_node:", last_node)

        if edp != edps_for_s_d[len(edps_for_s_d) - 1]:
            #print("[Debug] Current EDP is not the last one in edps_for_s_d")
            #print("[RouteWithTripleCheckpointOneTree] edp:", edp)

            currentNode = edp[edpIndex]
            #print("[Debug] Updated currentNode to:", currentNode)

            # every edp is traversed until d or faulty edge
            while currentNode != d:
                #print("[Debug] Inside while loop. currentNode:", currentNode, "last_node:", last_node)

                # Check if the edge is faulty
                #print("Checke ob:",(edp[edpIndex], edp[edpIndex + 1]), "oder", (edp[edpIndex + 1], edp[edpIndex]), "in fails")
                #print("Fails:", fails)
                if ((edp[edpIndex], edp[edpIndex + 1]) in fails) or ((edp[edpIndex + 1], edp[edpIndex]) in fails):
                    #print("[Debug] Faulty edge detected between:", edp[edpIndex], "and", edp[edpIndex + 1])

                    switches += 1
                    #print("[Debug] Incremented switches to:", switches)

                    detour_edges.append((edp[edpIndex], edp[edpIndex + 1]))
                    #print("[Debug] Added to detour_edges:", detour_edges[-1])

                    tmp_node = currentNode
                    currentNode = last_node
                    last_node = tmp_node
                    hops += 1
                    #print("[Debug] Switched direction. New currentNode:", currentNode, "last_node:", last_node, "hops:", hops)
                    break

                else:
                    edpIndex += 1
                    hops += 1
                    #print("[Debug] Edge not faulty. Moving to next node. edpIndex:", edpIndex, "hops:", hops)

                    last_node = currentNode
                    currentNode = edp[edpIndex]
                    #print("[Debug] Updated currentNode to:", currentNode, "last_node to:", last_node)
                # endif
            # endwhile

            # Breaking out of the while loop potentially has 2 reasons: d reached / faulty edge detected
            print("[Debug] Exited while loop. currentNode:", currentNode, "d:", d)

            if currentNode == d:
                print('[Debug] Routing TripleCheckpointOneTree done via EDP')
                print('------------------------------------------------------')
                #print("[Debug] Returning:", False, hops, switches, detour_edges)
                return (False, hops, switches, detour_edges)
            # endif

            # Case: faulty edge detected --> traverse back to s
            while currentNode != s:
                #print("[Debug] Backtracking to source. Current node:", currentNode)

                detour_edges.append((last_node, currentNode))
                #print("[Debug] Added to detour_edges during backtracking:", detour_edges[-1])

                last_node = currentNode

                printIndex = edpIndex - 1
                #print("[Debug] printIndex during backtracking:", printIndex)

                currentNode = edp[edpIndex - 1]
                edpIndex = edpIndex - 1
                hops += 1
                #print("[Debug] Backtracked to currentNode:", currentNode, "edpIndex:", edpIndex, "hops:", hops)
            # endwhile
        # endif
    # for loop end


    # if the Structure is S -> (<= 2 CPs) -> D than the structure consists only of one EPD containing all these nodes
    # and the edp was routed in the for loop before (so if didnt reach the destination, the routing failed)
    check_all_edps_less_than_four = True
    
    for edp in edps_for_s_d:
        print("Checking:",edp)
        if len(edp)>4:
            check_all_edps_less_than_four = False

    if ( check_all_edps_less_than_four ):
        print("[RouteWithTripleCheckpointOneTree] (Special Case 1) edps_for_s_d:", edps_for_s_d)
        print("Routing failed via EDPs from S to CP because special case 1 (Structure has less than 3 CPs) ")
        print("fails:", fails)   
        print(" ")
        return (True, hops, switches, detour_edges)


    cp1 = cps[0]
    cp2 = cps[1]
    cp3 = cps[2]

    #### routing s -> cp1 via faces ####
    print("[RouteWithTripleCheckpointOneTree] Face-Routing started S(",s,") - CP1(",cp1,")")
    #input(" ")
    #from here on the structures all contain at least 5 nodes and alternating routing between faces and trees is possible
    routing_failure_faces_s_to_cp = False

    #now the first step of the routing consists of face-routing from S to CP
    routing_failure_faces_s_to_cp, hops_faces_s_to_cp, switches_faces_s_to_cp, detour_edges_faces_s_to_cp = route(s, cps[0], tree_cp1_to_s, fails)
    
    hops = hops_faces_s_to_cp + hops
    switches = switches_faces_s_to_cp + switches
    
    # Füge die Kanten aus der zweiten Liste hinzu
    for edge in detour_edges_faces_s_to_cp:
        detour_edges.append(edge)
        

    if(routing_failure_faces_s_to_cp):
        print("Routing failed via Faces from S to CP1 ")
        draw_tree_with_highlights(fails=fails,tree=tree_cp1_to_s,nodes=[s,cps[0]],showplot=False)
        print("fails:", fails)   
        print(" ")
        return (True, hops, switches, detour_edges)
    
    
    #### routing cp1 -> cp2 via tree ####
    
    print("[RouteWithTripleCheckpointOneTree] Tree-Routing started CP1(",cp1,") - CP2(",cp2,")")
    

    #the first step of the overall routing (s->cp1->cp2->cp3->d) is done
    #this first step (face routing s->cp) required a new paths object structure which does not fit into the second step (tree routing c), this structure had more keys since the face routing needed the faces
    #the object needed in the second step of the routing needs the tree & the edps of the first structure with the indices cp as the source and the destination as the destination
    
    #converted_paths[cp1][cp2]{
    #           'tree': paths[source][destination]['tree_cp1_to_cp2'],        
    #           'edps': paths[source][destination]['edps_cp1_to_cp2']
    #}

    # Create a new variable for the converted paths
    converted_paths_cp1_to_cp2 = {}
    converted_paths_cp1_to_cp2[cp1] = {}
    converted_paths_cp1_to_cp2[cp1][cp2] = {
        'tree': paths[s][d]['tree_cp1_to_cp2'],
        'edps': paths[s][d]['edps_cp1_to_cp2']
    }
    # for item1 in paths:

    #     for item2 in paths[item1]:
            
    #         #print("[RouteWithTripleCheckpointOneTree] cps:",paths[item1][item2]['cps'])
    #         if(len(paths[item1][item2]['cps'])==1):

    #             cp1_of_item = item1

    #             cp2_of_item = item2

            
    #         else:
    #             cp1_of_item = paths[item1][item2]['cps'][0]

    #             cp2_of_item = paths[item1][item2]['cps'][1]
            
    #         if cp1_of_item not in converted_paths_cp1_to_cp2:
                
    #             converted_paths_cp1_to_cp2[cp1_of_item] = {}
                
    #         converted_paths_cp1_to_cp2[cp1_of_item][cp2_of_item]= {
    #             'tree': paths[item1][item2]['tree_cp1_to_cp2'],
    #             'edps': paths[item1][item2]['edps_cp1_to_cp2']
    #         }

    #print("CP1:", cp1)        
    #print("CP2:", cp2)
    #print("fails:", fails)     
    print("ConvertedPathsCP1toCP2 tree",converted_paths_cp1_to_cp2[cp1][cp2]['tree'].nodes) 
    print("ConvertedPathsCP1toCP2 edps",converted_paths_cp1_to_cp2[cp1][cp2]['edps'])        
    routing_failure_tree_cp1_to_cp2, hops_tree_cp1_to_cp2, switches_tree_cp1_to_cp2, detour_edges_tree_cp1_to_cp2 = RouteOneTree_CP(cp1,cp2,fails,converted_paths_cp1_to_cp2)
    
    hops = hops + hops_tree_cp1_to_cp2
    switches = switches + switches_tree_cp1_to_cp2
    
    #draw_tree_with_highlights(fails=fails,tree=converted_paths_cp1_to_cp2[cp1][cp2]['tree'],nodes=[cp1,cp2],showplot=True)
    # Füge die Kanten aus der ersten Liste hinzu
    for edge in detour_edges_tree_cp1_to_cp2:
        detour_edges.append(edge)

    if(routing_failure_tree_cp1_to_cp2):
        print("Routing failed via Tree from CP1 to CP2 ")
        draw_tree_with_highlights(fails=fails,tree=converted_paths_cp1_to_cp2[cp1][cp2]['tree'],nodes=[cp1,cp2],showplot=False)
        print("fails:", fails)   
        print(" ")
        return (True, hops, switches, detour_edges)    
    

    #input(" ")
    
    ##### routing cp2->cp3 via faces ####
    print("[RouteWithTripleCheckpointOneTree] Face-Routing started CP2(",cp2,") - CP3(",cp3,")")
    

    routing_failure_faces_cp2_to_cp3 = False
    #print("CP2:", cp2)        
    #print("CP3:", cp3)
    #print("fails:", fails)     
    print("CP2toCP3 tree",tree_cp3_to_cp2.nodes) 
    #draw_tree_with_highlights(tree_cp3_to_cp2,[cp2,cp3],fails)

    #now the first step of the routing consists of face-routing from S to CP
    routing_failure_faces_cp2_to_cp3, hops_faces_cp2_to_cp3, switches_faces_cp2_to_cp3, detour_edges_faces_cp2_to_cp3 = route(cp2, cp3, tree_cp3_to_cp2, fails)
    
    hops = hops_faces_cp2_to_cp3 + hops
    switches = switches_faces_cp2_to_cp3 + switches
    
    # Füge die Kanten aus der zweiten Liste hinzu
    for edge in detour_edges_faces_cp2_to_cp3:
        detour_edges.append(edge)
        

    if(routing_failure_faces_cp2_to_cp3):
        print("Routing failed via Faces from CP2 to CP3 ")
        draw_tree_with_highlights(fails=fails,tree=tree_cp3_to_cp2,nodes=[cp2,cp3],showplot=False)
        print("fails:", fails)   
        print(" ")
        return (True, hops, switches, detour_edges)
    
    #input(" ")
    ##### routing cp3->d via tree ####

    print("[RouteWithTripleCheckpointOneTree] Tree-Routing started CP3(",cp3,") - D(",d,")")
    

    converted_paths_cp3_to_d = {}  
    converted_paths_cp3_to_d[cp3] = {}  
    converted_paths_cp3_to_d[cp3][d]= {
        'tree': paths[s][d]['tree_cp3_to_d'],
        'edps': paths[s][d]['edps_cp3_to_d']
        }
    
    #print("CP3:", cp3)        
    #print("D:", d)
    #print("fails:", fails)     
    print("ConvertedPathsCP3toD tree",converted_paths_cp3_to_d[cp3][d]['tree'].nodes) 
    print("ConvertedPathsCP3toD edps",converted_paths_cp3_to_d[cp3][d]['edps']) 
    #draw_tree_with_highlights(fails=fails,tree=converted_paths_cp3_to_d[cp3][d]['tree'],nodes=[cp3,d],showplot=True)
    routing_failure_tree_cp2_to_cp3, hops_tree_cp2_to_cp3, switches_tree_cp2_to_cp3, detour_edges_tree_cp2_to_cp3 = RouteOneTree_CP(cp3,d,fails,converted_paths_cp3_to_d)
    
    hops = hops + hops_tree_cp2_to_cp3
    switches = switches + switches_tree_cp2_to_cp3


    # Füge die Kanten aus der ersten Liste hinzu
    for edge in detour_edges_tree_cp2_to_cp3:
        detour_edges.append(edge)
    
    #input(" ")

    if(routing_failure_tree_cp2_to_cp3):
        print("Routing failed via Tree from CP3 to D ")
        draw_tree_with_highlights(fails=fails,tree=converted_paths_cp3_to_d[cp3][d]['tree'],nodes=[cp3,d],showplot=False)
        print("fails:", fails)   
        print(" ")
        return (True, hops, switches, detour_edges)    
    


    #if all parts were successfull we got to the destination
    print("Routing succesful with the Checkpoint")
    print('------------------------------------------------------')
    print(" ")
    return (False, hops, switches, detour_edges)

########################################################################################################################

def RouteWithTripleCheckpointMultipleTrees(s,d,fails,paths):
    print()  
    
    detour_edges = []
    hops = 0
    switches = 0

    edps_for_s_d = paths[s][d]['edps_s_to_d']

    cps = paths[s][d]['cps']

    #das sind 2 variablen die jeweils 1 baum beinhalten
    #beim facerouting wurden alle bäume zu 1 baum kombiniert, weil dieser ja eh nur mit facerouting funktioinert
    trees_cp1_to_s  = paths[s][d]['trees_cp1_to_s']
    trees_cp3_to_cp2  = paths[s][d]['trees_cp3_to_cp2']

    print("Routing with a checkpoint started for : ", s , " -> " , cps, " -> ",d)  
    currentNode = -1
    edpIndex = 0
    detour_edges = []
    hops = 0
    switches = 0
    
    

    print('Routing TripleCheckpointMultipleTrees via EDPs started for ' , s , " to " , d )

    for edp in edps_for_s_d:

        currentNode = s
        last_node = s

        if edp != edps_for_s_d[len(edps_for_s_d) - 1]:

            currentNode = edp[edpIndex]

            # every edp is traversed until d or faulty edge
            while currentNode != d:

                # Check if the edge is faulty
                if ((edp[edpIndex], edp[edpIndex + 1]) in fails) or ((edp[edpIndex + 1], edp[edpIndex]) in fails):

                    switches += 1

                    detour_edges.append((edp[edpIndex], edp[edpIndex + 1]))

                    tmp_node = currentNode
                    currentNode = last_node
                    last_node = tmp_node
                    hops += 1
                    break

                else:
                    edpIndex += 1
                    hops += 1

                    last_node = currentNode
                    currentNode = edp[edpIndex]
                # endif
            # endwhile

            # Breaking out of the while loop potentially has 2 reasons: d reached / faulty edge detected

            if currentNode == d:
                print('[Debug] Routing TripleCheckpointMultipleTrees done via EDP')
                print('------------------------------------------------------')
                return (False, hops, switches, detour_edges)
            # endif

            # Case: faulty edge detected --> traverse back to s
            while currentNode != s:

                detour_edges.append((last_node, currentNode))

                last_node = currentNode

                printIndex = edpIndex - 1

                currentNode = edp[edpIndex - 1]
                edpIndex = edpIndex - 1
                hops += 1
            # endwhile
        # endif
    # for loop end


    # if the Structure is S -> (<= 2 CPs) -> D than the structure consists only of one EPD containing all these nodes
    # and the edp was routed in the for loop before (so if didnt reach the destination, the routing failed)
    check_all_edps_less_than_four = True
    
    for edp in edps_for_s_d:
        #print("Checking:",edp)
        if len(edp)>4:
            check_all_edps_less_than_four = False

    if ( check_all_edps_less_than_four ):
        print("[RouteWithTripleCheckpointMultipleTrees] (Special Case 1) edps_for_s_d:", edps_for_s_d)
        print("Routing failed via EDPs from S to CP because special case 1 (Structure has less than 3 CPs) ")
        print("fails:", fails)   
        print(" ")
        return (True, hops, switches, detour_edges)


    cp1 = cps[0]
    cp2 = cps[1]
    cp3 = cps[2]

    #### routing s -> cp1 via faces ####

    print("[RouteWithTripleCheckpointMultipleTrees] Face-Routing started S(",s,") - CP1(",cp1,")")

    #from here on the structures all contain at least 5 nodes and alternating routing between faces and trees is possible

    #now the first step of the routing consists of face-routing from S to CP
    routing_failure_faces_s_to_cp1, hops_faces_s_to_cp1, switches_faces_s_to_cp1, detour_edges_faces_s_to_cp1 = route(s, cps[0], trees_cp1_to_s, fails)
    
    hops = hops_faces_s_to_cp1 + hops
    switches = switches_faces_s_to_cp1 + switches
    
    # Füge die Kanten aus der zweiten Liste hinzu
    for edge in detour_edges_faces_s_to_cp1:
        detour_edges.append(edge)
        

    if(routing_failure_faces_s_to_cp1):
        print("Routing failed via Faces from S to CP1 ")  
        draw_tree_with_highlights(fails=fails,tree=trees_cp1_to_s,nodes=[s,cps[0]],showplot=False)
        
        print("fails:", fails)   
        print(" ")
        return (True, hops, switches, detour_edges)
    
    
    #### routing cp1 -> cp2 via tree ####
    
    print("[RouteWithTripleCheckpointMultipleTrees] Tree-Routing started CP1(",cp1,") - CP2(",cp2,")")

    # Create a new variable for the converted paths
    converted_paths_cp1_to_cp2 = {}
    converted_paths_cp1_to_cp2[cp1] = {}
    converted_paths_cp1_to_cp2[cp1][cp2] = {
        'trees': paths[s][d]['trees_cp1_to_cp2'],
        'edps': paths[s][d]['edps_cp1_to_cp2']
    }
    tree_index = 0
    for tree in converted_paths_cp1_to_cp2[cp1][cp2]['trees']:
        print("ConvertedPathsCP1toCP2 tree (",tree_index,")",tree.nodes) 
        tree_index +=1
    
    print("ConvertedPathsCP1toCP2 edps",converted_paths_cp1_to_cp2[cp1][cp2]['edps'])    

    routing_failure_trees_cp1_to_cp2 = False
    routing_failure_trees_cp1_to_cp2, hops_trees_cp1_to_cp2, switches_trees_cp1_to_cp2, detour_edges_trees_cp1_to_cp2 = RouteMultipleTrees(cp1,cp2,fails,converted_paths_cp1_to_cp2)
    
    hops = hops + hops_trees_cp1_to_cp2
    switches = switches + switches_trees_cp1_to_cp2
    
    # Füge die Kanten aus der ersten Liste hinzu
    for edge in detour_edges_trees_cp1_to_cp2:
        detour_edges.append(edge)

    if(routing_failure_trees_cp1_to_cp2):
        print("Routing failed via Tree from CP1 to CP2 ")
        #for tree in paths[s][d]['trees_cp1_to_cp2']:
        draw_multipletree_with_highlights(fails=fails,trees=paths[s][d]['trees_cp1_to_cp2'],nodes=[cp1,cp2],showplot=False,einzeln=False)
        print("fails:", fails)   
        print(" ")
        return (True, hops, switches, detour_edges)    
    

    #input(" ")
    
    ##### routing cp2->cp3 via faces ####
    print("[RouteWithTripleCheckpointOneTree] Face-Routing started CP2(",cp2,") - CP3(",cp3,")")
    

    routing_failure_faces_cp2_to_cp3 = False
       
    #now the first step of the routing consists of face-routing from S to CP
    routing_failure_faces_cp2_to_cp3, hops_faces_cp2_to_cp3, switches_faces_cp2_to_cp3, detour_edges_faces_cp2_to_cp3 = route(cp2, cp3, trees_cp3_to_cp2, fails)
    
    hops = hops_faces_cp2_to_cp3 + hops
    switches = switches_faces_cp2_to_cp3 + switches
    
    # Füge die Kanten aus der zweiten Liste hinzu
    for edge in detour_edges_faces_cp2_to_cp3:
        detour_edges.append(edge)
        

    if(routing_failure_faces_cp2_to_cp3):
        print("Routing failed via Faces from CP2 to CP3 ")
        
        draw_tree_with_highlights(fails=fails,tree=trees_cp3_to_cp2,nodes=[cp2,cp3],showplot=False)
        print("fails:", fails)   
        print(" ")
        return (True, hops, switches, detour_edges)
    
    #input(" ")
    ##### routing cp3->d via tree ####

    print("[RouteWithTripleCheckpointMultipleTrees] Tree-Routing started CP3(",cp3,") - D(",d,")")
    

    converted_paths_cp3_to_d = {}  
    converted_paths_cp3_to_d[cp3] = {}  
    converted_paths_cp3_to_d[cp3][d]= {
        'trees': paths[s][d]['trees_cp3_to_d'],
        'edps': paths[s][d]['edps_cp3_to_d']
        }
    
    routing_failure_trees_cp2_to_cp3, hops_trees_cp2_to_cp3, switches_trees_cp2_to_cp3, detour_edges_trees_cp2_to_cp3 = RouteMultipleTrees(cp3,d,fails,converted_paths_cp3_to_d)
    
    hops = hops + hops_trees_cp2_to_cp3
    switches = switches + switches_trees_cp2_to_cp3


    # Füge die Kanten aus der ersten Liste hinzu
    for edge in detour_edges_trees_cp2_to_cp3:
        detour_edges.append(edge)
    
    #input(" ")

    if(routing_failure_trees_cp2_to_cp3):
        print("Routing failed via Tree from CP3 to D ")
        #for tree in converted_paths_cp3_to_d[cp3][d]['trees']:
        #draw_tree_with_highlights(fails=fails,tree=converted_paths_cp3_to_d[cp3][d]['trees'],nodes=[cp3,d],showplot=False)
        draw_multipletree_with_highlights(fails=fails,trees=converted_paths_cp3_to_d[cp3][d]['trees'],nodes=[cp3,d],showplot=False, einzeln=False)
        print("fails:", fails)   #    
        print(" ")
        return (True, hops, switches, detour_edges)    
    


    #if all parts were successfull we got to the destination
    print("Routing succesful with the Checkpoint")
    print('------------------------------------------------------')
    print(" ")
    return (False, hops, switches, detour_edges)


########################################################################################################################

def RouteWithOneCheckpointMultipleTrees(s,d,fails,paths):
    print()  
    
    detour_edges = []
    hops = 0
    switches = 0
    
    cp = paths[s][d]['cp']
    edps_cp_to_s = paths[s][d]['edps_cp_to_s']
    trees_cp_to_d  = paths[s][d]['trees_cp_to_d']
    edps_cp_to_d   = paths[s][d]['edps_cp_to_d']
    edps_s_to_d = paths[s][d]['edps_s_to_d']
    trees_cp_to_s = paths[s][d]['trees_cp_to_s']

    #before routing through the structure, the edps are traversed
    print("Routing with a checkpoint started for : ", s , " -> " , cp, " -> ",d)  
    currentNode = -1
    edpIndex = 0
    detour_edges = []
    hops = 0
    switches = 0
    
    edps_for_s_d = edps_s_to_d

    print('Routing MTCP via EDPs started for ' , s , "-", cp , "-" , d )
    
    for edp in edps_for_s_d:
        
        currentNode = s
        last_node = s 

        if edp != edps_for_s_d[len(edps_for_s_d) -1]:

            currentNode = edp[edpIndex]


            #every edp is traversed until d or faulty edge
            while (currentNode != d):


                #since the structure of the edps consists of a line a->b->c-> ... -> n the direct neighbor is checked
                if (edp[edpIndex], edp[edpIndex +1]) in fails or (edp[edpIndex +1], edp[edpIndex]) in fails:
                
                    switches += 1

                    
                    detour_edges.append( (edp[edpIndex], edp[edpIndex +1]) )

                    
                    tmp_node = currentNode 
                    currentNode = last_node 
                    last_node = tmp_node
                    hops += 1
                    break

                else :
                    edpIndex += 1
                    hops += 1
                    last_node = currentNode 
                    currentNode = edp[edpIndex]
                #endif

            #endwhile

            # breaking out of the while loop potentially has 2 reasons : d reached / faulty edge detected


            if currentNode == d : 
                print('Routing MultipleTreesCP done via EDP')
                print('------------------------------------------------------')
                return (False, hops, switches, detour_edges)
            #endif
            
            # case : faulty edge detected --> traverse back to s
            while currentNode != s: 
                detour_edges.append( (last_node,currentNode) )

                last_node = currentNode 
                currentNode = edp[edpIndex-1] 
                edpIndex = edpIndex-1
                hops += 1

            #endwhile
        #endif

    #endfor

    #if the routing s->d via edps failed, the routing is partitioned by facerouting s->cp and treerouting cp->d
    
    #starting with the facerouting s->cp

    routing_failure_faces = False

    #now the first step of the routing consists of face-routing from S to CP
    routing_failure_faces, hops_faces, switches_faces, detour_edges_faces = route(s, cp, trees_cp_to_s, fails)

    hops = hops + hops_faces
    switches = switches + switches_faces
    
    for edge in detour_edges_faces:
        detour_edges.append(edge)

    if(routing_failure_faces):
        print("Routing failed via Faces from S to CP ")
        print(" ")
        return (True, hops, switches, detour_edges)
    else:
        if(len(edps_s_to_d)==1):
            print("[RouteOneCheckpointMult] edps_s_to_d:", edps_s_to_d)
            return (False, hops, switches, detour_edges)
    
    # face routing succesfull s->cp, next step tree routing cp->d
    # but for the old routing function of multipletrees to  function, the paths structure needs to be converted
    #new structure:
   # paths[source][destination] = {
   #                                'cp': cp, 
   #                                'edps_cp_to_s': edps_cp_to_s,
   #                                'trees_cp_to_d': tree_cp_to_d, 
   #                                'edps_cp_to_d': edps_cp_to_d,
   #                                'edps_s_to_d': edps,
   #                                'trees_cp_to_s':tree_cp_to_s
   #                               }
    
    #old structure:
    #paths[source][destination] = {
    #                             'trees': all trees from s->d,
    #                             'edps': all edps from s->
    #                              }

    
    # Create a new variable for the converted paths
    converted_paths = {}

    #the first step of the overall routing (s->cp->d) is done
    #this first step (face routing s->cp) required a new paths object structure which does not fit into the second step (tree routing c), this structure had more keys since the face routing needed the faces
    #the object needed in the second step of the routing needs the tree & the edps of the first structure with the indices cp as the source and the destination as the destination
    
    #converted_paths[cp][destination]{
    #           'tree': paths[source][destination]['tree_cp_to_d'],        
    #           'edps': paths[source][destination]['edps_cp_to_d']
    #}
    for item1 in paths:

        for item2 in paths[item1]:
            
            checkpoint_of_item = paths[item1][item2]['cp']
            
            if checkpoint_of_item not in converted_paths:
                
                converted_paths[checkpoint_of_item] = {}
                
            converted_paths[checkpoint_of_item] [item2]= {
                'trees': paths[item1][item2]['trees_cp_to_d'],
                'edps': paths[item1][item2]['edps_cp_to_d']
            }

    #after that the routing continues from CP to D using the tree-routing
    routing_failure_tree, hops_tree, switches_tree, detour_edges_tree = RouteMultipleTrees(cp,d,fails,converted_paths)

    hops = hops + hops_tree
    switches = switches + switches_tree

    for edge in detour_edges_tree:
        detour_edges.append(edge)
    
    if(routing_failure_tree):
        print("Routing failed via MultipleTrees Tree from CP to D ")
        print(" ")
        return (True, hops, switches, detour_edges)    
    
        
    print("Routing MultipleTrees succesful with the Checkpoint")
    print('------------------------------------------------------')
    print(" ")
    return (False, hops, switches, detour_edges)

########################################################################################################################
#paths structure  for routing with a checkpoint: 
# paths[source][destination] = {
#                                                 'cp': cp,
#                                                 'faces_cp_to_s': faces_cp_to_s, 
#                                                 'edps_cp_to_s': edps_cp_to_s,
#                                                 'tree_cp_to_d': tree_cp_to_d, 
#                                                 'edps_cp_to_d': edps_cp_to_d,
#                                                 'edps_s_to_d': edps,
#                                                 'tree_planar_embedding_cp_to_s':tree_planar_embedding_cp_to_s,
#                                                 'tree_cp_to_s':tree_cp_to_s
#                                             }

# the routing with ONE checkpoint and with ONE tree first tries to route using 
# the face-routing from s -> cp and after that the tree-routing from cp -> d
def RouteWithOneCheckpointOneTree(s,d,fails,paths):
    
    print()  
    
    detour_edges = []
    hops = 0
    switches = 0
    
    cp = paths[s][d]['cp']
    faces_cp_to_s  = paths[s][d]['faces_cp_to_s']
    edps_cp_to_s = paths[s][d]['edps_cp_to_s']
    tree_cp_to_d  = paths[s][d]['tree_cp_to_d']
    edps_cp_to_d   = paths[s][d]['edps_cp_to_d']
    edps_s_to_d = paths[s][d]['edps_s_to_d']
    #tree_planar_embedding_cp_to_s = paths[s][d]['tree_planar_embedding_cp_to_s']
    tree_cp_to_s = paths[s][d]['tree_cp_to_s']
    #print("EDPS s to d :", edps_s_to_d)
    #before routing through the structure, the edps are traversed
    print("Routing with a checkpoint started for : ", s , " -> " , cp, " -> ",d)  
    currentNode = -1
    edpIndex = 0
    detour_edges = []
    hops = 0
    switches = 0
    
    edps_for_s_d = edps_s_to_d

    print('Routing OTCP via EDPs started for ' , s , " to " , d )
    #print('EDPs:', edps_for_s_d)
    for edp in edps_for_s_d:
        
        currentNode = s
        last_node = s 

        if edp != edps_for_s_d[len(edps_for_s_d) -1]:

            currentNode = edp[edpIndex]


            #every edp is traversed until d or faulty edge
            while (currentNode != d):


                #since the structure of the edps consists of a line a->b->c-> ... -> n the direct neighbor is checked
                if (edp[edpIndex], edp[edpIndex +1]) in fails or (edp[edpIndex +1], edp[edpIndex]) in fails:
                
                    switches += 1

                    
                    detour_edges.append( (edp[edpIndex], edp[edpIndex +1]) )

                    
                    tmp_node = currentNode 
                    currentNode = last_node 
                    last_node = tmp_node
                    hops += 1
                    break

                else :
                    edpIndex += 1
                    hops += 1
                    last_node = currentNode 
                    currentNode = edp[edpIndex]
                #endif

            #endwhile

            # breaking out of the while loop potentially has 2 reasons : d reached / faulty edge detected


            if currentNode == d : 
                print('Routing OneTreeCP done via EDP')
                print('------------------------------------------------------')
                return (False, hops, switches, detour_edges)
            #endif
            
            # case : faulty edge detected --> traverse back to s
            while currentNode != s: 
                detour_edges.append( (last_node,currentNode) )

                last_node = currentNode 
                
                printIndex = edpIndex-1
                
                
                # print("Source : ", s , " Destination : ", d)
                # print("Edp : ", edp)
                # print("EdpIndex-1 : ", printIndex)
                # print("edp[edpIndex-1] : ", edp[edpIndex-1])
                # print(" ")
                
                
                currentNode = edp[edpIndex-1] 
                edpIndex = edpIndex-1
                hops += 1

            #endwhile
        #endif

    #endfor

    routing_failure_faces = False

    #now the first step of the routing consists of face-routing from S to CP
    routing_failure_faces, hops_faces, switches_faces, detour_edges_faces = route(s, cp, tree_cp_to_s, fails)

    hops = hops_faces + hops
    switches = switches_faces

    
    # Füge die Kanten aus der zweiten Liste hinzu
    for edge in detour_edges_faces:
        detour_edges.append(edge)

    if(routing_failure_faces):
        print("Routing failed via Faces from S to CP ")
        print(" ")
        return (True, hops_faces, switches_faces, detour_edges_faces)
    
    #since the routing for the trees was build prior to routing of the faces, the paths structure has changed
    #therefore the new paths structure needs to be converted to the old structure
    
    #new structure:
   # paths[source][destination] = {
#                                                 'cp': cp,
#                                                 'faces_cp_to_s': faces_cp_to_s, 
#                                                 'edps_cp_to_s': edps_cp_to_s,
#                                                 'tree_cp_to_d': tree_cp_to_d, 
#                                                 'edps_cp_to_d': edps_cp_to_d,
#                                                 'edps_s_to_d': edps,
#                                                 'tree_planar_embedding_cp_to_s':tree_planar_embedding_cp_to_s,
#                                                 'tree_cp_to_s':tree_cp_to_s
#                                             }
    
    #old structure:
    #paths[source][destination] = {
    #                               'tree': tree,
    #                               'edps': edps
    #                              }

    
    # Create a new variable for the converted paths
    converted_paths = {}

    #the first step of the overall routing (s->cp->d) is done
    #this first step (face routing s->cp) required a new paths object structure which does not fit into the second step (tree routing c), this structure had more keys since the face routing needed the faces
    #the object needed in the second step of the routing needs the tree & the edps of the first structure with the indices cp as the source and the destination as the destination
    
    #converted_paths[cp][destination]{
    #           'tree': paths[source][destination]['tree_cp_to_d'],        
    #           'edps': paths[source][destination]['edps_cp_to_d']
    #}
    for item1 in paths:

        for item2 in paths[item1]:
            
            checkpoint_of_item = paths[item1][item2]['cp']
            
            if checkpoint_of_item not in converted_paths:
                
                converted_paths[checkpoint_of_item] = {}
                
            converted_paths[checkpoint_of_item] [item2]= {
                'tree': paths[item1][item2]['tree_cp_to_d'],
                'edps': paths[item1][item2]['edps_cp_to_d']
            }
            
                 
    #after that the routing continues from CP to D using the tree-routing
    routing_failure_tree, hops_tree, switches_tree, detour_edges_tree = RouteOneTree_CP(cp,d,fails,converted_paths)
    
    hops = hops + hops_tree
    switches = switches+ switches_tree
    

    # Füge die Kanten aus der ersten Liste hinzu
    for edge in detour_edges_tree:
        detour_edges.append(edge)
    
    if(routing_failure_tree):
        print("Routing failed via Tree from CP to D ")
        print(" ")
        return (True, hops, switches, detour_edges)    
    
    #if both parts of the routing did not fail then the results of each one need to be combined
    
    
        
    print("Routing succesful with the Checkpoint")
    print('------------------------------------------------------')
    print(" ")
    return (False, hops, switches, detour_edges)

#just for simplicity of debugging the onetreecp routing has its on function
#but it is the same as the routeonetree
def RouteOneTree_CP (s,d,fails,paths):
    
    #print("RouteOneTreeCP] Checkpoint 0")
    if s != d :
        #print("RouteOneTreeCP] Checkpoint 1")
        currentNode = -1
        edpIndex = 0
        detour_edges = []
        hops = 0
        switches = 0
        tree = paths[s][d]['tree']
        edps_for_s_d = paths[s][d]['edps']

        print('Routing started for ' , s , " to " , d )
        print("[RouteOneTreeCP] EDPs:",edps_for_s_d)
        #als erstes anhand der EDPs (außer dem längsten, also dem letzten) versuchen zu routen
        for edp in edps_for_s_d:
            
            currentNode = s
            last_node = s 

            if len(edp) == 2: #sonderfall wenn der edp nur 2 lang ist
                if (edp[0], edp[1]) in fails or (edp[1], edp[0]) in fails:
                    continue
                else:
                    switches += 1
                    detour_edges.append( (edp[0], edp[1]) )
                    hops +=1
                    print('Routing done via EDP')
                    print('------------------------------------------------------')
                    return (False, hops, switches, detour_edges)


            if edp != edps_for_s_d[len(edps_for_s_d) -1]:

                currentNode = edp[edpIndex]


                #jeder EDP wird so weit durchlaufen bis man mit dem currentNode zum Ziel kommt oder man auf eine kaputte Kante stößt
                while (currentNode != d):


                    #man prüft ob die nächste Kante im EDP kaputt ist so, indem man guckt ob eine Kante vom currentNode edp[edpIndex] zum nächsten Node im EDP edp[edpIndex+1] in Fails ist
                    #dies beruht auf lokalen Informationen, da EDPs nur eine eingehende Kante haben ( auf der das Paket ankommt ) und eine ausgehende Kante (auf der das Paket nicht ankommt)
                    if (edp[edpIndex], edp[edpIndex +1]) in fails or (edp[edpIndex +1], edp[edpIndex]) in fails:
                        

                        #wenn man auf eine fehlerhafte Kante stößt dann wechselt man den Pfad
                        switches += 1

                        #die kanten die wir wieder zurückgehen sind die kanten die wir schon in dem edp gelaufen sind
                        detour_edges.append( (edp[edpIndex], edp[edpIndex +1]) )

                        #wir fangen beim neuen edp ganz am anfang an
                        tmp_node = currentNode #und gehen eine Kante hoch, also den edp zurück
                        currentNode = last_node #das "rückwärts den edp gehen" kann so gemacht werden, da die pakete so nur über den port gehen müssen über den sie reingekommen sind
                        last_node = tmp_node
                        hops += 1
                        break

                    else :#wenn die kante die man gehen will inordnung ist, die kante gehen und zum nächsten knoten schalten
                        edpIndex += 1
                        hops += 1
                        last_node = currentNode 
                        currentNode = edp[edpIndex] #man kann hier currentnode direkt so setzen, da es im edp für jeden knoten jeweils 1 ausgehende
                                                    #und genau eine eingehende Kante gibt
                    #endif

                #endwhile

                #nun gibt es 2 Möglichkeiten aus denen die while-Schleife abgebrochen wurde : Ziel erreicht / EDP hat kaputte Kante 


                if currentNode == d : #wir haben die destination mit einem der edps erreicht
                    print('Routing done via EDP')
                    print('------------------------------------------------------')
                    return (False, hops, switches, detour_edges)
                #endif
                
                #wenn man hier angelangt ist, dann bedeutet dies, dass die while(currentNode != d) beendet wurde weil man auf eine kaputte kante gestoßen ist 
                #und dass man nicht an der destination angekommen ist, daher muss man jetzt an die source zurück um den nächsten edp zu starten
                while currentNode != s: #hier findet die Rückführung statt
                    detour_edges.append( (last_node,currentNode) )

                    last_node = currentNode #man geht den edp so weit hoch bis man an der source ist
                    
                    printIndex = edpIndex-1
                    
                    
                    print("Source : ", s , " Destination : ", d)
                    print("Edp : ", edp)
                    print("EdpIndex-1 : ", printIndex)
                    print("edp[edpIndex-1] : ", edp[edpIndex-1])
                    print(" ")
                    
                    
                    currentNode = edp[edpIndex-1] #man kann auch hier direkt den edp index verwenden da man genau 1 eingehende kante hat
                    edpIndex = edpIndex-1
                    hops += 1

                #endwhile
            #endif

        #endfor

        # wenn wir es nicht geschafft haben anhand der edps allein zum ziel zu routen dann geht es am längsten edp weiter
        print('Routing via EDPs FAILED')
        
        currentNode = s
        print("Routing via Tree started")
        last_node = currentNode


        while(currentNode != d):#in dieser Schleife findet das Routing im Tree statt
                                #die idee hinter dieser schleife ist ein großes switch-case
                                #bei dem man je nach eingehenden und funktionierenden ausgehenden ports switcht
                                #nach jedem schritt den man im baum geht folgt die prüfung ob man schon am ziel angekommen ist


            #kommt das paket von einer eingehenden kante an dann wird der kleinste rang der kinder gewählt
            #denn man war noch nicht an diesem node
            if last_node == get_parent_node(tree,currentNode) or last_node == currentNode:

                #suche das kind mit dem kleinsten  rang

                children = []
                #es werden alle Kinder gespeichert zu denen der jetzige Knoten einen Verbindung hat und sortiert nach ihren Rängen
                out_edges_with_fails = tree.out_edges(currentNode)
                out_edges = []
                for edge in out_edges_with_fails:
                    if edge in fails or tuple(reversed(edge)) in fails:
                        continue
                    else: 
                        out_edges.append(edge)
                    #endif
                #endfor
                for nodes in out_edges:
                    children.append(nodes[1])
                #endfor
                children.sort(key=lambda x: (getRank(tree, x)))


                if len(children) >  0 : #wenn es kinder gibt, zu denen die Kanten nicht kaputt sind
                    #setze lastnode auf currentnode
                    #setze current node auf das kind mit dem kleinesten rang
                    #dadurch "geht man" die kante zum kind
                    last_node = currentNode
                    currentNode = children[0]
                    hops += 1
                   

                else: #wenn alle Kanten zu den Kindern kaputt sind dann ist man fertig wenn man an der source ist oder man muss eine kante hoch
                    if currentNode == s: 
                        break; #das routing ist gescheitert
                    #endif


                    #man nimmt die eingehende kante des currentnode und "geht eine stufe hoch"
                    hops += 1
                    detour_edges.append( (currentNode, last_node) )
                    last_node = currentNode
                    currentNode = get_parent_node(tree,currentNode)

                #endif
            #endif



            children_of_currentNode = []

            for nodes in tree.out_edges(currentNode):
                    children_of_currentNode.append(nodes[1])
            #endfor

            #wenn das Paket nicht aus einer eingehenden Kante kommt, dann muss es aus einer ausgehenden kommen
            #dafür muss man den Rang des Kindes bestimmen von dem das Paket kommt
            #das Kind mit dem nächsthöheren Rang suchen
            if last_node in children_of_currentNode:
            
                #alle funktionierenden Kinder finden
                children = []
                out_edges_with_fails = tree.out_edges(currentNode)
                out_edges = []
                for edge in out_edges_with_fails:
                    if edge in fails or tuple(reversed(edge)) in fails:
                        continue
                        
                    else: 
                        out_edges.append(edge)
                    #endif

                #endfor
                for nodes in out_edges:
                    children.append(nodes[1])
                #endfor
                children.sort(key=lambda x: (getRank(tree, x)))

                

                #wenn es Funktionierende Kinder gibt dann muss man das Kind suchen mit dem nächstgrößeren Rang
                if len(children) > 0: 
                    #prüfen ob es noch kinder gibt mit größerem rang , also ob es noch zu durchlaufene kinder gibt
                    

                    #welchen index hat das kind nach seinem "rank" in der sortierten liste
                    index_of_last_node = children.index(last_node) if last_node in children else -1 
                
                    #alle  kinder ohne das wo das paket herkommt
                    children_without_last = [a for a in children if a != last_node] 

                    

                    #es gibt keine möglichen kinder mehr und man ist an der Source
                    #dann ist das Routing fehlgeschlagen
                    if len(children_without_last) == 0 and currentNode == s : 
                        break;

                    #Sonderfall (noch unklar ob nötig)
                    #wenn man aus einem Kind kommt, zu dem die Kante fehlerhaft ist
                    #man nimmt trotzdem das nächste Kind
                    elif index_of_last_node == -1:
                        
                        hops += 1
                        last_node = currentNode
                        currentNode = children[0]


                    #das kind wo das paket herkommt hatte den höchsten rang der kinder, also das letztmögliche
                    #daher muss man den Baum eine Stufe hoch
                    elif index_of_last_node == len(children)-1: 
                        
                        if currentNode != s: #man muss eine stufe hoch gehen
                            hops += 1
                            detour_edges.append( (currentNode, last_node) )
                            last_node = currentNode
                            currentNode = get_parent_node(tree,currentNode)
                        else:#sonderfall wenn man an der Source ist dann ist das Routing gescheitert
                            break;

                    #es gibt noch mindestens 1 Kind mit höherem Rang
                    elif index_of_last_node < len(children)-1 : 
                        
                        #wenn ja dann nimm das Kind mit dem nächst größeren Rang aus der sortierten Children Liste
                        hops += 1
                        last_node = currentNode
                        currentNode = children[index_of_last_node+1]


                    #es gibt keine kinder mehr am currentnode
                    else: 
                        
                        #wenn nein dann setze currentnode auf den parent
                        hops += 1
                        detour_edges.append( (currentNode, last_node) )
                        last_node = currentNode
                        currentNode = get_parent_node(tree,currentNode)
                    #endif

                #wenn es keine funktionierenden Kinder gibt dann geht man eine Stufe hoch
                else: 
                    detour_edges.append( (currentNode, last_node) )
                    hops += 1
                    
                    last_node = currentNode
                    currentNode = get_parent_node(tree,currentNode)
                   
                #endif
            
                
        #endwhile

        #hier kommt man an wenn die while schleife die den tree durchläuft "gebreakt" wurde und man mit dem tree nicht zum ziel gekommen ist
        #oder wenn die bedingung nicht mehr gilt (currentNode != d) und man das ziel erreicht hat

        if currentNode == d : #wir haben die destination mit dem tree erreicht
            print('Routing done via Tree')
            print('------------------------------------------------------')
            return (False, hops, switches, detour_edges)
        #endif
        
        print('Routing via Tree failed')
        print('------------------------------------------------------')
        return (True, hops, switches, detour_edges)
    else: 
        print("RouteOneTreeCP] Checkpoint 99")
        return (True, 0, 0, [])
        
        
########################### HELPER FUNCTIONS FACES ###################################################################

# Helper function to find the closer point to the destination
def closer_point(point1, point2, reference_point):

    distance1 = math.sqrt((point1[0] - reference_point[0])**2 + (point1[1] - reference_point[1])**2)
    distance2 = math.sqrt((point2[0] - reference_point[0])**2 + (point2[1] - reference_point[1])**2)

    if distance1 < distance2:
        return point1
    else:
        return point2

# Helper function to get the intersection point of 2 edges using the position parameters
def intersection_point(pos_edge1, pos_edge2):
    x1, y1 = pos_edge1[0]
    x2, y2 = pos_edge1[1]
    x3, y3 = pos_edge2[0]
    x4, y4 = pos_edge2[1]

    # Calculate the parameters for the line equations of the two edges
    a1 = y2 - y1
    b1 = x1 - x2
    c1 = x2 * y1 - x1 * y2

    a2 = y4 - y3
    b2 = x3 - x4
    c2 = x4 * y3 - x3 * y4

    # Calculate the intersection point
    det = a1 * b2 - a2 * b1

    if det == 0:
        # The edges are parallel, there's no unique intersection point
        return None
    else:
        x = (b1 * c2 - b2 * c1) / det
        y = (a2 * c1 - a1 * c2) / det

        # Check if the intersection point lies within the bounded segment
        if min(x1, x2) <= x <= max(x1, x2) and min(y1, y2) <= y <= max(y1, y2) and \
           min(x3, x4) <= x <= max(x3, x4) and min(y3, y4) <= y <= max(y3, y4):
            return x, y
        else:
            return None
        
# Find the distance between 2 points
# Used to find the closer point
def euclidean_distance(point1, point2):
    return ((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2) ** 0.5

# Helper function to find the opposite face of an edge
def find_opposite_face(currentEdge, faces):
    # Iterate over the neighbors of the current face
    for neighborFace in faces:
        # Find the common edges
        common_edges = set(currentEdge) & set(neighborFace.edges)
        # If there is exactly one common edge, it's the opposite face
        if len(common_edges) == 1:
            oppositeFace = neighborFace
            return oppositeFace

    # If no or more than one common edge was found, something is wrong
    return None

# Helper function to create the intersection structure
def create_intersection_structure(length):

    intersection_structure = []

    for i in range(length):
        intersection_structure.append((-99999999999, -99999999999999, -99999999999999999999, -99999999999999999))
    
    return intersection_structure

# Helper function to find the closer point to point3 between point1 and point2
def find_closest_point(point1, point2, target_point):
    x1, y1, id1 = point1
    x2, y2, id2 = point2
    x_target, y_target, target_id = target_point

    distance1 = ((x1 - x_target)**2 + (y1 - y_target)**2)**0.5
    distance2 = ((x2 - x_target)**2 + (y2 - y_target)**2)**0.5

    if distance1 < distance2:
        return point1
    else:
        return point2
############################################################################################################

def RouteFaces(s,d,fails,faces):
    
    print("Routing in faces started for : ", s , " -> " , d) 
    
    detour_edges = []
    
    hops = 0
    
    switches = 0

    skipSonderfall = False

    #im letzten index von faces ist der ganze graph drin
    try:
        faces[len(faces)-1].add_edge(s, d)
        
    except: #special case where there are no faces for s (special case that didnt get tested earlier and only happens in topology zoo graphs)
        print("Special case PRE")
        for face in faces:
            print(face.nodes())
        faces = [nx.PlanarEmbedding()]
        faces[0].add_edge(s, d)

    imaginary_edge = (s,d)
    #print("Faces : ")
    #for face in faces:
    #    print(face.nodes())
    
    
    try:
        pos_imaginary_edge = (
                faces[len(faces) - 1].nodes[s]['pos'],
                faces[len(faces) - 1].nodes[d]['pos']
            )
    except:
        print("Special case 0 Fail Face Routing")
        print("[RouteFaces]:", s, "->", d)
        print("[RouteFaces] Special Case Faces:")
        for face in faces:
            print(face.nodes())
        
        # Zeichne den Graphen mit farbig markierten Knoten für source und destination
        graph_to_draw = faces[len(faces) - 1]
        if len(graph_to_draw.nodes) > 0:  # Prüfen, ob der Graph Knoten enthält
            plt.figure(figsize=(8, 6))

            # Node positions
            pos = nx.get_node_attributes(graph_to_draw, 'pos')
            if not pos:
                # Falls keine Positionen vorhanden sind, automatische Layout-Positionen erzeugen
                pos = nx.spring_layout(graph_to_draw)

            # Farben für die Knoten festlegen
            node_colors = []
            for node in graph_to_draw.nodes:
                if node == s:
                    node_colors.append('red')  # Source ist rot
                elif node == d:
                    node_colors.append('blue')  # Destination ist blau
                else:
                    node_colors.append('lightgray')  # Andere Knoten sind hellgrau
            
            # Graphen zeichnen
            nx.draw(
                graph_to_draw, pos, with_labels=True, node_color=node_colors, edge_color="gray", node_size=500
            )
            plt.title(f"Special Case Graph for {s} -> {d}")
            plt.show()
        else:
            print("Graph has no nodes to visualize.")
    
    


    currentNode = s

    #als erstes muss man das erste Face finden von dem man aus startet, dafür stehen nur die Faces von s zur verfügung
    possible_start_faces = [face for face in faces[:-1] if s in face]

    #hier speicher ich mir die schnittpunkte von jedem face

    intersection_points_start_faces = []

    for i in range(len(possible_start_faces)+1):  

        #positionX intersection , positionY intersection, nodeX , nodeY 
        item = (-99999999999,-99999999999999 , -99999999999999999999 , -99999999999999999)

        intersection_points_start_faces.append(item)

    indexJ = 0
    
    for start_face in possible_start_faces:
        
        old_Node = s

        currentNodeInStartFace = s

        next_node = list(start_face.neighbors(s))[0]
        
        #den schnittpunkt von jedem start-face mit der imaginären kante speichern

        #-2 weil der letzte Index der ganze Graph ist und man immer +1 auf den Index rechnet
        for i in range(len(start_face.nodes)-1):

           
            
            #jetzt muss ich hier für jede Kante prüfen ob sie geschnitten wird und den Schnittpunkt speichern
            current_edge = (currentNodeInStartFace, next_node)
            
            # hier bekomme ich die Position der current_edge
            pos_current_edge = (
                faces[len(faces) - 1].nodes[currentNodeInStartFace]['pos'],
                faces[len(faces) - 1].nodes[next_node]['pos']
            )

            if(next_node == d):
                print("Routing succesful via Start-Faces")
                print('------------------------------------------------------')
                print(" ")
                return (False, hops, switches, detour_edges)
            

            #prüfen ob die edge geschnitten wird
            intersection = intersection_point(pos_current_edge, pos_imaginary_edge)

            #wenn es zu einem Schnittpunkt kommt, dann muss man gucken ob der derzeitige Schnittpunkt besser ist als der neu gefundene
            if(intersection != None):

                currentNewIntersectionPoint = (intersection[0],intersection[1])

                currentIntersectionPoint = (intersection_points_start_faces[indexJ][0],intersection_points_start_faces[indexJ][1])

                currentImaginaryPoint = (faces[len(faces) - 1].nodes[d]['pos'][0], faces[len(faces) - 1].nodes[d]['pos'][1])


                
                new_intersection = closer_point(currentNewIntersectionPoint, currentIntersectionPoint ,currentImaginaryPoint)

                if(new_intersection == intersection):
     
                    intersection_points_start_faces[indexJ] = (intersection[0],intersection[1],currentNode,next_node)
                    

                
            detour_edges.append((currentNode,next_node))

            #wenns nicht geklappt hat muss man die nächsten nodes nehmen
            #dabei schaltet man einen Knoten in jeder Position weiter
            old_Node = currentNodeInStartFace

            currentNodeInStartFace = next_node
            hops= hops+1

            #da es nur 2 Nachbarn in jedem Face in einem Knoten gibt und ich nicht weiß
            #nach welchem parameter networkx die nachbarn ausgibt
            if list(start_face.neighbors(currentNodeInStartFace))[0] != old_Node:
                next_node = list(start_face.neighbors(currentNodeInStartFace))[0]
            else:
                next_node = list(start_face.neighbors(currentNodeInStartFace))[1]

        indexJ = indexJ + 1
        switches = switches +1#G = G.copy()

    #nx.draw(G, pos, with_labels=True, node_size=700, node_color="green", font_size=8)
    #plt.show()


    print(" ")
    print("Routing via second Faces started")
    currentFace = []

    #jetzt müssen die faces rausgeworfen werden, die keinen Schnittpunkt haben
    update_intersection_points_start_faces = []

    for face_intersection in intersection_points_start_faces:
        if(face_intersection[0] != -99999999999):
            update_intersection_points_start_faces.append(face_intersection)
        
    best_intersection = None

    min_distance = float('inf')

    # hier wird der beste schnittpunkt ermittelt
    for intersection in update_intersection_points_start_faces:

        intersection_pos = (intersection[0], intersection[1])

        distance = euclidean_distance(intersection_pos, currentImaginaryPoint)

        if distance < min_distance:

            min_distance = distance

            best_intersection = intersection
            
            
    #wenn das Startface keinen Schnittpunkt hat dann wird auf dem outerFace analog geroutet
    if(len(update_intersection_points_start_faces) == 0):
        
        
        #da die Bestimmung des OuterFaces nicht so leicht ist müssen alle Faces durchlaufen werden und geprüft,
        #ob man direkt die destination erreicht
        
        updateFaces = faces[:-1]
        print("Special Case 1: Trying to route on the outer face")
        face_pool = [face for face in updateFaces if currentNode in face]
        
        if currentFace in face_pool:  
               
            face_pool.remove(currentFace)
        
        #jedes Face, in dem dann die Source drin ist könnte das outerFace sein
        
        for face in face_pool:

            currentFace = face
            
            lastNode = currentNode

            currentSource = currentNode
            
            nextNode = neighbors = list(currentFace.neighbors(currentNode))[0]

            detour_edges1 = []
            hops1 = 0
         
            #hier findet dann das durchlaufen eines Faces statt
            while(nextNode != currentSource):
            
                # Finde die Nachbarn des aktuellen Knotens im aktuellen Face
                neighbors = list(currentFace.neighbors(currentNode))

                # Überprüfe, welcher Nachbar der nächste ist
                if neighbors[0] == lastNode:
                    
                    nextNode = neighbors[1]
                    
                else:
                    nextNode = neighbors[0]
                
                detour_edges1.append((currentNode,next_node))
                hops1 = hops +1
                lastNode = currentNode
                currentNode = nextNode    
                
                if(nextNode == d):
                    for edge in detour_edges1:
                        detour_edges.append(edge)
    
                    hops = hops + hops1
                    print("Routing successful via OuterFace")
                    return (False, hops, switches, detour_edges)
                
        print("Routing failed via Faces, No Intersection with Start Face")
        return (True, hops, switches, detour_edges)
        
        

            

    #jetzt muss das Face gefunden werden, welches beide knoten enthält
    #for face in faces:
    for i in range(len(faces)-2):
        
        if(faces[i].has_node(best_intersection[2]) and faces[i].has_node(best_intersection[3])):

            currentFace = faces[i]

    #das currentFace bis zum Schnittpunkt durchlaufen
    
    currentNode = s
    
    lastNode = s
    
    nextNode = s

    
    #schleife um das jetzige face bis zum besten schnittpunkt durchzugehen
    while currentNode != best_intersection[2]:
        
        # Finde die Nachbarn des aktuellen Knotens im aktuellen Face
        neighbors = list(currentFace.neighbors(currentNode))

        # Überprüfe, welcher Nachbar der nächste ist
        if neighbors[0] == lastNode:
            
            nextNode = neighbors[1]
            
        else:
            nextNode = neighbors[0]

        # Aktualisiere die Knoten für die nächste Iteration
        detour_edges.append((currentNode,next_node))
        hops = hops +1
        lastNode = currentNode
        currentNode = nextNode    


    #jetzt muss ich currentFace auf das nächste Face setzen
    #als erstes hab ich versucht nach der Regel zu gehen: "jede Kante ist in genau 2 Faces drin"
    #das stimmt leider nicht, da die äußeren Kanten in nur 1 face sind
    #daher nehme ich jetzt einfach ein anderes Face mit CurrentNode drin 
    
    updateFaces = faces[:-1]

    face_pool = [face for face in updateFaces if currentNode in face]
    
    if currentFace in face_pool:     
        face_pool.remove(currentFace)
    
    #Sonderfall, bei dem die geschnittene Kante vom Startface nicht in 2 verschiedenen Faces liegt
    #diesen Fall hier könnte man so ausweiten, dass der nächst-beste Schnittpunkt oder Punkt im Face gefunden wird, der auch 
    #ein weiteres Face besitzt von dem man aus weiter Routen könnte
    else:
        
        print("Sonderfall 2: Versuche den besten Punkt in den StartFaces zu finden")
        skipSonderfall = True
        
        #hier müsste man die StartFaces nochmal durchgehen und den besten Punkt finden zur Destination,
        #der auch ein weiteres Face besitzt
        
        #(coordX,coordY,nodeID)
        closest_point = (-999,-999,-999)
        #um nachher durch das beste Face zu routen
        best_face = []
        
        coordXd , coordYd = faces[len(faces) - 1].nodes[d]['pos']
        destination_point = (coordXd,coordYd,d)
        
        faces_without_start_faces = set(faces[:-1]) - set(possible_start_faces)
        
        #jedes Start-Face durchgehen
        for face in possible_start_faces:
            
            #jeden Knoten im jeweiligen StartFace durchgehen
            for nodeI in face:
                
                #test
                
                #prüfen ob der current_node besser ist als der closest_point
                current_node = nodeI
                
                coordXcurrent, coordYcurrent = faces[len(faces) - 1].nodes[current_node]['pos']
                current_node_point = (coordXcurrent,coordYcurrent,current_node)
                
                if(closest_point != closer_point(current_node_point,closest_point,destination_point)):
                    
                    #der jetzige Punkt ist näher an der Destination dran
                    
                    #jetzt muss geprüft werden ob der jetzige Punkt auch in einem anderen Face drin ist,
                    #welches nicht zu den StartFaces gehört
                    
                    for face_without_start in faces_without_start_faces:
                        
                        #wenn der jetzige beste punkt in einem anderen Face ist, von dem man das Routing aus
                        #fortsetzen kann dann geht man in dieses Face rein
                        if(current_node in face_without_start.nodes):
                            
                            closest_point = current_node_point
                            best_face = face_without_start
        
        if(best_face == (-999,-999,-999)):     
            print("Routing failed via Faces, starting Face Intersection has no opposite Interface and no Point was found with an opposite Face")
            print('------------------------------------------------------')
            return (True, hops, switches, detour_edges)
        
        else: 
            currentFace = best_face
            currentNode = list(best_face.nodes)[0]
    
    #in dem Fall hätte man regulär den Schnittpunkt gefunden
    if(not skipSonderfall):            
        currentFace = face_pool[0]
        currentNode = list(currentFace.nodes)[0]

    scouting = True
    lastWalk = False

    intersection_structure = create_intersection_structure(len(list(currentFace.nodes())))
    best_intersection = (-999,-999,-999,-999)
    scoutSource = currentNode
    lastScoutNode = currentNode
    nextNodeScout = list(currentFace.neighbors(currentNode))[0]

    intersection_index = 0
    
    faceSwitchCounter = 1

    #danach kommt die schleife über die nächsten faces
    while(currentNode != d):
        
        #als erstes muss gescoutet werden
        #dabei läuft man das ganze currentface entlang und speichert sich die schnittpunkte
        if(scouting==True):
            
            #als  erstes müssen die nodes weitergeschaltet werden

            neighbors_next_node_scout = list(currentFace.neighbors(currentNode))
            
            if(neighbors_next_node_scout[0] == lastScoutNode):

                currentNode = nextNodeScout

                nextNodeScout = neighbors_next_node_scout[1]

            else: 
                
                currentNode = nextNodeScout

                nextNodeScout = neighbors_next_node_scout[1]

            detour_edges.append((currentNode,nextNodeScout))
            
            hops = hops +1
            
            #das scouting ist beendet und wir haben die Schnittpunkte alle gespeichert
            if(nextNodeScout == scoutSource):

                scouting = False
                lastWalk = True


            #dann muss die intersection bestimmt werden
            pos_current_edge = (
                faces[len(faces) - 1].nodes[currentNode]['pos'],
                faces[len(faces) - 1].nodes[nextNodeScout]['pos']
            )

            #prüfen ob die edge geschnitten wird
            intersection = intersection_point(pos_current_edge, pos_imaginary_edge)

            #wenn es zu einem Schnittpunkt kommt, dann muss man gucken ob der derzeitige Schnittpunkt besser ist als der neu gefundene
            if(intersection != None):

                currentNewIntersectionPoint = (intersection[0],intersection[1])

                currentIntersectionPoint = (best_intersection[0],best_intersection[1])

                currentImaginaryPoint = (faces[len(faces) - 1].nodes[d]['pos'][0], faces[len(faces) - 1].nodes[d]['pos'][1])

                #hier wird die jetzige Intersection mit der derzeut besten Intersection verglichen
                new_intersection = closer_point(currentNewIntersectionPoint, currentIntersectionPoint ,currentImaginaryPoint)

                #wenn die neu gefundene Intersection besser ist als die alte
                if(new_intersection == intersection):
                    
                    best_intersection = (intersection[0],intersection[1],currentNode,nextNodeScout)


        #dann das face bis zum schnittpunkt durchlaufen und currentface umsetzen
        if(lastWalk == True):
            
            #der letzte Lauf durch das Face ist fertig und es fängt ein neues face an

            if(currentNode == best_intersection[3]):
                lastWalk = False
                scouting = True

                #das nächste face wird mit der Methode bestimmt die oben auch verwendet wurde
                #es wird das erste face genommen, welches einen Knoten der intersection enthält

                updateFaces = faces[:-1]

                face_pool = [face for face in updateFaces if currentNode in face]

                currentFace = face_pool[0]

                switches = switches + 1

                faceSwitchCounter = faceSwitchCounter + 1

            #der letzte Lauf durch das Face ist NOCH NICHT fertig und es geht weiter
            else: 
                neighbors_next_node_scout = list(currentFace.neighbors(currentNode))
            
                if(neighbors_next_node_scout[0] == lastScoutNode):

                    currentNode = nextNodeScout

                    nextNodeScout = neighbors_next_node_scout[1]

                else: 
                    
                    currentNode = nextNodeScout

                    nextNodeScout = neighbors_next_node_scout[1]

                detour_edges.append((currentNode,nextNodeScout))
                
                hops = hops +1

        break
    

    #hier würde man rauskommen, wenn die currentNode == d wird
    #das könnte passieren, wenn man beim scouten des Faces auf die Destination stößt
    print("Routing success via Faces")
    print('------------------------------------------------------')
    print(" ")
    return (False, hops, switches, detour_edges)



#in dieser funktion findet das routing eines source-destination-paares für multipletrees statt
#dies geschieht indem man nach weiterleitung eines pakets an jedem knoten den nächst besten rang bestimmt
def RouteMultipleTrees(s,d,fails,paths):
    
    print("FAIL ANZAHL : ", len(fails))
    #########################################   FOR DEBUG ONLY                #####################################################
    skip_edps = False
    skip_trees = False
    if(skip_edps):
        print("Skipping the EDPs")
    #endif
    if(skip_trees):
        print("Skipping Trees")
    #endif

    ###############################################################################################################################


     #alle EDPS entlang routen
    currentNode = s
    last_node = currentNode
    detour_edges = []
    hops = 0
    switches = 0
    trees = paths[s][d]['trees']
    print(" ")
    print('Routing started for ' , s , " to " , d )

    if(not skip_trees):

        print(" ")
        print("Routing via Tree started")

        for tree in trees:
            
            #hier wurde das Routing von OneTree eingesetzt
            while(currentNode != d):#in dieser Schleife findet das Routing im Tree statt
                                #die idee hinter dieser schleife ist ein großes switch-case
                                #bei dem man je nach eingehenden und funktionierenden ausgehenden ports switcht
                                #nach jedem schritt den man im baum geht folgt die prüfung ob man schon am ziel angekommen ist


                #kommt das paket von einer eingehenden kante (parent) an dann wird der kleinste rang der kinder gewählt
                #denn man war noch nicht an diesem node
                if last_node == get_parent_node(tree,currentNode) or last_node == currentNode:


                    #suche das kind mit dem kleinsten  rang



                    children = []
                    #es werden alle Kinder gespeichert zu denen der jetzige Knoten einen Verbindung hat und sortiert nach ihren Rängen
                    out_edges_with_fails = tree.out_edges(currentNode)
                    out_edges = []
                    for edge in out_edges_with_fails:
                        if edge in fails or tuple(reversed(edge)) in fails:
                            continue

                        else: 
                            out_edges.append(edge)
                        #endif
                    #endfor
                    for nodes in out_edges:
                        children.append(nodes[1])
                    #endfor
                    
                    #print("Versuche auf die Kinder zuzugreifen : " , children)
                    
                    children.sort(key=lambda x: (getRank(tree, x)))


                    if len(children) >  0 : #wenn es kinder gibt, zu denen die Kanten nicht kaputt sind
                        #setze lastnode auf currentnode
                        #setze current node auf das kind mit dem kleinesten rang
                        #dadurch "geht man" die kante zum kind
                        last_node = currentNode
                        currentNode = children[0]
                        hops += 1
                    

                    else: #wenn alle Kanten zu den Kindern kaputt sind dann ist man fertig wenn man an der source ist oder man muss eine kante hoch
                        if currentNode == s: 
                            break; #das routing für diesen Baum
                        #endif


                        #man nimmt die eingehende kante des currentnode und "geht eine stufe hoch"
                        hops += 1
                        detour_edges.append( (currentNode, last_node) )
                        last_node = currentNode
                        currentNode = get_parent_node(tree,currentNode)

                    #endif
                #endif



                children_of_currentNode = []

                for nodes in tree.out_edges(currentNode):
                        children_of_currentNode.append(nodes[1])
                #endfor

                #wenn das Paket nicht aus einer eingehenden Kante kommt, dann muss es aus einer ausgehenden (kind) kommen
                #dafür muss man den Rang des Kindes bestimmen von dem das Paket kommt
                #das Kind mit dem nächsthöheren Rang suchen
                if last_node in children_of_currentNode:

                    #alle funktionierenden Kinder finden
                    children = []
                    out_edges_with_fails = tree.out_edges(currentNode)
                    out_edges = []
                    for edge in out_edges_with_fails:
                        if edge in fails or tuple(reversed(edge)) in fails:
                            continue 
                        else: 
                            out_edges.append(edge)
                        #endif

                    #endfor
                    for nodes in out_edges:
                        children.append(nodes[1])
                    #endfor
                    children.sort(key=lambda x: (getRank(tree, x)))

                    

                    #wenn es Funktionierende Kinder gibt dann muss man das Kind suchen mit dem nächstgrößeren Rang
                    if len(children) > 0: 
                        #prüfen ob es noch kinder gibt mit größerem rang , also ob es noch zu durchlaufene kinder gibt
                        

                        #welchen index hat das kind nach seinem "rank" in der sortierten liste
                        index_of_last_node = children.index(last_node) if last_node in children else -1 
                    
                        #alle  kinder ohne das wo das paket herkommt
                        children_without_last = [a for a in children if a != last_node] 

                        #es gibt keine möglichen kinder mehr und man ist an der Source
                        #dann ist das Routing fehlgeschlagen für diesen Baum
                        if len(children_without_last) == 0 and currentNode == s : 
                            break;

                        #Sonderfall (noch unklar ob nötig)
                        #wenn man aus einem Kind kommt, zu dem die Kante fehlerhaft ist
                        #man nimmt trotzdem das nächste Kind
                        elif index_of_last_node == -1:
                            hops += 1
                            last_node = currentNode
                            currentNode = children[0]

                        #das kind wo das paket herkommt hatte den höchsten rang der kinder, also das letztmögliche
                        #daher muss man den Baum eine Stufe hoch
                        elif index_of_last_node == len(children)-1: 
                            
                            if currentNode != s: #man muss eine stufe hoch gehen
                                hops += 1
                                detour_edges.append( (currentNode, last_node) )
                                last_node = currentNode
                                currentNode = get_parent_node(tree,currentNode)
                            else:#sonderfall wenn man an der Source ist dann ist das Routing gescheitert
                                break;

                        #es gibt noch mindestens 1 Kind mit höherem Rang
                        elif index_of_last_node < len(children)-1 : 
                            #wenn ja dann nimm das Kind mit dem nächst größeren Rang aus der sortierten Children Liste
                            hops += 1
                            last_node = currentNode
                            currentNode = children[index_of_last_node+1]

                        #es gibt keine kinder mehr am currentnode
                        else: 
                            #wenn nein dann setze currentnode auf den parent
                            hops += 1
                            detour_edges.append( (currentNode, last_node) )
                            last_node = currentNode
                            currentNode = get_parent_node(tree,currentNode)
                        #endif

                    #wenn es keine funktionierenden Kinder gibt dann geht man eine Stufe hoch
                    else: 
                        detour_edges.append( (currentNode, last_node) )
                        hops += 1
                        last_node = currentNode
                        currentNode = get_parent_node(tree,currentNode)
                    #endif
                
                    
            #endwhile

            if (currentNode == d):#falls wir am ziel angekommen sind
                print("Routing done via the Tree : ", list(tree.nodes))
                print(" ")
                return (False, hops, switches, detour_edges)
            #endif

            #das war in OneTree egal, da man nur 1 Tree hatte, aber hier kann die Source der Bäume nicht an den gleichen Knoten dran sein wie
            #andere Bäume, daher muss der eingehende Port des Pakets auf den currentNode gesetzt werden (hier die Source) damit man 
            #weiß dass man den nächsten Baum nimmt
            last_node = currentNode
        #endfor 
        print("Routing failed via Trees ")
        print(" ")
        return (True, hops, switches, detour_edges)    
    #endif

###########################################################################################################################################################
def print_paths(paths):
    for source, destinations in paths.items():
        print(f"Source: {source}")
        for destination, values in destinations.items():
            print(f"  Destination: {destination}")
            for key, value in values.items():
                print(f"    {key}: {value}")
        print()
###########################################################################################################################################################



#routing methode von onetree um zwischen einem source-destination paar zu routen
#dies geschieht indem man nach weiterleitung eines pakets an jedem knoten den nächst besten rang bestimmt
def RouteOneTree (s,d,fails,paths):
    
    #print("Fails in  ONETREE Routing : ", fails)
    ####################ONLY FOR RUNNING WITH CLUSTERED FAILURES VIA THE CUSTOM RUN COMMAND#################
    # # Assuming paths is a dictionary with keys as possible values
    # keys = list(paths.keys())
    # # Select a random key for s
    # s = keys[0]
    # neighbors = list(paths[s])
    # print("Neighbors : ", neighbors)
    # # Select a different random key for d
    # d = keys[len(keys)-1]
    ##################################
    
    #print("FAIL NUMBER : ", len(fails))

    if s != d :
        
        currentNode = -1
        edpIndex = 0
        detour_edges = []
        hops = 0
        switches = 0
        tree = paths[s][d]['tree']
        edps_for_s_d = paths[s][d]['edps']

        print('Routing OT started for ' , s , " to " , d )
        #print("EDPs: ", edps_for_s_d)
        #als erstes anhand der EDPs (außer dem längsten, also dem letzten) versuchen zu routen
        for edp in edps_for_s_d:

            #print("[RouteOneTree] current EDP:", edp)
            currentNode = s
            last_node = s 

            if edp != edps_for_s_d[len(edps_for_s_d) -1]:

                currentNode = edp[edpIndex]


                #jeder EDP wird so weit durchlaufen bis man mit dem currentNode zum Ziel kommt oder man auf eine kaputte Kante stößt
                while (currentNode != d):


                    #man prüft ob die nächste Kante im EDP kaputt ist so, indem man guckt ob eine Kante vom currentNode edp[edpIndex] zum nächsten Node im EDP edp[edpIndex+1] in Fails ist
                    #dies beruht auf lokalen Informationen, da EDPs nur eine eingehende Kante haben ( auf der das Paket ankommt ) und eine ausgehende Kante (auf der das Paket nicht ankommt)
                    if (edp[edpIndex], edp[edpIndex +1]) in fails or (edp[edpIndex +1], edp[edpIndex]) in fails:
                        
                        matching_fails = [edge for edge in fails if edp[edpIndex] in edge]
                        #print("[RouteOneTree] Kante im EDP ist kaputt")
                        #print("[RouteOneTree] Untersuche Kante im EDP:", (edp[edpIndex], edp[edpIndex +1]))
                        #print("[RouteOneTree] passende fails:", matching_fails)

                        #wenn man auf eine fehlerhafte Kante stößt dann wechselt man den Pfad
                        switches += 1

                        #die kanten die wir wieder zurückgehen sind die kanten die wir schon in dem edp gelaufen sind
                        detour_edges.append( (edp[edpIndex], edp[edpIndex +1]) )

                        #wir fangen beim neuen edp ganz am anfang an
                        tmp_node = currentNode #und gehen eine Kante hoch, also den edp zurück
                        currentNode = last_node #das "rückwärts den edp gehen" kann so gemacht werden, da die pakete so nur über den port gehen müssen über den sie reingekommen sind
                        last_node = tmp_node
                        hops += 1
                        break

                    else :#wenn die kante die man gehen will inordnung ist, die kante gehen und zum nächsten knoten schalten
                        edpIndex += 1
                        hops += 1
                        last_node = currentNode 
                        currentNode = edp[edpIndex] #man kann hier currentnode direkt so setzen, da es im edp für jeden knoten jeweils 1 ausgehende
                                                    #und genau eine eingehende Kante gibt
                    #endif

                #endwhile

                #nun gibt es 2 Möglichkeiten aus denen die while-Schleife abgebrochen wurde : Ziel erreicht / EDP hat kaputte Kante 


                if currentNode == d : #wir haben die destination mit einem der edps erreicht
                    print('Routing done via EDP')
                    print('------------------------------------------------------')
                    return (False, hops, switches, detour_edges)
                #endif
                
                #wenn man hier angelangt ist, dann bedeutet dies, dass die while(currentNode != d) beendet wurde weil man auf eine kaputte kante gestoßen ist 
                #und dass man nicht an der destination angekommen ist, daher muss man jetzt an die source zurück um den nächsten edp zu starten
                while currentNode != s: #hier findet die Rückführung statt
                    detour_edges.append( (last_node,currentNode) )

                    last_node = currentNode #man geht den edp so weit hoch bis man an der source ist
                    
                    printIndex = edpIndex-1
                    
                    
                    #print("Source : ", s , " Destination : ", d)
                    #print("Edp : ", edp)
                    #print("EdpIndex-1 : ", printIndex)
                    #print("edp[edpIndex-1] : ", edp[edpIndex-1])
                    #print(" ")
                    
                    
                    currentNode = edp[edpIndex-1] #man kann auch hier direkt den edp index verwenden da man genau 1 eingehende kante hat
                    edpIndex = edpIndex-1
                    hops += 1

                #endwhile
            #endif

        #endfor

        # wenn wir es nicht geschafft haben anhand der edps allein zum ziel zu routen dann geht es am längsten edp weiter
        print('Routing via EDPs FAILED')
        
        currentNode = s
        print("Routing via Tree started")
        last_node = currentNode


        while(currentNode != d):#in dieser Schleife findet das Routing im Tree statt
                                #die idee hinter dieser schleife ist ein großes switch-case
                                #bei dem man je nach eingehenden und funktionierenden ausgehenden ports switcht
                                #nach jedem schritt den man im baum geht folgt die prüfung ob man schon am ziel angekommen ist


            #kommt das paket von einer eingehenden kante an dann wird der kleinste rang der kinder gewählt
            #denn man war noch nicht an diesem node
            if last_node == get_parent_node(tree,currentNode) or last_node == currentNode:

                #suche das kind mit dem kleinsten  rang

                children = []
                #es werden alle Kinder gespeichert zu denen der jetzige Knoten einen Verbindung hat und sortiert nach ihren Rängen
                out_edges_with_fails = tree.out_edges(currentNode)
                out_edges = []
                for edge in out_edges_with_fails:
                    if edge in fails or tuple(reversed(edge)) in fails:
                        continue
                    else: 
                        out_edges.append(edge)
                    #endif
                #endfor
                for nodes in out_edges:
                    children.append(nodes[1])
                #endfor
                children.sort(key=lambda x: (getRank(tree, x)))


                if len(children) >  0 : #wenn es kinder gibt, zu denen die Kanten nicht kaputt sind
                    #setze lastnode auf currentnode
                    #setze current node auf das kind mit dem kleinesten rang
                    #dadurch "geht man" die kante zum kind
                    last_node = currentNode
                    currentNode = children[0]
                    hops += 1
                    #print("[RouteOneTree] keine Kinder mehr da, gehe hoch")
                   

                else: #wenn alle Kanten zu den Kindern kaputt sind dann ist man fertig wenn man an der source ist oder man muss eine kante hoch
                    if currentNode == s: 
                        break; #das routing ist gescheitert
                    #endif


                    #man nimmt die eingehende kante des currentnode und "geht eine stufe hoch"
                    hops += 1
                    detour_edges.append( (currentNode, last_node) )
                    last_node = currentNode
                    currentNode = get_parent_node(tree,currentNode)
                    #print("[RouteOneTree] keine Kinder mehr da, ENDE")
                #endif
            #endif



            children_of_currentNode = []

            for nodes in tree.out_edges(currentNode):
                    children_of_currentNode.append(nodes[1])
            #endfor

            #wenn das Paket nicht aus einer eingehenden Kante kommt, dann muss es aus einer ausgehenden kommen
            #dafür muss man den Rang des Kindes bestimmen von dem das Paket kommt
            #das Kind mit dem nächsthöheren Rang suchen
            if last_node in children_of_currentNode:
            
                #alle funktionierenden Kinder finden
                children = []
                out_edges_with_fails = tree.out_edges(currentNode)
                out_edges = []
                for edge in out_edges_with_fails:
                    if edge in fails or tuple(reversed(edge)) in fails:
                        continue
                        
                    else: 
                        out_edges.append(edge)
                    #endif

                #endfor
                for nodes in out_edges:
                    children.append(nodes[1])
                #endfor
                children.sort(key=lambda x: (getRank(tree, x)))

                

                #wenn es Funktionierende Kinder gibt dann muss man das Kind suchen mit dem nächstgrößeren Rang
                if len(children) > 0: 
                    #prüfen ob es noch kinder gibt mit größerem rang , also ob es noch zu durchlaufene kinder gibt
                    

                    #welchen index hat das kind nach seinem "rank" in der sortierten liste
                    index_of_last_node = children.index(last_node) if last_node in children else -1 
                
                    #alle  kinder ohne das wo das paket herkommt
                    children_without_last = [a for a in children if a != last_node] 

                    

                    #es gibt keine möglichen kinder mehr und man ist an der Source
                    #dann ist das Routing fehlgeschlagen
                    if len(children_without_last) == 0 and currentNode == s : 
                        break;

                    #Sonderfall (noch unklar ob nötig)
                    #wenn man aus einem Kind kommt, zu dem die Kante fehlerhaft ist
                    #man nimmt trotzdem das nächste Kind
                    elif index_of_last_node == -1:
                        
                        hops += 1
                        last_node = currentNode
                        currentNode = children[0]


                    #das kind wo das paket herkommt hatte den höchsten rang der kinder, also das letztmögliche
                    #daher muss man den Baum eine Stufe hoch
                    elif index_of_last_node == len(children)-1: 
                        
                        if currentNode != s: #man muss eine stufe hoch gehen
                            hops += 1
                            detour_edges.append( (currentNode, last_node) )
                            last_node = currentNode
                            currentNode = get_parent_node(tree,currentNode)
                        else:#sonderfall wenn man an der Source ist dann ist das Routing gescheitert
                            break;

                    #es gibt noch mindestens 1 Kind mit höherem Rang
                    elif index_of_last_node < len(children)-1 : 
                        
                        #wenn ja dann nimm das Kind mit dem nächst größeren Rang aus der sortierten Children Liste
                        hops += 1
                        last_node = currentNode
                        currentNode = children[index_of_last_node+1]


                    #es gibt keine kinder mehr am currentnode
                    else: 
                        
                        #wenn nein dann setze currentnode auf den parent
                        hops += 1
                        detour_edges.append( (currentNode, last_node) )
                        last_node = currentNode
                        currentNode = get_parent_node(tree,currentNode)
                    #endif

                #wenn es keine funktionierenden Kinder gibt dann geht man eine Stufe hoch
                else: 
                    detour_edges.append( (currentNode, last_node) )
                    hops += 1
                    
                    last_node = currentNode
                    currentNode = get_parent_node(tree,currentNode)
                   
                #endif
            
                
        #endwhile

        #hier kommt man an wenn die while schleife die den tree durchläuft "gebreakt" wurde und man mit dem tree nicht zum ziel gekommen ist
        #oder wenn die bedingung nicht mehr gilt (currentNode != d) und man das ziel erreicht hat

        if currentNode == d : #wir haben die destination mit dem tree erreicht
            print('Routing done via Tree')
            print('------------------------------------------------------')
            return (False, hops, switches, detour_edges)
        #endif
        
        print('Routing via Tree failed')
        print('------------------------------------------------------')
        return (True, hops, switches, detour_edges)
    else: 
        return (True, 0, 0, [])


def getRank(tree, el):
    return tree.nodes[el]["rank"]

# Route according to deterministic circular routing as described by Chiesa et al.
# source s
# destination d
# link failure set fails
# arborescence decomposition T
def RouteDetCirc(s, d, fails, T):
    print("[RouteDetCirc] routing started for", s, "->", d)
    curT = 0
    detour_edges = []
    hops = 0
    switches = 0
    n = len(T[0].nodes())
    k = len(T)
    while (s != d):
        while (s not in T[curT].nodes()) and switches < k*n:
            curT = (curT+1) % k
            switches += 1
        if switches >= k*n:
            break
        nxt = list(T[curT].neighbors(s))
        if len(nxt) != 1:
            print("Bug: too many or to few neighbours")
        nxt = nxt[0]

        print("[RouteDetCirc] current s:", s, "and fails with s:", [fail for fail in fails if s in fail])

        
        
        if (nxt, s) in fails or (s, nxt) in fails:
            curT = (curT+1) % k
            switches += 1
        else:
            if switches > 0 and curT > 0:
                detour_edges.append((s, nxt))
            s = nxt
            hops += 1
        if hops > n or switches > k*n:
            print("[RouteDetCirc] Routing Failed with RouteDetCirc")
            return (True, -1, switches, detour_edges)
    print("[RouteDetCirc] Routing Success with RouteDetCirc")
    return (False, hops, switches, detour_edges)

#select next arborescence to bounce
def Bounce(s, d, T, cur):
    for i in range(len(T)):
        if (d, s) in T[i].edges():
            return i
    else:
        return (cur+1) % len(T)

# Route with bouncing for 3-connected graph by Chiesa et al.
# source s
# destination d
# link failure set fails
# arborescence decomposition T
def RouteDetBounce(s, d, fails, T):
    detour_edges = []
    curT = 0
    hops = 0
    switches = 0
    n = len(T[0].nodes())
    while (s != d):
        nxt = list(T[curT].neighbors(s))
        if len(nxt) != 1:
            print("Bug: too many or to few neighbours")
        nxt = nxt[0]
        if (nxt, s) in fails or (s, nxt) in fails:
            if curT == 0:
                curT = Bounce(s, nxt, T, curT)
            else:
                curT = 3 - curT
            switches += 1
        else:
            if switches > 0:
                detour_edges.append((s, nxt))
            s = nxt
            hops += 1
        if hops > 3*n or switches > k*n:
            print("cycle Bounce")
            return (True, hops, switches, detour_edges)
    return (False, hops, switches, detour_edges)

#construct BIDB 7 matrix
def PrepareBIBD(connectivity):
    global matrix
    matrix = []
    matrix.append([5,0,6,1,2,4,3])
    matrix.append([0,1,2,3,4,5,6])
    matrix.append([6,2,0,4,1,3,5])
    matrix.append([4,3,5,0,6,1,2])
    matrix.append([1,4,3,2,5,6,0])
    matrix.append([2,5,4,6,3,0,1])
    matrix.append([3,6,1,5,0,2,4])

# Route with BIBD matrix
# source s
# destination d
# link failure set fails
# arborescence decomposition T
def RouteBIBD(s, d, fails, T):
    if len(matrix) == 0:
        PrepareBIBD(k)
    detour_edges = []
    curT = matrix[int(s) % (k-1)][0]
    hops = 0
    switches = 0
    source = s
    n = len(T[0].nodes())
    while (s != d):
        nxt = list(T[curT].neighbors(s))
        if len(nxt) != 1:
            print("Bug: too many or to few neighbours")
        nxt = nxt[0]
        if (nxt, s) in fails or (s, nxt) in fails:
            switches += 1
            # print(switches)
            curT = matrix[int(source) % (k-1)][switches % k]
        else:
            if switches > 0:
                detour_edges.append((s, nxt))
            s = nxt
            hops += 1
        if hops > 3*n or switches > k*n:
            print("cycle BIBD")
            return (True, hops, switches, detour_edges)
    return (False, hops, switches, detour_edges)

#build data structure for square one algorithm
SQ1 = {}
def PrepareSQ1(G, d):
    global SQ1
    H = build_auxiliary_edge_connectivity(G)
    R = build_residual_network(H, 'capacity')
    SQ1 = {n: {} for n in G}
    for u in G.nodes():
        
        if (u != d):
            print("Baue die Struktur für " , u , " nach " , d)
            k = sorted(list(nx.edge_disjoint_paths(
                G, u, d, auxiliary=H, residual=R)), key=len)
            SQ1[u][d] = k
            print(" ")
            print("SQ : " , SQ1)
    print(" ")
    print("PREPARE SQ1 FERTIG")

def PrepareSQ1(G) :
    global SQ1
    H = build_auxiliary_edge_connectivity(G)
    R = build_residual_network(H, 'capacity')

    SQ1 = {n: {} for n in G}
    for key in SQ1 :
        #print(key)
        #print(SQ1[key])
        for n in G.nodes():
            if key != n:
                SQ1[key][n] = {}
        #print(SQ1[key])

    for s in G.nodes():
        for d in G.nodes():
            if (s != d):
                #print("SQ vorher : " , SQ1)
                #print("Baue die Struktur für " , s , " nach " , d)
                k = sorted(list(nx.edge_disjoint_paths(
                    G, s, d, auxiliary=H, residual=R)), key=len)
                if len(SQ1[s][d]) > 0:
                    SQ1[s][d].append(k)
                else:
                    SQ1[s][d] = k
                
                #print(" ")
                #print("SQ nachher : " , SQ1)

    #print(" ")
    #print("PREPARE SQ1 FERTIG")
    return SQ1


def returnSQ1():
    return SQ1

# Route with Square One algorithm
# source s
# destination d
# link failure set fails
# arborescence decomposition T
def RouteSQ1(s, d, fails, T):
    #print(" ")
    #print("Source :  ", s , " Destination : ", d )
    #print(" ")
    #print("SQ1[s][d] : ", SQ1[s][d])
    #print(" ")
    curRoute = SQ1[s][d][0]
    k = len(SQ1[s][d])
    detour_edges = []
    index = 1
    hops = 0
    switches = 0
    c = s  # current node
    #print(T)
    # n = len(T[0].nodes())
    n = len(T[s])
    #print(" ")
    #print("len(T[0]) : ",n)
    #print(" ")
    while (c != d):
        #print("CurRoute :" , curRoute)
        nxt = curRoute[index]
        if (nxt, c) in fails or (c, nxt) in fails:
            for i in range(2, index+1):
                detour_edges.append((c, curRoute[index-i]))
                c = curRoute[index-i]
            switches += 1
            c = s
            hops += (index-1)
            curRoute = SQ1[s][d][switches % k]
            index = 1
        else:
            if switches > 0:
                detour_edges.append((c, nxt))
            c = nxt
            index += 1
            hops += 1
        if hops > 3*n or switches > k*n:
            print("cycle square one")
            return (True, hops, switches, detour_edges)
    return (False, hops, switches, detour_edges)


# Route with randomization as described by Chiesa et al.
# source s
# destination d
# link failure set fails
# arborescence decomposition T
P = 0.5358  # bounce probability
def RoutePR(s, d, fails, T):
    detour_edges = []
    curT = 0
    hops = 0
    switches = 0
    n = len(T[0].nodes())
    while (s != d):
        nxt = list(T[curT].neighbors(s))
        if len(nxt) != 1:
            print("Bug: too many or to few neighbours")
        nxt = nxt[0]
        if (nxt, s) in fails or (s, nxt) in fails:
            x = random.random()
            if x <= P:
                curT = Bounce(s, nxt, T, curT)
            else:
                newT = random.randint(0, len(T)-2)
                if newT >= curT:
                    newT = (newT+1) % len(T)
                curT = newT
            switches += 1
        else:
            if switches > 0:
                detour_edges.append((s, nxt))
            s = nxt
            hops += 1
        if hops > 3*n or switches > k*n:
            print("cycle PR")
            return (True, hops, switches, detour_edges)
    return (False, hops, switches, detour_edges)

# Route randomly without bouncing as described by Chiesa et al.
# source s
# destination d
# link failure set fails
# arborescence decomposition T
def RoutePRNB(s, d, fails, T):
    detour_edges = []
    curT = 0
    hops = 0
    switches = 0
    n = len(T[0].nodes())
    while (s != d):
        nxt = list(T[curT].neighbors(s))
        if len(nxt) != 1:
            print("Bug: too many or to few neighbours")
        nxt = nxt[0]
        if (nxt, s) in fails or (s, nxt) in fails:
            newT = random.randint(0, len(T)-2)
            if newT >= curT:
                newT = (newT+1) % len(T)
            curT = newT
            switches += 1
        else:
            if switches > 0:
                detour_edges.append((s, nxt))
            s = nxt
            hops += 1
        if hops > 3*n or switches > k*n:
            print("cycle PRNB")
            return (True, hops, switches, detour_edges)
    return (False, hops, switches, detour_edges)

# Route with bouncing variant by Chiesa et al.
# source s
# destination d
# link failure set fails
# arborescence decomposition T
def RouteDetBounce2(s, d, fails, T):
    detour_edges = []
    curT = 0
    hops = 0
    switches = 0
    n = len(T[0].nodes())
    while (s != d):
        nxt = list(T[curT].neighbors(s))
        nxt = nxt[0]
        if (nxt, s) in fails or (s, nxt) in fails:
            if curT == 0:
                curT = Bounce(s, nxt, T, curT)
            else:
                curT = 1+(curT) % (len(T)-1)
            switches += 1
        else:
            if switches > 0:
                detour_edges.append((s, nxt))
            s = nxt
            hops += 1
        if hops > 3*n or switches > k*n:
            #print("cycle DetBounce2")
            return (True, hops, switches, detour_edges)
    return (False, hops, switches, detour_edges)

#compute best arb for low stretch to use next
arb_order = {}
def next_stretch_arb(s, curT):
    indices = arb_order[s]
    index = (indices.index_of(curT) + 1) % k
    return index

# Choose next arborescence to minimize stretch when facing failures
# source s
# destination d
# link failure set fails
# arborescence decomposition T
def Route_Stretch(s, d, fails, T):
    curT = 0
    detour_edges = []
    hops = 0
    switches = 0
    n = len(T[0].nodes())
    while (s != d):
        # print "At ", s, curT
        nxt = list(T[curT].neighbors(s))
        # print "neighbours:", nxt
        if len(nxt) != 1:
            print("Bug: too many or to few neighbours")
        nxt = nxt[0]
        if (nxt, s) in fails or (s, nxt) in fails:
            curT = next_stretch_arb(s, curT)
            switches += 1
        else:
            if switches > 0 and curT > 0:
                detour_edges.append((s, nxt))
            s = nxt
            hops += 1
        if hops > 2*n or switches > k*n:
            print("cycle det circ")
            return (True, hops, switches, detour_edges)
    return (False, hops, switches, detour_edges)


# run routing algorithm on graph g
# RANDOM: don't use failset associated with g, but construct one at random
# stats: statistics object to fill
# f: number of failed links
# samplesize: number of nodes from which we route towards the root
# dest: nodes to exclude from using in sample
# tree: arborescence decomposition to use
def SimulateGraph(g, RANDOM, stats, f, samplesize, precomputation=None, dest=None, tree=None, targeted=False):
    edg = list(g.edges())
    fails = g.graph['fails']
    f = len(fails)
    #print("[SimulateGraph] (1) fails:",fails)
    #print("[SimulateGraph] len(fails):" , len(fails))
    if fails != None:
        if len(fails) < f:
            fails = fails + edg[:f - len(fails) + 1]
        edg = fails
    if f > len(edg):
        #print('more failures than edges')
        #print('simulate', len(g.edges()), len(fails), f)
        return -1
    d = g.graph['root']
    g.graph['k'] = k
    if precomputation is None:
        precomputation = tree
        if precomputation is None:
            precomputation = GreedyArborescenceDecomposition(g)
            if precomputation is None:
               return -1
    fails = edg[:f]
    #print("[SimulateGraph] (2) fails:",fails)
    if targeted: #neu eingefügt für die clustered failures
        fails = []
        
    failures1 = {(u, v): g[u][v]['arb'] for (u, v) in fails}
    failures1.update({(v, u): g[u][v]['arb'] for (u, v) in fails})

    g = g.copy(as_view=False)

    #######################################################################
    debugGraphShow = False

    if(debugGraphShow):   
        print("[SimulateGraph] DEBUG ON")
        print("[SimulateGraph] Failed Experiment with fails : ", failures1)
        # Extract positions (if they exist)
        # Extract positions (if they exist)
        pos = {}
        for node in g.nodes(data=True):
            if 'pos' in node[1]:
                x, y = node[1]['pos']  # Directly unpack the tuple (x, y)
                pos[node[0]] = (x, y)

        # If positions are not provided, generate them
        if not pos:
            pos = nx.spring_layout(g)

        # Assign edge colors: failed edges in red, normal edges in black
        edge_colors = []
        for edge in g.edges():
            if edge in failures1 or (edge[1], edge[0]) in failures1:
                edge_colors.append('red')  # Highlight failed edges in red
            else:
                edge_colors.append('black')  # Normal edges in black

        # Draw the graph
        plt.figure(figsize=(12, 8))
        nx.draw(
            g,
            pos,
            with_labels=True,
            node_color='lightblue',
            edge_color=edge_colors,
            node_size=500,
            font_size=8,
            arrows=True,
        )
    
        # Add legend for fails
        legend_labels = ['Fail Edges', 'Normal Edges']
        legend_colors = ['red', 'black']
        for color, label in zip(legend_colors, legend_labels):
            plt.plot([], [], color=color, label=label)

        plt.legend(loc='upper right')
        plt.title("Graph Visualization with Fail Edges Highlighted")
        plt.show()

    ####################################################
    g.remove_edges_from(failures1.keys())

    nodes = list(set(connected_component_nodes_with_d_after_failures(g,[],d))-set([dest, d]))
    dist = nx.shortest_path_length(g, target=d)
    if len(nodes) < samplesize:

        print('Not enough nodes in connected component of destination (%i nodes, %i sample size), adapting it' % (len(nodes), samplesize))
        PG = nx.nx_pydot.write_dot(g , "./graphen/failedGraphs/graph")
        samplesize = len(nodes)
    
    nodes = list(set(g.nodes())-set([dest, d]))
    random.shuffle(nodes)
    count = 0
    for s in nodes[:samplesize]:
        print("Loop over samplesize is runing")
        count += 1
        for stat in stats:
            print("Loop over stats is runing")
            if targeted:
                fails = list(nx.minimum_edge_cut(g,s=s,t=d))[1:]
                random.shuffle(fails)
                failures1 = {(u, v): g[u][v]['arb'] for (u, v) in fails}
                g.remove_edges_from(failures1.keys())
                x = dist[s]
                dist[s] = nx.shortest_path_length(g,source=s,target=d)
                #print(len(fails),x,dist[s]) #DEBUG
                print("[SimulateGraph targeted] len(fails):", len(fails))
                print("[SimulateGraph targeted] fails:", fails)
            if (s == d) or (not s in dist):
                stat.fails += 1
                continue
            (fail, hops) = stat.update(s, d, fails, precomputation, dist[s])
            if fail:
                stat.hops = stat.hops[:-1]
                stat.stretch = stat.stretch[:-1]
            elif hops < 0:
                stat.hops = stat.hops[:-1]
                stat.stretch = stat.stretch[:-1]
                stat.succ = stat.succ - 1
            if targeted:
                for ((u, v), i) in failures1.items():
                    g.add_edge(u, v)
                    g[u][v]['arb'] = i
            if stat.succ + stat.fails != count:
                print('problem, success and failures do not add up', stat.succ, stat.fails, count)
                print('source', s)
                if stat.has_graph:
                    drawGraphWithLabels(stat.graph, "results/problem.png")
    if not targeted:
        for ((u, v), i) in failures1.items():
            g.add_edge(u, v)
            g[u][v]['arb'] = i
    for stat in stats:
        stat.finalize()
    sys.stdout.flush()
    return fails

# class to collect statistics on routing simulation
class Statistic:
    def __init__(self, routeFunction, name, g=None):
        self.funct = routeFunction
        self.name = name
        self.has_graph = g is not None
        if g is not None:
            self.graph = g

    def reset(self, nodes):
        self.totalHops = 0
        self.totalSwitches = 0
        self.fails = 0
        self.succ = 0
        self.stretch = [-2]
        self.hops = [-2]
        self.lastsuc = True
        self.load = {(u, v): 0 for u in nodes for v in nodes}
        self.lat = 0

    # add data for routing simulations from source s to destination
    # despite the failures in fails, using arborescences T and the shortest
    # path length is captured in shortest
    def update(self, s, d, fails, T, shortest):
        if not self.has_graph:
            (fail, hops, switches, detour_edges_used) = self.funct(s, d, fails, T)
        else:
            (fail, hops, switches, detour_edges_used) = self.funct(s, d, fails, T, self.graph)
        #if switches == 0:
        #    fail = False
        if fail:
            self.fails += 1
            self.lastsuc = False
            self.stretch.append(-1)
            self.hops.append(-1)
            for e in detour_edges_used:
                self.load[e] += 1
        else:
            self.totalHops += hops
            self.succ += 1
            self.totalSwitches += switches
            if shortest == 0:
                shortest = 1
            self.stretch.append(hops-shortest)
            self.hops.append(hops)
            for e in detour_edges_used:
                self.load[e] += 1
            self.lastsuc = True
        return (fail, hops)

    def max_stretch(self):
        return max(self.stretch)

    # compute statistics when no more data will be added
    def finalize(self):
        self.lat = -1
        self.load = max(self.load.values())
        if len(self.hops) > 1:
            self.hops = self.hops[1:]
            self.stretch = self.stretch[1:]
        else:
            self.hops = [0]
            self.stretch = [0]

        if len(self.hops) > 0:
            self.lat = np.mean(self.hops)
        return max(self.stretch)

    def max_load(self):
        return max(self.load.values())

    def load_distribution(self):
        return [x*1.0/self.size**2 for x in np.bincount(self.load.values())]


import os
import matplotlib.pyplot as plt
import networkx as nx
from datetime import datetime

def draw_tree_with_highlights(tree, nodes=None, fails=None, current_edge=None, showplot=True):
    """
    Zeichnet einen Baum-Graphen und hebt bestimmte Knoten, fehlerhafte Kanten und die aktuelle Kante hervor.

    Parameter:
    - tree: NetworkX-Graph-Objekt, das den Baum darstellt.
    - nodes: Liste von Knoten, die hervorgehoben werden sollen (optional).
    - fails: Liste von fehlerhaften Kanten, die hervorgehoben werden sollen (optional).
    - current_edge: Aktuelle Kante, die hervorgehoben werden soll (optional).
    - showplot: Boolescher Wert, ob der Plot angezeigt (True) oder gespeichert (False) werden soll.
    """

    #print("[draw_tree] tree:",tree)
    #print("[draw_tree] tree.nodes:", tree.nodes)
    # Positionen der Knoten bestimmen
    pos = {node: tree.nodes[node]['pos'] for node in tree.nodes}  # Positionen der Knoten

    plt.figure(figsize=(10, 8))

    # Zeichne alle Kanten in Grau
    nx.draw_networkx_edges(tree, pos, edge_color='gray')

    # Zeichne fehlerhafte Kanten in Rot, falls vorhanden
    if fails:
        failed_edges = []
        for u, v in fails:
            if tree.has_edge(u, v):
                failed_edges.append((u, v))
            if tree.has_edge(v, u):
                failed_edges.append((v, u))

        nx.draw_networkx_edges(tree, pos, edgelist=failed_edges, edge_color='red', width=2)

    # Highlight aktuelle Kante in Blau, falls vorhanden
    if current_edge:
        if tree.has_edge(*current_edge):
            nx.draw_networkx_edges(tree, pos, edgelist=[current_edge], edge_color='blue', width=2)

    # Zeichne alle Knoten
    nx.draw_networkx_nodes(tree, pos, node_color='lightgray', node_size=500)
    nx.draw_networkx_labels(tree, pos)

    # Hervorheben spezieller Knoten in Orange, falls vorhanden
    if nodes:
        nx.draw_networkx_nodes(tree, pos, nodelist=nodes, node_color="orange", node_size=700)

    # Plot anzeigen oder speichern
    if showplot:
        plt.show()
    else:
        # Ordner "failedgraphs" erstellen, falls nicht vorhanden
        os.makedirs("failedgraphs", exist_ok=True)

        # Einzigartiger Dateiname basierend auf Zeitstempel
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        filename = f"failedgraphs/tree_{timestamp}.png"

        # Plot speichern
        plt.savefig(filename, format='png')
        plt.close()

        print(f"Graph gespeichert unter: {filename}")


import networkx as nx
import matplotlib.pyplot as plt
import os
from datetime import datetime

import networkx as nx
import matplotlib.pyplot as plt
import os
from datetime import datetime

import networkx as nx
import matplotlib.pyplot as plt
import os
from datetime import datetime
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import os
from datetime import datetime
import itertools

import networkx as nx
import matplotlib.pyplot as plt
import os
from datetime import datetime
import itertools

def draw_multipletree_with_highlights(trees, nodes=None, fails=None, current_edge=None, showplot=True, einzeln=True):
    """
    Zeichnet mehrere Bäume in einem gerichteten Graphen und hebt bestimmte Knoten, fehlerhafte Kanten und die aktuelle Kante hervor.

    Parameter:
    - trees: Liste von NetworkX-Graph-Objekten, die die Bäume darstellen.
    - nodes: Liste von Knoten, die hervorgehoben werden sollen (optional).
    - fails: Liste von fehlerhaften Kanten, die hervorgehoben werden sollen (optional).
    - current_edge: Aktuelle Kante, die hervorgehoben werden soll (optional).
    - showplot: Boolescher Wert, ob der Plot angezeigt (True) oder gespeichert (False) werden soll.
    - einzeln: Boolescher Wert, ob für jeden Baum eine eigene Abbildung erstellt werden soll (True) oder nicht (False).
    """

    # Zusammengefasster gerichteter Graph
    combined_graph = nx.DiGraph()
    pos = {}

    # Farbpalette für Bäume
    colors = itertools.cycle(plt.cm.tab10.colors)  # Wiederholbare Farben
    tree_colors = []

    # Iteriere über alle Bäume und füge Knoten und Kanten hinzu
    for tree in trees:
        tree_color = next(colors)  # Einzigartige Farbe für diesen Baum
        tree_colors.append((tree, tree_color))

        for node in tree.nodes:
            if node not in combined_graph:
                combined_graph.add_node(node, **tree.nodes[node])
                if 'pos' in tree.nodes[node]:
                    pos[node] = tree.nodes[node]['pos']

        for edge in tree.edges:
            if not combined_graph.has_edge(*edge):
                combined_graph.add_edge(*edge, color=tree_color)

    # Falls keine Positionen vorhanden sind, generiere sie automatisch
    if not pos:
        pos = nx.spring_layout(combined_graph)  # Automatische Positionierung

    if einzeln:
        for tree, tree_color in tree_colors:
            plt.figure(figsize=(12, 10))

            # Zeichne die Kanten des aktuellen Baums
            tree_edges = [edge for edge in tree.edges if combined_graph.has_edge(*edge)]
            nx.draw_networkx_edges(combined_graph, pos, edgelist=tree_edges, edge_color=[tree_color] * len(tree_edges), arrows=True)

            # Zeichne fehlerhafte Kanten in Rot (beide Richtungen), falls vorhanden
            if fails:
                failed_edges = []
                for u, v in fails:
                    if combined_graph.has_edge(u, v):
                        failed_edges.append((u, v))
                    if combined_graph.has_edge(v, u):
                        failed_edges.append((v, u))

                nx.draw_networkx_edges(combined_graph, pos, edgelist=failed_edges, edge_color='red', width=2, arrows=True)

            # Zeichne alle Knoten mit kleineren Punkten
            nx.draw_networkx_nodes(combined_graph, pos, node_color='lightgray', node_size=300)
            nx.draw_networkx_labels(combined_graph, pos)

            # Hervorheben spezieller Knoten in Orange, falls vorhanden
            if nodes:
                nx.draw_networkx_nodes(combined_graph, pos, nodelist=nodes, node_color="orange", node_size=400)

            # Plot anzeigen oder speichern
            if showplot:
                plt.show()
            else:
                # Ordner "failedgraphs" erstellen, falls nicht vorhanden
                os.makedirs("failedgraphs", exist_ok=True)

                # Einzigartiger Dateiname basierend auf Zeitstempel
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                filename = f"failedgraphs/tree_{timestamp}.png"

                # Plot speichern
                plt.savefig(filename, format='png')
                plt.close()

                print(f"Graph gespeichert unter: {filename}")
    else:
        plt.figure(figsize=(12, 10))

        # Zeichne alle Kanten für jeden Baum in seiner eigenen Farbe
        for tree, tree_color in tree_colors:
            tree_edges = [edge for edge in tree.edges if combined_graph.has_edge(*edge)]
            nx.draw_networkx_edges(combined_graph, pos, edgelist=tree_edges, edge_color=[tree_color] * len(tree_edges), arrows=True)

        # Zeichne fehlerhafte Kanten in Rot (beide Richtungen), falls vorhanden
        if fails:
            failed_edges = []
            for u, v in fails:
                if combined_graph.has_edge(u, v):
                    failed_edges.append((u, v))
                if combined_graph.has_edge(v, u):
                    failed_edges.append((v, u))

            nx.draw_networkx_edges(combined_graph, pos, edgelist=failed_edges, edge_color='red', width=2, arrows=True)

        # Highlight aktuelle Kante in Blau, falls vorhanden
        if current_edge:
            if combined_graph.has_edge(*current_edge):
                nx.draw_networkx_edges(combined_graph, pos, edgelist=[current_edge], edge_color='blue', width=2, arrows=True)

        # Zeichne alle Knoten mit kleineren Punkten
        nx.draw_networkx_nodes(combined_graph, pos, node_color='lightgray', node_size=300)
        nx.draw_networkx_labels(combined_graph, pos)

        # Hervorheben spezieller Knoten in Orange, falls vorhanden
        if nodes:
            nx.draw_networkx_nodes(combined_graph, pos, nodelist=nodes, node_color="orange", node_size=400)

        # Plot anzeigen oder speichern
        if showplot:
            plt.show()
        else:
            # Ordner "failedgraphs" erstellen, falls nicht vorhanden
            os.makedirs("failedgraphs", exist_ok=True)

            # Einzigartiger Dateiname basierend auf Zeitstempel
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            filename = f"failedgraphs/combined_graph_{timestamp}.png"

            # Plot speichern
            plt.savefig(filename, format='png')
            plt.close()

            print(f"Graph gespeichert unter: {filename}")
