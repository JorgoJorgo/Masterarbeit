##################### Routing Algorithms ##########################


# the routing with ONE checkpoint and with ONE tree first tries to route using 
# the face-routing from s -> cp and after that the tree-routing from cp -> d
from routing import RouteFaces, getRank
from trees import get_parent_node


def RouteWithOneCheckpointOneTreeCLUSTERED(graph,s,d,fails,paths):
    
   
    
    ####################ONLY FOR RUNNING WITH CLUSTERED FAILURES VIA THE CUSTOM RUN COMMAND#################
    #when running clustered failures the given algorithm computes clustered failures at the destination
    #to invert this, the source and destination are flipped when the routing starts
    s_old = s
    s = d
    d = s_old
    ##################################
    print()
    print("Routing with a checkpoint started for : ", s , " -> " , d) 
    
    detour_edges = []
    hops = 0
    switches = 0
    
    cp = paths[s][d]['cp']
    faces_cp_to_s  = paths[s][d]['faces_cp_to_s']
    edps_cp_to_s = paths[s][d]['edps_cp_to_s']
    tree_cp_to_d  = paths[s][d]['tree_cp_to_d']
    edps_cp_to_d   = paths[s][d]['edps_cp_to_d']
    
    routing_failure_faces = False
    #now the first step of the routing consists of face-routing from S to CP
    
    routing_failure_faces, hops_faces, switches_faces, detour_edges_faces = RouteFaces(s,cp,fails,faces_cp_to_s)
    
    if(routing_failure_faces):
        print("Routing failed via Faces from S to CP ")
        print(" ")
        return (True, hops_faces, switches_faces, detour_edges_faces)
    
    #since the routing for the trees was build prior to routing of the faces, the paths structure has changed
    #therefore the new paths structure needs to be converted to the old structure
    
    #new structure:
    #paths[source][destination] = {
    #                              'cp': cp,
    #                              'faces_cp_to_s': faces_cp_to_s, 
    #                              'edps_cp_to_s': edps_cp_to_s,
    #                              'tree_cp_to_d': tree_cp_to_d, 
    #                              'edps_cp_to_d': edps_cp_to_d,
    #                             } 
    
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
    
    if(routing_failure_tree):
        print("Routing failed via Tree from CP to D ")
        print(" ")
        return (True, hops_tree, switches_tree, detour_edges_tree)    
    
    #if both parts of the routing did not fail then the results of each one need to be combined
    
    hops = hops_faces + hops_tree
    switches = switches_faces + switches_tree
    detour_edges = []
    

    # Füge die Kanten aus der ersten Liste hinzu
    for edge in detour_edges_tree:
        detour_edges.append(edge)

    # Füge die Kanten aus der zweiten Liste hinzu
    for edge in detour_edges_faces:
        detour_edges.append(edge)
        
    print("Routing succesful with the Checkpoint")
    print('------------------------------------------------------')
    print(" ")
    return (False, hops, switches, detour_edges)


    
    
    
    #routing methode von onetree um zwischen einem source-destination paar zu routen
#dies geschieht indem man nach weiterleitung eines pakets an jedem knoten den nächst besten rang bestimmt
def RouteOneTreeCLUSTERED (graph,s,d,fails,paths):
    
    
    ####################ONLY FOR RUNNING WITH CLUSTERED FAILURES VIA THE CUSTOM RUN COMMAND#################
    # Assuming paths is a dictionary with keys as possible values
    keys = list(paths.keys())
    # Select a random key for s
    s = keys[0]
    neighbors = list(paths[s])
    print("Neighbors : ", neighbors)
    # Select a different random key for d
    d = keys[len(keys)-1]
    ##################################
    
    print("FAIL NUMBER : ", len(fails))
    if s != d :
        
        currentNode = -1
        edpIndex = 0
        detour_edges = []
        hops = 0
        switches = 0
        tree = paths[s][d]['tree']
        edps_for_s_d = paths[s][d]['edps']

        print('Routing started for ' , s , " to " , d )

        #als erstes anhand der EDPs (außer dem längsten, also dem letzten) versuchen zu routen
        for edp in edps_for_s_d:

            currentNode = s
            last_node = s 

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
        return (True, 0, 0, [])
    
    
    
    #Hier erfolgt die Ausführung von OneTree
