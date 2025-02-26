from datetime import datetime
from platform import node
import sys
import time
from traceback import print_list
import traceback
from typing import List, Any, Union
import random
import matplotlib
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
import os

from faces import find_faces_pre
from trees import all_edps, multiple_trees

def squareOne_with_cuts_pre(graph):
    paths = {}
    print("[SQ1 with Cuts] Start Precomputation")


    for source in graph.nodes:
       
        for destination in graph.nodes:
            
            if source != destination:
                print(f"[SQ1 with Cuts] Precomputation: {source} -> {destination}")
                
                cut_edges = nx.minimum_edge_cut(graph, source, destination)

            
                cut_nodes = set()
                for edge in cut_edges:
                    cut_nodes.add(edge[0])
                    cut_nodes.add(edge[1])

                print("[SQ1 with Cuts] Cut Nodes", cut_nodes)
                
                structure = squareOne_with_cuts(source,destination,graph,cut_edges,cut_nodes)

                if source in paths:
                    paths[source][destination] = { 
                                                    'structure': structure,
                                                    'cut_edges': cut_edges,
                                                    'cut_nodes': cut_nodes
                                                }
                else:
                    paths[source] = {}
                    paths[source][destination] = {
                                                'structure': structure,
                                                'cut_edges': cut_edges,
                                                'cut_nodes': cut_nodes
                    }

    input("Press Enter to continue...")
    return paths


def squareOne_with_cuts(source, destination, graph, cut_edges, cut_nodes):
    
    structure_from_s = nx.Graph()
    structure_from_d = nx.Graph()
    combined_structure = nx.Graph()

    # Add source and destination nodes to the cut nodes set
    cut_nodes.add(source)
    cut_nodes.add(destination)

    # First add the cut edges and cut nodes to the structures 
    for node in cut_nodes:
        structure_from_s.add_node(node)
        structure_from_d.add_node(node)

    for edge in cut_edges:
        structure_from_s.add_edge(*edge)
        structure_from_d.add_edge(*edge)

    for node in cut_nodes:
        if source != node:
            all_edps_from_s_to_cut_node_i = all_edps(source, node, graph)
            for edp in all_edps_from_s_to_cut_node_i:
                for i in range(len(edp) - 1):
                    structure_from_s.add_edge(edp[i], edp[i + 1])

        if destination != node:
            all_edps_from_d_to_cut_node_i = all_edps(destination, node, graph)
            for edp in all_edps_from_d_to_cut_node_i:
                for i in range(len(edp) - 1):
                    structure_from_d.add_edge(edp[i], edp[i + 1])


    for node in structure_from_s.nodes:
        structure_from_s.nodes[node]['pos'] = graph.nodes[node]['pos']

    for node in structure_from_d.nodes:
        structure_from_d.nodes[node]['pos'] = graph.nodes[node]['pos']

    all_edps_from_s_to_d = all_edps(source, destination, graph)

    for edp in all_edps_from_s_to_d:
        for i in range(len(edp) - 1):
            combined_structure.add_edge(edp[i], edp[i + 1])
            combined_structure.nodes[edp[i]]['pos'] = graph.nodes[edp[i]]['pos']   
            combined_structure.nodes[edp[i + 1]]['pos'] = graph.nodes[edp[i + 1]]['pos']


    #print("[SQ1 with Cuts] Structure from S", structure_from_s.nodes)
    #print("[SQ1 with Cuts] Structure from D", structure_from_d.nodes)

    #print("[SQ1 with Cuts] Adding Edges from S to Combined")
    for edge in structure_from_s.edges:
        fake_combined_structure = combined_structure.copy()
        fake_combined_structure.add_edge(*edge)
        edge_accepted = False
        faces = find_faces_pre(combined_structure,source,destination)
        for face in faces:
            if source in face and destination in face:
                edge_accepted = True
                break

        if edge_accepted:
            combined_structure.add_edge(*edge)
            combined_structure.nodes[edge[0]]['pos'] = graph.nodes[edge[0]]['pos']
            combined_structure.nodes[edge[1]]['pos'] = graph.nodes[edge[1]]['pos']


    #print("[SQ1 with Cuts] Adding Edges from D to Combined")

    for edge in structure_from_d.edges:
        fake_combined_structure = combined_structure.copy()
        fake_combined_structure.add_edge(*edge)
        edge_accepted = False
        faces = find_faces_pre(combined_structure,source,destination)
        for face in faces:
            if source in face and destination in face:
                edge_accepted = True
                break

        if edge_accepted:
            combined_structure.add_edge(*edge)
            combined_structure.nodes[edge[0]]['pos'] = graph.nodes[edge[0]]['pos']
            combined_structure.nodes[edge[1]]['pos'] = graph.nodes[edge[1]]['pos']

    # Add nodes and edges from structure_from_d to combined_structure
    for node in structure_from_d.nodes:
        combined_structure.add_node(node)
        combined_structure.nodes[node]['pos'] = graph.nodes[node]['pos']

    return combined_structure

def multipleTrees_with_cuts_pre(graph):
    paths = {}
    print("[MT with Cuts] Start Precomputation")


    for source in graph.nodes:
       
        for destination in graph.nodes:
            
            if source != destination:
                cut_edges = nx.minimum_edge_cut(graph, source, destination)
                #print("[MT with Cuts] building structure for", source, destination)
            
                cut_nodes = set()
                for edge in cut_edges:
                    cut_nodes.add(edge[0])
                    cut_nodes.add(edge[1])

                
                structure = multipleTrees_with_cuts(source,destination,graph,cut_edges,cut_nodes)

                if source in paths:
                    paths[source][destination] = { 
                                                    'structure': structure,
                                                    'cut_edges': cut_edges,
                                                    'cut_nodes': cut_nodes
                                                }
                else:
                    paths[source] = {}
                    paths[source][destination] = {
                                                'structure': structure,
                                                'cut_edges': cut_edges,
                                                'cut_nodes': cut_nodes
                    }

                #print_cut_structure(
                #    highlighted_nodes=list(cut_nodes),
                #    cut_edges=cut_edges,
                #    structure=structure,
                #    source=source,
                #    destination=destination,
                #    save_plot=False,
                #    filename=f"failedgraphs/graph_{source}_{destination}.png"
                #)
    return paths


def multipleTrees_with_cuts(source, destination, graph, cut_edges, cut_nodes):
    
    structure_from_s = nx.Graph()
    structure_from_d = nx.Graph()
    combined_structure = nx.Graph()

    # Add source and destination nodes to the cut nodes set
    cut_nodes.add(source)
    cut_nodes.add(destination)

    # First add the cut edges and cut nodes to the structures 
    for node in cut_nodes:
        structure_from_s.add_node(node)
        structure_from_d.add_node(node)

    for edge in cut_edges:
        structure_from_s.add_edge(*edge)
        structure_from_d.add_edge(*edge)

    for node in cut_nodes:
        if source != node:

            #hier muss rein das die Bäume zu dem CutNode gespeichert werden
            all_edps_from_s_to_cut_node_i = all_edps(source, node, graph)

            temp_structure = multiple_trees(source, node, graph, all_edps_from_s_to_cut_node_i)
            #print("Temp Structure", temp_structure)
            for tree in temp_structure:
                for edge in tree.edges:
                    structure_from_s.add_edge(edge[0], edge[1])

        if destination != node:
            all_edps_from_d_to_cut_node_i = all_edps(destination, node, graph)
            temp_structure = multiple_trees(destination, node, graph, all_edps_from_d_to_cut_node_i)
            #print("Temp Structure", temp_structure)
            for tree in temp_structure:
                for edge in tree.edges:
                    structure_from_d.add_edge(edge[0], edge[1])

    # Add nodes and edges from structure_from_s to combined_structure
    for node in structure_from_s.nodes:
        combined_structure.add_node(node)
        combined_structure.nodes[node]['pos'] = graph.nodes[node]['pos']

    for edge in structure_from_s.edges:
        combined_structure.add_edge(*edge)

    # Add nodes and edges from structure_from_d to combined_structure
    for node in structure_from_d.nodes:
        combined_structure.add_node(node)
        combined_structure.nodes[node]['pos'] = graph.nodes[node]['pos']

    for edge in structure_from_d.edges:
        combined_structure.add_edge(*edge)

    #print_cut_structure(source=source, destination=destination, structure=combined_structure, highlighted_nodes=[source,destination], cut_edges=cut_edges)

    return combined_structure



import os
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime

def print_cut_structure(highlighted_nodes, cut_edges, structure, source, destination, fails=[], current_edge=None, save_plot=False, filename="graphen/graph.png"):
    pos = nx.get_node_attributes(structure, 'pos')
    
    plt.figure(figsize=(10, 10))
    
    # Zeichne den gesamten Graphen mit normalen Kanten in Schwarz
    nx.draw(structure, pos, with_labels=True, node_color='lightblue', edge_color='black', node_size=500, font_size=10)
    
    # Markiere hervorgehobene Knoten
    nx.draw_networkx_nodes(structure, pos, nodelist=highlighted_nodes, node_color='red')
    
    # Markiere Cut-Kanten in Grün
    nx.draw_networkx_edges(structure, pos, edgelist=cut_edges, edge_color='green', width=2)
    
    # Markiere Source- und Destination-Knoten
    nx.draw_networkx_nodes(structure, pos, nodelist=[source], node_color='green')
    nx.draw_networkx_nodes(structure, pos, nodelist=[destination], node_color='yellow')
    
    # Markiere die aktuelle Kante, falls vorhanden
    if current_edge:
        nx.draw_networkx_edges(structure, pos, edgelist=[current_edge], edge_color='blue', width=2)
    
    # **Fix: Fail-Kanten in beide Richtungen prüfen**
    valid_fails = [(u, v) for (u, v) in structure.edges if (u, v) in fails or (v, u) in fails]
    
    if valid_fails:
        nx.draw_networkx_edges(structure, pos, edgelist=valid_fails, edge_color='red', width=2)
    
    if save_plot:
        os.makedirs("failedgraphs", exist_ok=True)
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"failedgraphs/graph_{source}_{destination}_{current_time}.png"
        plt.savefig(filename)
    else:
        plt.show()

    plt.close()
