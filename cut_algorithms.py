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
import os

from trees import all_edps

def squareOne_with_cuts_pre(graph):
    paths = {}
    print("[SQ1 with Cuts] Start Precomputation")


    for source in graph.nodes:
       
        for destination in graph.nodes:
            
            if source != destination:
                print("[SQ1 with Cuts] S", source)
                print("[SQ1 with Cuts] D", destination)
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

                print_cut_structure(
                    highlighted_nodes=list(cut_nodes),
                    cut_edges=cut_edges,
                    structure=structure,
                    source=source,
                    destination=destination,
                    save_plot=False,
                    filename=f"failedgraphs/graph_{source}_{destination}.png"
                )
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

    return combined_structure

def print_cut_structure(highlighted_nodes, cut_edges, structure, source, destination, fails=[], current_edge=None, save_plot=False, filename="failedgraphs/graph.png"):
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
    if fails:
        nx.draw_networkx_edges(structure, pos, edgelist=fails, edge_color='black', width=2, style='dotted')
    
    if save_plot:
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        plt.savefig(filename)
    else:
        plt.show()