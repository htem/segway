import networkx as nx
from daisy import Coordinate
import math
import json
import os
from os import path


# This file primary contains helper methods for
# synapse evaluation.

def remove_intraneuron_synapses(pred_graph):
    counter = 0
    for presyn, postsyn in list(pred_graph.edges):
        presyn_neuron = pred_graph.nodes[presyn]['seg_label']
        postsyn_neuron = pred_graph.nodes[postsyn]['seg_label']
        if presyn_neuron == postsyn_neuron:
            pred_graph.remove_nodes_from((presyn, postsyn))
            counter += 1
    print("%s synapses between non distinct cells removed" % counter)
    return pred_graph


def distance(coord_1, coord_2):
	diff = Coordinate(coord_1) - Coordinate(coord_2)
	return math.sqrt(sum(diff * diff))


# This helper method returns a subgraph view of the nodes
# corresponding to postsynaptic sites.
def postsyn_subgraph(syn_graph):
	postsyn_sites = [node for node in syn_graph if syn_graph.in_degree(node)]
	return syn_graph.subgraph(postsyn_sites)

# This helper method converts the daisy array coordinate of voxel
# to the coordinate at which it can be found in neuroglancer.
def daisy_zyx_to_voxel_xyz(daisy_zyx, voxel_size):
    voxel_zyx = Coordinate(daisy_zyx) / Coordinate(voxel_size)
    voxel_xyz = voxel_zyx[::-1]
    return voxel_xyz

# This helper method for extract_synapse_predictions converts
# the ndarray index of a voxel to its coordinate in the corresponding
# daisy array
def np_index_to_daisy_zyx(np_index, voxel_size, roi_offset):
    return Coordinate(voxel_size) * Coordinate(np_index) + roi_offset


# The methods below enable the storage of graph data in jsons
def syn_graph_to_json(graph, output_path):
    try:
        os.makedirs(output_path)
    except FileExistsError:
        pass
    basename = '{}_{}.json'.format(graph.graph['min_inference_value'],
                                   "syn_graph")
    syn_graph_json = path.join(output_path, basename)
    with open(syn_graph_json, "w") as f:
        json.dump(graph_to_dictionary(graph), f, indent=2)
    print("Graph saved as %s" % path.join(output_path, basename))
    


def graph_to_dictionary(graph):
    dictionary = {'attr': dict(graph.graph), 'nodes': {}}
    for node in graph.nodes:
        dictionary['nodes'][node] = {'attr': dict(graph.nodes[node]),
                                     'adj': dict(graph.adj[node])}
    return dictionary


def json_to_syn_graph(json_path):
    print("Loading inference graph from %s" % json_path)
    with open(json_path, "r") as f:
        graph_dict = json.load(f)
    graph =  dictionary_to_graph(graph_dict)
    print("%s potential synapses loaded" % (len(graph) // 2))
    return graph


def dictionary_to_graph(dictionary, directed=True):
    graph = nx.DiGraph(**dictionary['attr'])
    for node, data in dictionary['nodes'].items():
        graph.add_node(int(node), **data['attr'])
        graph.add_edges_from([(int(node), int(adj), attr) for (adj, attr)
                              in data['adj'].items()])
    return graph

def print_delimiter(char='-', length=70):
	delimeter = ""
	for i in range(length):
		delimeter += char
	print(delimeter)