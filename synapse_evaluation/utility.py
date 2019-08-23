import networkx as nx
from daisy import Coordinate
import math
import json
import os
from os import path
import matplotlib.pyplot as plt

# This file primary contains helper methods for
# synapse evaluation.

def plot_node_attr_scatter(graph, attr_1, attr_2,
                           output_path, plot_title):
    plt.scatter([val for node, val in graph.nodes(data=attr_1)],
                [val for node, val in graph.nodes(data=attr_2)])
    plt.title(plot_title)
    plt.xlabel(attr_1)
    plt.ylabel(attr_2)
    plt.savefig(path.join(output_path, plot_title))
    plt.clf()

def plot_node_attr_hist(graph, attr, output_path, plot_title):
    plt.hist([val for node, val in graph.nodes(data=attr)])
    plt.title(plot_title)
    plt.savefig(path.join(output_path, plot_title))
    plt.clf()

def postsyn_subgraph(syn_graph):
    postsyn_sites = [node for node in syn_graph if syn_graph.in_degree(node)]
    return syn_graph.subgraph(postsyn_sites)

def neuron_pairs_dict(syn_graph):
	conn_dict = {}
	neuron_ids = syn_graph.nodes(data='seg_label')
	for presyn, postsyn, attr in syn_graph.edges(data=True):
		neuron_pair = (neuron_ids[presyn], neuron_ids[postsyn])
		if neuron_pair not in conn_dict:
			conn_dict[neuron_pair] = {}
		conn_dict[neuron_pair][(presyn, postsyn)] = syn_graph[presyn][postsyn]
	return conn_dict


def print_delimiter(char='-', length=80):
	delimeter = ""
	for i in range(length):
		delimeter += char
	print(delimeter)

#### Methods for working with coordinates ####
def distance(coord_1, coord_2):
	diff = Coordinate(coord_1) - Coordinate(coord_2)
	return math.sqrt(sum(diff * diff))

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


#### Methods for writing/reading json representations of nx graphs ####
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
