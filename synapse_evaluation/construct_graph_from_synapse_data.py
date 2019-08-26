import utility as util
import json
import time
from os import path
from itertools import product
import daisy
from daisy import Coordinate
import numpy as np
import networkx as nx
from scipy import ndimage
from skimage import measure


# This method uses the data in the provided catmaid skeleton json
# to construct a directed graph in which nodes represent pre and postsynaptic sites.
def load_synapses_from_catmaid_json(json_path):
    with open(json_path, 'r') as f:
        catmaid_data = json.load(f)
    conn_dict = {}
    syn_graph = nx.DiGraph()
    for sk_id, sk_dict in catmaid_data['skeletons'].items():
        for conn, attr in sk_dict['connectors'].items():
            if conn not in conn_dict:
                conn_dict[conn] = {}
                zyx_coord = (int(attr['location'][2]),
                             int(attr['location'][1]),
                             int(attr['location'][0]))
                conn_dict[conn]['zyx_coord'] = zyx_coord
                conn_dict[conn]['presyn_sites'] = set()
                conn_dict[conn]['postsyn_sites'] = set()   
            for presyn in attr['presnaptic_to']:
                (x, y, z) = sk_dict['treenodes'][str(presyn)]['location']
                syn_graph.add_node(presyn,
                                   zyx_coord=(int(z), int(y), int(x)),
                                   sk_id=sk_id)
                conn_dict[conn]['presyn_sites'].add(presyn)
            for postsyn in attr['postsynaptic_to']:
                (x, y, z) = sk_dict['treenodes'][str(postsyn)]['location']
                syn_graph.add_node(postsyn,
                                   zyx_coord=(int(z), int(y), int(x)),
                                   sk_id=sk_id)
                conn_dict[conn]['postsyn_sites'].add(postsyn)
    for conn, attr in conn_dict.items():
        for presyn, postsyn in product(attr['presyn_sites'],
                                       attr['postsyn_sites']):
            syn_graph.add_edge(presyn, postsyn,
                               conn_id=conn,
                               conn_zyx_coord=attr['zyx_coord'])
    return syn_graph



def construct_prediction_graph(synapse_data, segmentation_data,
                               extraction_config, voxel_size):
    pred_graph = extract_postsynaptic_sites(**synapse_data['postsynaptic'],
                                            voxel_size=voxel_size,
                                            min_inference_value=extraction_config['min_inference_value'])
    pred_graph.graph = {'segmentation': segmentation_data,
                        'min_inference_value': extraction_config['min_inference_value'],
                        'remove_intraneuron_synapses': extraction_config['remove_intraneuron_synapses'],
                        'voxel_size': voxel_size,
                        'filter_metric': extraction_config['filter_metric']}
    util.print_delimiter()
    pred_graph = extract_presynaptic_sites(pred_graph,
                                           **synapse_data['vector'],
                                           voxel_size=voxel_size)
    
    util.print_delimiter()
    pred_graph = add_segmentation_labels(pred_graph,
                                         **segmentation_data)
    if extraction_config['remove_intraneuron_synapses']:
        util.print_delimiter()
        pred_graph = remove_intraneuron_synapses(pred_graph)
    return pred_graph
    

# This method extracts the connected components from the inference
# array produced by the network. Each connected component receives
# various scores, each of which is positively correlated with its
# probability of being an actual synapse.
# TODO: Maybe consider using mask instead of regionprops
def extract_postsynaptic_sites(zarr_path, dataset,
                               voxel_size, min_inference_value):
    print("Extracting postsynaptic sites from {}".format(dataset),
          "at min inference value of {}".format(min_inference_value))
    start_time = time.time()
    prediction_ds = daisy.open_ds(zarr_path, dataset)
    roi = prediction_ds.roi
    
    inference_array = prediction_ds.to_ndarray(roi)
    labels, _ = ndimage.label(inference_array > min_inference_value)
    extracted_syns = measure.regionprops(labels, inference_array)
    
    syn_graph = nx.DiGraph()
    for i, syn in enumerate(extracted_syns):
        syn_id = i + 1
        centroid_index = tuple(int(index) for index in syn.centroid)
        zyx_coord = util.np_index_to_daisy_zyx(centroid_index,
                                               voxel_size,
                                               roi.get_offset())
        syn_graph.add_node(syn_id,
                           zyx_coord = zyx_coord,
                           max=int(syn.max_intensity),
                           area=int(syn.area),
                           mean=int(syn.mean_intensity),
                           sum=int(syn.area * syn.mean_intensity))
        zyx = syn_graph.nodes[syn_id]['zyx_coord']
    print("Extraction took {} seconds".format(time.time()- start_time))
    return syn_graph


# presynaptic site nodes are negative
# the fact that this returns post synaptic sites is confusing
def extract_presynaptic_sites(pred_graph, zarr_path,
                              dataset, voxel_size):
    print("Extracting vector predictions from {}".format(dataset))
    start_time = time.time()
    prediction_ds = daisy.open_ds(zarr_path, dataset)
    prediction_ds = prediction_ds[prediction_ds.roi]
    postsyns = list(pred_graph.nodes(data='zyx_coord'))
    for i, (postsyn_id, postsyn_zyx) in enumerate(postsyns):
        presyn_id = postsyn_id * -1
        vector = prediction_ds[postsyn_zyx]
        presyn_zyx = tuple(Coordinate(vector) *
                          Coordinate(voxel_size) +
                          Coordinate(postsyn_zyx))
        pred_graph.add_node(presyn_id,
                            zyx_coord=presyn_zyx)
        pred_graph.add_edge(presyn_id, postsyn_id)
    print("Extraction took {} seconds".format(time.time()- start_time))
    return pred_graph


def add_segmentation_labels(graph, zarr_path, dataset):
    print("Adding segmentation labels from {}".format(dataset))
    print("{} nodes to label".format(graph.number_of_nodes()))
    start_time = time.time()
    segment_array = daisy.open_ds(
        zarr_path,
        dataset)
    segment_array = segment_array[segment_array.roi]
    nodes_outside_roi = []  
    for i, (treenode_id, attr) in enumerate(graph.nodes(data=True)):
        try:    
            attr['seg_label'] = int(segment_array[daisy.Coordinate(attr['zyx_coord'])])
        except AssertionError:
            nodes_outside_roi.append(treenode_id)
        if i == (graph.number_of_nodes() // 2):
            print("%s seconds remaining" % (time.time() - start_time))
    for node in nodes_outside_roi:
        graph.remove_node(node)
    print("Segmentation labels added in %s seconds" % (time.time() - start_time))
    return graph

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