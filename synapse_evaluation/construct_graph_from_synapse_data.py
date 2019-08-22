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


def generate_score_plot(pred_graph, output_path, metric_1, metric_2=None):
    metric_1_vals = [attr[metric_1] for node, attr in pred_graph.nodes(data=True)]
    min_inference_value = pred_graph.graph['min_inference_value']
    if metric_2:
        metric_2_vals = [attr[metric_2] for node, attr in pred_graph.nodes(data=True)]
        plt.scatter(metric_1_vals, metric_2_vals)
        plot_title = '{}_{}_{}_{}'.format(metric_1, metric_2, min_inference_value, 'scatter')
        plt.title(plot_title)
        plt.xlabel(metric_1)
        plt.ylabel(metric_2)
        plt.savefig(path.join(output_path, plot_title))
        plt.clf()
    else:
        max_val = int(max(metric_1_vals))
        num_bins = 256
        if max_val < num_bins:
            hist_bins = range(num_bins)
        else:
            hist_bins = [i * max_val // num_bins for i in range(num_bins)]
        plt.hist(metric_1_vals,
                 bins=hist_bins,
                 density=True)
        plot_title = '{}_{}_{}'.format(metric_1, min_inference_value, 'hist')
        plt.title(plot_title)
        plt.savefig(path.join(output_path, plot_title))
        plt.clf()


def plot_metrics(pred_graph, output_path, metrics):
    try:
        os.makedirs(output_path)
    except FileExistsError:
        pass
    plot_metric = partial(generate_score_plot,
                          pred_graph=util.postsyn_subgraph(pred_graph),
                          output_path=output_path)
    for metric in metrics:
        if isinstance(metric, str):
            plot_metric(metric_1=metric)
            print("Plotted histogram of %s values" % metric)
        else:
            plot_metric(metric_1=metric[0],
                        metric_2=metric[1])
            print("Plotted scatterplot of %s as a function of %s" % (metric[1], metric[0]))
# inp = input parameters
# outp = output parameters
# extp = extraction parameters
def construct_prediction_graph(inp, outp, extp):
    if 'inference_graph_json' in inp:
        pred_graph = util.json_to_syn_graph(inp['inference_graph_json'])
    else:
        pred_graph = extract_postsynaptic_sites(inp['postsynaptic']['zarr_path'],
                                                inp['postsynaptic']['dataset'],
                                                inp['voxel_size'],
                                                extp['min_inference_value'])
        pred_graph.graph = {'segmentation': inp['segmentation'],
                            'min_inference_value': extp['min_inference_value'],
                            'remove_intraneuron_synapses': extp['remove_intraneuron_synapses'],
                            'voxel_size': inp['voxel_size']}
        util.print_delimiter()
        plot_metrics(pred_graph,
                     outp['output_path'],
                     outp['metric_plots'])
        util.print_delimiter()
        pred_graph = extract_presynaptic_sites(pred_graph,
                                               inp['vector']['zarr_path'],
                                               inp['vector']['dataset'],
                                               inp['voxel_size'])
        
        util.print_delimiter()
        pred_graph = add_segmentation_labels(pred_graph,
                                             inp['segmentation']['zarr_path'],
                                             inp['segmentation']['dataset'])
        if extp['remove_intraneuron_synapses']:
            util.print_delimiter()
            pred_graph = util.remove_intraneuron_synapses(pred_graph)
        
        util.print_delimiter()
        util.syn_graph_to_json(pred_graph,
                               outp['output_path'])
    return pred_graph
    

# This method extracts the connected components from the inference
# array produced by the network. Each connected component receives
# various scores, each of which is positively correlated with its
# probability of being an actual synapse.
# TODO: Maybe consider using mask instead of regionprops
def extract_postsynaptic_sites(zarr_path,
                               dataset,
                               voxel_size,
                               min_inference_value):
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
def extract_presynaptic_sites(pred_graph,
                              zarr_path,
                              dataset,
                              voxel_size):
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


def add_segmentation_labels(graph,
                            zarr_path,
                            dataset):
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

