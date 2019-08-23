from construct_graph_from_synapse_data import \
    load_synapses_from_catmaid_json, \
    add_segmentation_labels, \
    construct_prediction_graph
import utility as util

import numpy as np
import networkx as nx
from daisy import Coordinate
import matplotlib.pyplot as plt
import json
import argparse
import os
from os import path
from itertools import combinations
from functools import partial
from multiprocessing import Pool
import math

# TODO
# 1) more thoughtful method names
# 2) method headers
# 3) file headers? 

# This method extracts the parameter values specified in the json
# provided by the client. Unspecified parameters take on the values
# specified in segway/synapse_evaluation/synapse_task_defaults.json
def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('user_path')
    args = parser.parse_args()
    
    params = {}
    param_defaults_dir = path.dirname(path.realpath(__file__))
    param_defaults_path = path.join(param_defaults_dir,
                                    'parameter_defaults.json')
    with open(param_defaults_path, 'r') as f:
        params = json.load(f)
    with open(args.user_path, 'r') as f:
        user_configs = json.load(f)
    for key in params:
        try:
            params[key].update(user_configs[key])
        except KeyError:
            pass
    return format_params(params, args.user_path)

# This code is here to spare the user the inconvenience of
# specifying the parameters in the format used by the program and
# infer the values of parameters that need not be specified explicitly.
def format_params(params, user_path):
    if 'output_path' not in params['Output']:
        output_dir = path.dirname(user_path)
        config_file_name = path.splitext(path.basename(user_path))[0]
        params['Output']['output_path'] = path.join(output_dir,
                                                    config_file_name +
                                                    '_outputs')
    if 'inference_graph_json' not in params['Input']:
        pred_graph_base = '{}_syn_graph.json'.format(params['Extraction']['min_inference_value'])
        pred_graph_path = path.join(params['Output']['output_path'], pred_graph_base)
        if path.exists(pred_graph_path):
            params['Input']['inference_graph_json'] = pred_graph_path
    for key in ['postsynaptic', 'vector', 'segmentation']:
        if key not in params['Input']:
            params['Input'][key] = {}
            params['Input'][key]['zarr_path'] = params['Input']['zarr_path']
            params['Input'][key]['datset'] = params['Input'][key+'_dataset']
    print("Config loaded from {}".format(user_path))
    return params


def plot_metric(graph, metric, output_path):
    min_inference_value = graph.graph['min_inference_value']
    if isinstance(metric, str):
        plot_title = '{}_{}_hist'.format(metric, min_inference_value)
        util.plot_node_attr_hist(graph, metric, output_path, plot_title)
        print("Plotted histogram of %s values" % metric)
    else:
        plot_title = '{}_{}_{}_scatter'.format(metric[0], metric[1],
                                               min_inference_value)
        util.plot_node_attr_scatter(graph, metric[0], metric[1],
                                    output_path, plot_title)
        print("Plotted scatterplot of %s as a function of %s"
               % (metric[1], metric[0]))


# This helper method for apply_score_filterreturns true if
# all the scores received by the potential synapse being
# evaluated exceed the cutoff values.
# TODO: give more descriptive name.
def meets_criteria(attr, score_filters):
    return all([attr[metric] >= cutoff for
                metric, cutoff in score_filters.items()])

# This method returns all the nodes failing to exceed
# the minimum score thresholds specified by the filter
def apply_score_filter(postsyn_graph, percentile=None, score_filter=None):
    if percentile:
        print("Applying {}th percentile score filter".format(percentile))
        score_filter = {}
        for metric in ['area', 'sum', 'mean', 'max']:
            score_filter[metric] = np.percentile([attr[metric] for node, attr in 
                                                  postsyn_graph.nodes(data=True)],
                                                  percentile)
    else:
        print("Applying filter %s" % score_filter)
    filtered_nodes = {node for node, attr in postsyn_graph.nodes(data=True)
                      if not meets_criteria(attr, score_filter)}
    print("%s potential synapses removed from consideration" % len(filtered_nodes))
    filtered_nodes.update({(-1 * node) for node in filtered_nodes})
    return filtered_nodes

def get_nearby_matches(gt_syn, pred_graph, max_dist):
    (gt_presyn, gt_postsyn) = gt_syn
    attr = pred_graph.nodes(data=True)
    matches = {}
    for pred_syn in pred_graph.edges:
        pred_presyn = attr[pred_syn[0]]
        pred_postsyn = attr[pred_syn[1]]
        if pred_presyn['seg_label'] == gt_presyn['seg_label'] \
                and pred_postsyn['seg_label'] == gt_postsyn['seg_label']:
            presyn_dist = util.distance(pred_presyn['zyx_coord'],
                                        gt_presyn['zyx_coord'])
            postsyn_dist = util.distance(pred_postsyn['zyx_coord'],
                                         gt_postsyn['zyx_coord'])
            if presyn_dist <= max_dist and postsyn_dist <= max_dist:
                matches[pred_syn] = presyn_dist + postsyn_dist
    return matches

def write_matches_to_file(conn_dict,
                          filtered_graph,
                          extraction,
                          percentile,
                          output_path,
                          voxel_size):
    basename = 'mininf{}_maxdist{}_percent{}.txt'.format(extraction['min_inference_value'],
                                     extraction['max_distance'],
                                     percentile)
    output_file = path.join(output_path, basename)
    print("Writing matches file: {}".format(output_file))
    with open(output_file, "w") as f:
        print("Min inference value = {}".format(extraction['min_inference_value']), file=f)
        print("Applied filter {}".format(percentile), file=f)
        print("Maximum distance of {}".format(extraction['max_distance']), file=f)
        pred_node_attr = filtered_graph.nodes(data=True)
        for cell_pair, gt_syns in conn_dict.items():
            print("Connections between {} and {}:".format(*cell_pair), file=f)
            for gt_syn, syn_attr in gt_syns.items():
                conn_id = syn_attr['conn_id']
                conn_xyz = syn_attr['conn_zyx_coord'][::-1]
                print("\tCATMAID connector {} {}".format(conn_id, conn_xyz), file=f)
                
                if not len(syn_attr['matches']):
                    print("\t\tNone", file=f)
                dist_sorted_matches = sorted(syn_attr['matches'].items(),
                                             key=lambda match: match[1])
                for (presyn_match, postsyn_match), dist in dist_sorted_matches:
                    pred_postsyn_zyx = pred_node_attr[postsyn_match]['zyx_coord']
                    pred_postsyn_coord = util.daisy_zyx_to_voxel_xyz(pred_postsyn_zyx,
                                                                     voxel_size) 
                    print("\t\tPredicted match at {}".format(pred_postsyn_coord), file=f)
                    
                    pred_presyn_zyx = pred_node_attr[presyn_match]['zyx_coord']
                    pred_presyn_coord = util.daisy_zyx_to_voxel_xyz(pred_presyn_zyx,
                                                                    voxel_size) 
                    print("\t\t\tPredicted presynaptic site at {}".format(pred_presyn_coord), file=f)
                    scores = pred_node_attr[postsyn_match]
                    for metric in ['area', 'sum', 'mean', 'max']:
                        print("\t\t\t{} = {}".format(metric, scores[metric]), file=f)
                print(file=f)


def plot_false_pos_false_neg(error_counts,
                             plot_title,
                             output_path):
    false_neg = []
    false_pos = []
    for percentile, counts in error_counts.items():
        false_neg.append(counts['false_neg'])
        false_pos.append(counts['false_pos'])
    plt.scatter(false_neg, false_pos, label=percentile)
    plt.title(plot_title)
    plt.xlabel('false negatives')
    plt.ylabel('false postives')
    plt.legend()
    plt.savefig(path.join(output_path, plot_title))
    plt.clf()

if __name__ == '__main__':
    params = parse_configs()
    output_path = params['Output']['output_path']
    try:
        os.makedirs(output_path)
    except FileExistsError:
        pass
    print("Output path:", output_path)

    util.print_delimiter()
    gt_graph = load_synapses_from_catmaid_json(params['Input']['skeleton'])
    gt_graph = add_segmentation_labels(gt_graph,
                                       **params['Input']['segmentation'])
    util.print_delimiter()
    pred_graph = construct_prediction_graph(params['Input'],
                                            params['Extraction'])
    util.print_delimiter()
    util.syn_graph_to_json(pred_graph,
                           output_path)
    

    util.print_delimiter()
    plot_dir = path.join(output_path, 'plots')
    try:
        os.makedirs(plot_dir)
    except FileExistsError:
        pass
    for metric in params['Output']['metric_plots']:
        plot_metric(util.postsyn_subgraph(pred_graph),
                    metric,
                    plot_dir)
    
    match_base = 'mininf{}_maxdist{}'.format(params['Extraction']['min_inference_value'],
                                             params['Extraction']['max_distance'])
    match_dir = path.join(output_path, match_base)
    error_counts = {}
    try:
        os.makedirs(match_dir)
    except FileExistsError:
        pass
    for percentile in params['Extraction']['percentiles']:
        util.print_delimiter()
        filtered_nodes = apply_score_filter(util.postsyn_subgraph(pred_graph),
                                            percentile)
        filtered_graph = pred_graph.subgraph([node for node in pred_graph 
                                              if node not in filtered_nodes])    
        get_matches = partial(get_nearby_matches,
                              pred_graph=filtered_graph,
                              max_dist=params['Extraction']['max_distance'])
        for presyn, postsyn in gt_graph.edges():
            gt_graph[presyn][postsyn]['matches'] = \
                    get_matches((gt_graph.nodes[presyn],
                                 gt_graph.nodes[postsyn]))
        error_counts[percentile] = {}
        num_true_pos = len([(pre, post) for pre, post, match in
                            gt_graph.edges(data='matches') if len(match)])
        error_counts[percentile]['true_pos'] = num_true_pos
        error_counts[percentile]['false_pos'] = filtered_graph.number_of_edges() - num_true_pos
        error_counts[percentile]['false_neg'] = gt_graph.number_of_edges() - num_true_pos
        write_matches_to_file(util.neuron_pairs_dict(gt_graph),
                              filtered_graph,
                              params['Extraction'],
                              percentile,
                              match_dir,
                              params['Input']['voxel_size'])
    util.print_delimiter()
    plot_false_pos_false_neg(error_counts, match_base, plot_dir)
    print("Errors plotted")
    print("Complete")
