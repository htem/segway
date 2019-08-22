from construct_graph_from_synapse_data import \
    load_synapses_from_catmaid_json, \
    add_segmentation_labels, \
    construct_prediction_graph
import utility as util

import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from daisy import Coordinate

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
def extract_parameters_from_config():
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

    # These code blocks are here to spare the user the inconvenience of
    # specifying the parameters in the format used by the program and
    # infer the values of parameters that need not be specified explicitly.
    if 'output_path' not in params['Output']:
        output_dir = path.dirname(args.user_path)
        config_file_name = path.splitext(path.basename(args.user_path))[0]
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
    print("Config loaded from {}".format(args.user_path))
    
    return params


# This helper method for apply_score_filterreturns true if
# all the scores received by the potential synapse being
# evaluated exceed the cutoff values.
# TODO: give more descriptive name.
def meets_criteria(attr, score_filters):
    return all([attr[metric] >= cutoff for
                metric, cutoff in score_filters.items()])


# This method returns all the nodes failing to exceed
# the minimum score threshold
def apply_score_filter(postsyn_graph, score_filter):
    for metric, cutoff in score_filter.items():
        if cutoff < 1:
            score_filter[metric] = np.percentile([attr[metric] for node, attr in 
                                                  postsyn_graph.nodes(data=True)],
                                                  cutoff * 100)
    print("Applying filter %s" % score_filter)
    filtered_nodes = [node for node, attr in postsyn_graph.nodes(data=True)
                      if not meets_criteria(attr, score_filter)]
    print("%s potential synapses removed from consideration" % len(filtered_nodes))
    filtered_nodes.extend([-1 * node for node in filtered_nodes])
    return filtered_nodes

    
def get_closest_matching_edge(gt_syn, pred_graph, max_dist=math.inf):
    (gt_presyn, gt_postsyn) = gt_syn
    analogue = None
    shortest_dist = 2 * max_dist
    num_matches = 0
    for pred_syn in pred_graph.edges:
        pred_presyn = pred_graph.nodes[pred_syn[0]]
        pred_postsyn = pred_graph.nodes[pred_syn[1]]
        if pred_presyn['seg_label'] == gt_presyn['seg_label'] \
                and pred_postsyn['seg_label'] == gt_postsyn['seg_label']:
            num_matches += 1
            presyn_dist = util.distance(pred_presyn['zyx_coord'],
                                        gt_presyn['zyx_coord'])
            postsyn_dist = util.distance(pred_postsyn['zyx_coord'],
                                         gt_postsyn['zyx_coord'])
            if presyn_dist <= max_dist and postsyn_dist <= max_dist:
                if (presyn_dist + postsyn_dist) < shortest_dist:
                    analogue = pred_syn
                    shortest_dist = presyn_dist + postsyn_dist
    return analogue, shortest_dist, num_matches

def get_nearby_matches(gt_syn, pred_graph, max_dist=math.inf):
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
def find_match_predictions(gt_graph,
                           filtered_graph,
                           outp,
                           extp):
    matches = {}
    attr = gt_graph.nodes(data=True)
    for gt_presyn, gt_postsyn in gt_graph.edges():
        neuron_pair = (attr[gt_presyn]['seg_label'], attr[gt_postsyn]['seg_label'])
        if neuron_pair not in matches:
            matches[neuron_pair] = {}
        matches[neuron_pair][(gt_presyn, gt_postsyn)] = get_nearby_matches((attr[gt_presyn],
                                                                            attr[gt_postsyn]),
                                                                           filtered_graph,
                                                                           extp['max_distance'])
    output_file = "{}_{}".format(extp['min_inference_value'],
                                 extp['max_distance'])
    for metric, cutoff in extp['score_filter'].items():
        output_file += "_{}_{}".format(metric, str(cutoff))
    output_file += ".txt"
    with open(path.join(outp['output_path'], output_file), "w") as f:
        print(extp, file=f)
        for cell_pair, gt_syns in matches.items():
            print(cell_pair, file=f)
            for gt_syn, pred_matches in gt_syns.items():
                print("\t{}".format(gt_syn), file=f)
                if len(pred_matches):
                    for pred_match, dist in pred_matches.items():
                        print("\t\t{}".format(pred_match), dist, file=f)
                else:
                    print("\t\tNone", file=f)
        print(file=f)   


if __name__ == '__main__':
    params = extract_parameters_from_config()
    inp = params['Input']
    outp = params['Output']
    extp = params['Extraction']
    print("Output path:", outp['output_path'])

    util.print_delimiter()
    gt_graph = load_synapses_from_catmaid_json(inp['skeleton'])
    gt_graph = add_segmentation_labels(gt_graph,
                                       inp['segmentation']['zarr_path'],
                                       inp['segmentation']['dataset'])
    util.print_delimiter()
    pred_graph = construct_prediction_graph(inp, outp, extp)
    
    util.print_delimiter()
    filtered_nodes = apply_score_filter(util.postsyn_subgraph(pred_graph),
                                        extp['score_filter'])
    filtered_graph = pred_graph.subgraph([node for node in pred_graph 
                                          if node not in filtered_nodes])    

    find_match_predictions(gt_graph,
                           filtered_graph,
                           outp,
                           extp)
