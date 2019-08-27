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


# This method extracts the parameter values specified in the json
# provided by the client. Unspecified parameters take on the values
# specified in segway/synapse_evaluation/synapse_task_defaults.json
def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument('user_path')
    args = parser.parse_args()
    
    with open(args.user_path, 'r') as f:
        user_configs = json.load(f)
    
    try:
        param_defaults_path = user_configs['Input']['parameter_defaults']
    except KeyError:
        params = {}
        script_dir = path.dirname(path.realpath(__file__))
        param_defaults_path = path.join(script_dir,
                                        'parameter_defaults.json')
    print("Loading parameter defaults from {}".format(param_defaults_path))
    with open(param_defaults_path, 'r') as f:
        params = json.load(f)
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
    inp = params['Input']
    outp = params['Output']
    extp = params['Extraction']
    config_name = path.splitext(path.basename(user_path))[0]
    inp['config_name'] = config_name
    if 'output_path' not in outp:
        output_dir = path.dirname(user_path)
        outp['output_path'] = path.join(output_dir,
                                        config_name+'_outputs')
    if 'inf_graph_json' not in inp:
        pred_graph_base = '{}_syn_graph.json'.format(extp['min_inference_value'])
        pred_graph_path = path.join(outp['output_path'], pred_graph_base)
        if path.exists(pred_graph_path):
            inp['inf_graph_json'] = pred_graph_path
    if 'segmentation' not in inp:
        inp['segmentation'] = {'zarr_path': inp['zarr_path'],
                               'dataset': inp['segmentation_dataset']}
    if 'models' not in inp:
        try:
            model_name = inp['model_name']
        except KeyError:
            model_name = ""
        inp['models'] = {model_name: {}}
        model = inp['models'][model_name]
        for key in ['postsynaptic', 'vector']:
            try:
                model[key] = inp[key]
            except KeyError:
                model[key]['zarr_path'] = inp['zarr_path']
                model[key]['dataset'] = inp[key+'_dataset']
    for model_name, model_data in inp['models'].items():
        for key in ['postsynaptic', 'vector']:
            if key not in model_data:
                model_data[key] = {}
                try:
                    model_data[key]['zarr_path'] = model_data['zarr_path']
                except KeyError:
                    model_data[key]['zarr_path'] = model_data['Input']['zarr_path']
                model_data[key]['dataset'] = model_data[key+'_dataset']
        inf_graph_json = '{}{}_syn_graph.json'.format(model_name, extp['min_inference_value'])
        model_data['inf_graph_json'] = path.join(outp['output_path'], inf_graph_json)
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


# This method returns all the nodes that do NOT exceed
# the minimum score thresholds specified by the filter
def apply_score_filter(postsyn_graph, filter_metric, percentile):
    print("Applying {}th percentile {} score filter".format(percentile, filter_metric))
    threshold = np.percentile([score for node, score in 
                               postsyn_graph.nodes(data=filter_metric)],
                               percentile)
    filtered_nodes = {node for node, score in
                      postsyn_graph.nodes(data=filter_metric)
                      if score <= threshold}
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
                          model_name,
                          extraction,
                          percentile,
                          output_path,
                          voxel_size):
    template = 'mininf{}_maxdist{}_{}_percentile{}_matches.txt'
    basename = template.format(extraction['min_inference_value'],
                               extraction['max_distance'],
                               extraction['filter_metric'],
                               percentile)
    if model_name:
        basename = model_name+'_'+basename
    output_file = path.join(output_path, basename)
    print("Writing matches file: {}".format(output_file))
    true_pos = {}
    false_neg = {}
    with open(output_file, "w") as f:
        print("Min inference value = {}".format(extraction['min_inference_value']), file=f)
        print("Applied filter {} {}".format(extraction['filter_metric'], percentile), file=f)
        print("Maximum distance of {}".format(extraction['max_distance']), file=f)
        pred_node_attr = filtered_graph.nodes(data=True)
        for cell_pair, gt_syns in conn_dict.items():
            print("Connections between {} and {}:".format(*cell_pair), file=f)
            for gt_syn, syn_attr in gt_syns.items():
                conn_id = syn_attr['conn_id']
                conn_xyz = syn_attr['conn_zyx_coord'][::-1]
                ng_coord = util.daisy_zyx_to_voxel_xyz(syn_attr['conn_zyx_coord'], voxel_size)
                print("\tCATMAID connector {} {}".format(conn_id, conn_xyz), file=f)
                print("\tNeuroglancer coordinate {}".format(ng_coord), file=f)
                dist_sorted_matches = sorted(syn_attr['matches'].items(),
                                             key=lambda match: match[1])
                if len(dist_sorted_matches):
                    true_pos[gt_syn] = dist_sorted_matches[0][0]
                else:
                    false_neg[gt_syn] = syn_attr
                    print("\t\tNone", file=f)
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
        
        print("FALSE NEGATIVES", file=f)
        for gt_syn, syn_attr in false_neg.items():
            conn_id = syn_attr['conn_id']
            conn_xyz = syn_attr['conn_zyx_coord'][::-1]
            ng_coord = util.daisy_zyx_to_voxel_xyz(syn_attr['conn_zyx_coord'], voxel_size)
            print("\tCATMAID connector {} (neuroglancer: {}) (catmaid: {})".format(conn_id, ng_coord, conn_xyz), file=f)
        print(file=f)
        print("FALSE POSITIVES", file=f)
        false_pos = [(postsyn, filtered_graph.nodes[postsyn])
                     for (presyn, postsyn) in filtered_graph.edges
                     if (presyn, postsyn) not in true_pos.values()]
        for presyn, attr in false_pos:
            ng_coord = util.daisy_zyx_to_voxel_xyz(attr['zyx_coord'], voxel_size)
            print("\tCoordinate {} area {} sum {} mean {} max {}".format(ng_coord,
                                                                         attr['area'],
                                                                         attr['sum'],
                                                                         attr['mean'],
                                                                         attr['max']), file=f)

                



def write_false_negatives_file(conn_dict,
                          filtered_graph,
                          model_name,
                          extraction,
                          percentile,
                          output_path,
                          voxel_size):
    template = 'mininf{}_maxdist{}_{}_percentile{}_matches_false_negatives.txt'
    basename = template.format(extraction['min_inference_value'],
                               extraction['max_distance'],
                               extraction['filter_metric'],
                               percentile)
    if model_name:
        basename = model_name+'_'+basename
    output_file = path.join(output_path, basename)
    print("Writing matches file: {}".format(output_file))
    with open(output_file, "w") as f:
        print("Min inference value = {}".format(extraction['min_inference_value']), file=f)
        print("Applied filter {} {}".format(extraction['filter_metric'], percentile), file=f)
        print("Maximum distance of {}".format(extraction['max_distance']), file=f)
        pred_node_attr = filtered_graph.nodes(data=True)
        for cell_pair, gt_syns in conn_dict.items():
            for gt_syn, syn_attr in gt_syns.items():
                if not len(syn_attr['matches']):
                    print("Connector {} between skeletons {} and {}".format(syn_attr['conn_id'],
                                                                            gt_syn[0],
                                                                            gt_syn[1]), file=f)
                    catmaid_coord = syn_attr['conn_zyx_coord'][::-1]
                    ng_coord = util.daisy_zyx_to_voxel_xyz(syn_attr['conn_zyx_coord'],
                                                           voxel_size)
                    print("\tCATMAID Coordinate {}".format(catmaid_coord), file=f)
                    print("\tNeuroglancer Coordinate {}".format(ng_coord), file=f)
                print(file=f)


def plot_false_pos_false_neg(error_counts, plot_title, output_path):
    for model, percentiles in error_counts.items():
        false_neg = []
        false_pos = []
        for percentile, counts in percentiles.items():
            false_neg.append(counts['false_neg'])
            false_pos.append(counts['false_pos'])
        plt.plot(false_neg, false_pos, '-o', label=model)
    plt.title(plot_title)
    plt.xlabel('false negatives')
    plt.ylabel('false postives')
    plt.legend()
    file_name = path.join(output_path, plot_title)
    plt.savefig(file_name)
    plt.clf()
    print("Errors plotted: {}".format(file_name))

if __name__ == '__main__':
    params = parse_configs()
    inp = params['Input']
    outp = params['Output']
    extp = params['Extraction']

    output_path = outp['output_path']
    try:
        os.makedirs(output_path)
    except FileExistsError:
        pass
    plot_dir = path.join(output_path, 'plots')
    try:
        os.makedirs(plot_dir)
    except FileExistsError:
        pass
    
    match_dir = path.join(output_path, 'matches')
    try:
        os.makedirs(match_dir)
    except FileExistsError:
        pass
    print("Output path:", output_path)

    util.print_delimiter()
    gt_graph = load_synapses_from_catmaid_json(inp['skeleton'])
    gt_graph = add_segmentation_labels(gt_graph,
                                       **inp['segmentation'])
    
    error_counts = {}
    for model, model_data in inp['models'].items():
        error_counts[model] = {}
        model_errors = error_counts[model]
        util.print_delimiter()
        try:
            pred_graph = util.json_to_syn_graph(model_data['inf_graph_json'])
        except FileNotFoundError:  
            pred_graph = construct_prediction_graph(model_data,
                                                    inp['segmentation'],
                                                    extp,
                                                    inp['voxel_size'])
            util.print_delimiter()
            util.syn_graph_to_json(pred_graph,
                                   output_path,
                                   model)
        util.print_delimiter()
        for metric in outp['metric_plots']:
            plot_metric(util.postsyn_subgraph(pred_graph),
                        metric,
                        plot_dir)
        
        for percentile in extp['percentiles']:
            util.print_delimiter()
            filtered_nodes = apply_score_filter(util.postsyn_subgraph(pred_graph),
                                                extp['filter_metric'],
                                                percentile)
            filtered_graph = pred_graph.subgraph([node for node in pred_graph 
                                                  if node not in filtered_nodes])    
            get_matches = partial(get_nearby_matches,
                                  pred_graph=filtered_graph,
                                  max_dist=extp['max_distance'])
            for presyn, postsyn in gt_graph.edges():
                gt_graph[presyn][postsyn]['matches'] = \
                        get_matches((gt_graph.nodes[presyn],
                                     gt_graph.nodes[postsyn]))
            model_errors[percentile] = {}
            num_true_pos = len([(pre, post) for pre, post, match in
                                gt_graph.edges(data='matches') if len(match)])
            model_errors[percentile]['true_pos'] = num_true_pos
            model_errors[percentile]['false_pos'] = filtered_graph.number_of_edges() - num_true_pos
            model_errors[percentile]['false_neg'] = gt_graph.number_of_edges() - num_true_pos
            write_matches_to_file(util.neuron_pairs_dict(gt_graph),
                                  filtered_graph, model,
                                  extp, percentile,
                                  match_dir, inp['voxel_size'])
    util.print_delimiter()
    plot_title = "mininf{}_maxdist{}_{}".format(extp['min_inference_value'],
                                                   extp['max_distance'],
                                                   extp['filter_metric'])
    plot_false_pos_false_neg(error_counts, plot_title, plot_dir)
    print("Complete.")

