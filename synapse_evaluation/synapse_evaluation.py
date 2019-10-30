from construct_graph_from_synapse_data import \
    load_synapses_from_catmaid_json, \
    add_segmentation_labels, \
    construct_prediction_graph, remove_intraneuron_synapses, \
    remove_out_of_roi_synapses
import utility as util
from multiprocessing import Pool
import numpy as np
import networkx as nx
from daisy import Coordinate, Roi
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import json
import argparse
import os
from os import path
from itertools import combinations, product
from functools import partial
from multiprocessing import Pool
import math
import copy


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
    outp['config_name'] = config_name
    # outp['plot_dir'] = path.join(outp['output_path'], 'plots')
    # outp['plot_dir'] = path.join(outp['output_path'])
    outp['plot_dir'] = output_dir
    outp['match_dir'] = path.join(outp['output_path'], 'matches')
    if 'segmentation' not in inp:
        inp['segmentation'] = {'zarr_path': inp['zarr_path'],
                               'dataset': inp['segmentation_dataset']}
    for key in ['min_inference_value', 'filter_metric', 'max_distance']:
        if not isinstance(extp[key], list):
            value = extp[key]
            extp[key] = [value]
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
                model[key] = {}
                model[key]['zarr_path'] = inp['zarr_path']
                model[key]['dataset'] = inp[key+'_dataset']
    for model_name, model_data in inp['models'].items():
        for key in ['postsynaptic', 'vector']:
            if key not in model_data:
                model_data[key] = {}
            if 'zarr_path' not in model_data[key]:
                try:
                    model_data[key]['zarr_path'] = model_data['zarr_path']
                except KeyError:
                    model_data[key]['zarr_path'] = inp['zarr_path']
            if 'dataset' not in model_data[key]:
                try:
                    model_data[key]['dataset'] = model_data[key+'_dataset']
                except KeyError:
                    model_data[key]['dataset'] = inp[key+'_dataset']
        inf_graph_json = '{}_syn_graph.json'
        if model_name:
            inf_graph_json = '{}_{}'.format(model_name, inf_graph_json)
        model_data['inf_graph_json'] = path.join(outp['output_path'], inf_graph_json)
    print("Config loaded from {}".format(user_path))
    return params


def plot_metric(graph, model_name, metric, output_path, num_hist_bins):
    min_inference_value = graph.graph['min_inference_value']
    if isinstance(metric, str):
        plot_title = '{}_{}_hist'.format(metric, min_inference_value)
        if model_name:
            plot_title = "{}_{}".format(model_name, plot_title)
        util.plot_node_attr_hist(graph, metric, output_path,
                                 plot_title, num_hist_bins)
        print("Plotted histogram of %s values" % metric)
    else:
        plot_title = '{}_{}_{}_scatter'.format(metric[0], metric[1],
                                               min_inference_value)
        if model_name:
            plot_title = "{}_{}".format(model_name, plot_title)
        util.plot_node_attr_scatter(graph, metric[0], metric[1],
                                    output_path, plot_title)
        print("Plotted scatterplot of %s as a function of %s"
               % (metric[1], metric[0]))


# This method returns all the nodes that do NOT exceed
# the minimum score thresholds specified by the filter
def apply_score_filter(postsyn_graph, filter_metric, percentile):
    print("Applying {}th percentile {} score filter".format(percentile, filter_metric))
    # threshold = np.percentile([score for node, score in 
    #                            postsyn_graph.nodes(data=filter_metric)],
    #                            percentile)
    threshold = percentile
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
    near_matches = []
    far_matches = []
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
        elif pred_postsyn['seg_label'] == gt_postsyn['seg_label']:
            # debug
            postsyn_dist = util.distance(pred_postsyn['zyx_coord'],
                                         gt_postsyn['zyx_coord'])
            if postsyn_dist <= max_dist:
                # matches[pred_syn] = presyn_dist + postsyn_dist
                # print("Near postsyn match:", pred_syn)
                near_matches.append(pred_syn)
            far_matches.append(pred_syn)

    if len(matches) == 0:
        print(gt_syn)
        (gt_presyn, gt_postsyn) = gt_syn
        print("GT postsyn xyz:", util.np_index_to_pixel_xyz(gt_postsyn['zyx_coord']))
        if len(near_matches):
            print("Near postsyn match:")
            for n in near_matches:
                print(n)
            print("Far postsyn match:")
            for n in far_matches:
                print(n)
                # exit(0)
    return matches


def write_matches_to_file(conn_dict,
                          filtered_graph,
                          model_name,
                          extraction,
                          percentile,
                          output_path,
                          voxel_size):
    basename = 'percentile_{}.txt'.format(percentile)
    if model_name:
        basename = "{}_{}".format(model_name, basename)
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
        false_neg_template = "\tCATMAID connector {} (neuroglancer: {}) (catmaid: {})"
        for gt_syn, syn_attr in false_neg.items():
            conn_id = syn_attr['conn_id']
            conn_xyz = syn_attr['conn_zyx_coord'][::-1]
            ng_coord = util.daisy_zyx_to_voxel_xyz(syn_attr['conn_zyx_coord'], voxel_size)
            print(false_neg_template.format(conn_id, ng_coord, conn_xyz), file=f)
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



def plot_false_pos_false_neg(error_counts, plot_title,
                             output_path, markers, colors):
    first_iteration = True
    fig, ax = plt.subplots(figsize=(12, 8))
    for (mod, percentiles), col in zip(error_counts.items(), colors):
        vals = [(perc, counts['false_neg'], counts['false_pos'])
                for perc, counts in percentiles.items()]
        vals.sort(key=lambda val: val[0])
        ax.plot([val[1] for val in vals],
                 [val[2] for val in vals],
                 label=mod, color=col, alpha=.5)
        for (perc, fn, fp), mark in zip(vals, markers):
            if first_iteration:
                ax.scatter(fn, fp, label=perc,
                            color=col, marker=mark,
                            alpha=.5)
            else:
                ax.scatter(fn, fp, color=col,
                            marker=mark, alpha=.5)
        first_iteration = False
    box = ax.get_position()
    plt.xlabel('False Negatives')
    plt.ylabel('False Positives')
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    file_name = path.join(output_path, plot_title)
    plt.savefig(file_name)
    plt.clf()
    print("Errors plotted: {}".format(file_name))



def load_model_prediction_graphs(models, inp,
                                 extp, output_path,
                                 num_processes):
    pred_graph_constructor = partial(construct_prediction_graph,
                                     segmentation_data=inp['segmentation'],
                                     extraction_config=extp,
                                     voxel_size=inp['voxel_size'])
    model_param_combos = [(mod, (min_inf, mod_data))
                          for (mod, mod_data) in models.items()
                          for min_inf in extp['min_inference_value']]
    p = Pool(num_processes)
    pred_graphs = p.starmap(pred_graph_constructor,
                            [combo[1] for combo in model_param_combos])
    pred_graph_dict = {mod:{} for mod in models}
    util.print_delimiter()
    for (mod, (min_inf, mod_data)), pred_graph \
            in zip(model_param_combos, pred_graphs):
        pred_graph_dict[mod][min_inf] = pred_graph
        util.syn_graph_to_json(pred_graph,
                               output_path,
                               mod)
    return pred_graph_dict

if __name__ == '__main__':
    params = parse_configs()
    inp = params['Input']
    outp = params['Output']
    extp = params['Extraction']

    for out_dir in [outp['output_path'], outp['plot_dir'], outp['match_dir']]:
        try:
            os.makedirs(out_dir)
        except FileExistsError:
            pass
    output_path = outp['output_path']
    print("Output path:", output_path)

    util.print_delimiter()
    gt_graph = load_synapses_from_catmaid_json(inp['skeleton'])
    gt_graph = add_segmentation_labels(gt_graph,
                                       **inp['segmentation'],
                                       materialize=extp['materialize'])

    if extp['remove_intraneuron_synapses']:
        gt_graph = remove_intraneuron_synapses(gt_graph)

    pred_graph_dict = load_model_prediction_graphs(inp['models'], inp,
                                                   extp, output_path,
                                                   8)

    output_percentile_metrics = False
    if output_percentile_metrics:
        for model_name, min_inf_vals in pred_graph_dict.items():
            for min_inf, pred_graph in min_inf_vals.items():
                sub_dir = path.join(outp['plot_dir'], "min_inf_{}".format(min_inf))
                try:
                    os.makedirs(sub_dir)
                except FileExistsError:
                    pass
                util.print_delimiter()
                for metric in outp['metric_plots']:
                    plot_metric(util.postsyn_subgraph(pred_graph),
                                model_name, metric,
                                sub_dir,
                                outp['num_hist_bins'])

    for min_inf, fil_met, max_dist in product(extp['min_inference_value'],
                                              extp['filter_metric'],
                                              extp['max_distance']):
        error_counts = {}
        param_combo = 'inf{}_dist{}_{}'.format(min_inf, max_dist, fil_met)
        for model_name in inp['models']:
            error_counts[model_name] = {perc: {} for perc in extp['percentiles']}
            pred_graph = pred_graph_dict[model_name][min_inf]
            # print(pred_graph.graph)
            roi = Roi(pred_graph.graph['roi_offset'],
                            pred_graph.graph['roi_shape'])
            gt_graph_loc = copy.deepcopy(gt_graph)
            gt_graph_loc = remove_out_of_roi_synapses(gt_graph_loc, roi)
            for percentile in extp['percentiles']:
                util.print_delimiter()
                filtered_nodes = apply_score_filter(util.postsyn_subgraph(pred_graph),
                                                    fil_met,
                                                    percentile)
                filtered_graph = pred_graph.subgraph([node for node in pred_graph 
                                                      if node not in filtered_nodes])    
                get_matches = partial(get_nearby_matches,
                                      pred_graph=filtered_graph,
                                      max_dist=max_dist)
                for presyn, postsyn in gt_graph_loc.edges():
                    gt_graph_loc[presyn][postsyn]['matches'] = \
                            get_matches((gt_graph_loc.nodes[presyn],
                                         gt_graph_loc.nodes[postsyn]))
                num_true_pos = len([(pre, post) for pre, post, match in
                                    gt_graph_loc.edges(data='matches') if len(match)])
                num_predicted = filtered_graph.number_of_edges()
                num_actual = gt_graph_loc.number_of_edges()
                error_counts[model_name][percentile]['true_pos'] = num_true_pos
                error_counts[model_name][percentile]['false_pos'] = num_predicted - num_true_pos
                error_counts[model_name][percentile]['false_neg'] = num_actual - num_true_pos
                util.print_delimiter()
                submodel = 'inf{}_dist{}_{}'.format(min_inf, max_dist, fil_met)
                if model_name:
                    submodel = "{}_{}".format(model_name, submodel)
                sub_dir = path.join(outp['match_dir'], param_combo)
                try:
                    os.makedirs(sub_dir)
                except FileExistsError:
                    pass
                write_matches_to_file(util.neuron_pairs_dict(gt_graph_loc),
                                      filtered_graph, submodel,
                                      {'min_inference_value': min_inf,
                                       'filter_metric': fil_met,
                                       'max_distance': max_dist},
                                      percentile,
                                      sub_dir, inp['voxel_size'])
        util.print_delimiter()
        plot_title = "{}_error_plot".format(param_combo)
        plot_title = outp['config_name'] + '_' + plot_title
        plot_false_pos_false_neg(error_counts, plot_title, outp['plot_dir'],
                                 outp['markers'], outp['colors'])
    print("Complete.")

