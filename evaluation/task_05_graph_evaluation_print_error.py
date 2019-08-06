import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extract_segId_from_prediction import construct_graph_with_seg_labels
from evaluation_matrix import find_merge_errors, find_split_errors, rand_voi_split_merge
import re
import daisy
from daisy import Coordinate
from multiprocessing import Pool
from functools import partial
import numpy as np
import os
import sys
import csv
from itertools import product
matplotlib.use('Agg')


def compare_segmentation_to_ground_truth_skeleton(
        agglomeration_thresholds,
        segmentation_paths,
        model_name_mapping,
        num_processes,
        configs):
    split_and_merge, split_and_merge_rand, split_and_merge_voi = [], [], []
    for seg_path in segmentation_paths:
        numb_split, numb_merge = [], []
        (rand_split_list,
         rand_merge_list,
         voi_split_list,
         voi_merge_list) = [], [], [], []
        graph_list = construct_graphs_in_parallel(agglomeration_thresholds, seg_path,
                                                num_processes, configs['skeleton'])
        for graph in graph_list:
            if graph is None:
                numb_split.append(np.nan)
                numb_merge.append(np.nan)
                rand_split_list.append(np.nan)
                rand_merge_list.append(np.nan)
                voi_split_list.append(np.nan)
                voi_merge_list.append(np.nan)
            else:
                z_weight_multiplier = configs['error_count']['z_weight_multiplier']
                ignore_glia = configs['error_count']['ignore_glia']
                assume_minimal_merges = configs['error_count']['assume_minimal_merges']
                max_break_size = configs['error_count']['max_break_size']
                split_errors, _= find_split_errors(graph, ignore_glia, max_break_size)
                merge_errors, _ = find_merge_errors(graph, z_weight_multiplier, ignore_glia, assume_minimal_merges)
                write_txt, write_csv = configs['output']['write_TXT'], configs['output']['write_CSV']
                if write_txt or write_csv:
                    seg_vol = agglomeration_thresholds[graph_list.index(graph)]
                    output_path, file_name = generate_output_path_and_file_name(configs, seg_vol, seg_path)
                    write_error_files(output_path, file_name, merge_errors, split_errors, graph,
                                        configs['output']['voxel_size'], write_txt, write_csv)
                (rand_split, rand_merge,
                voi_split, voi_merge) = rand_voi_split_merge(graph)
                numb_split.append(len(split_errors))
                numb_merge.append(len(merge_errors))
                rand_split_list.append(rand_split)
                rand_merge_list.append(rand_merge)
                voi_split_list.append(voi_split)
                voi_merge_list.append(voi_merge)
        model = get_model_name(seg_path, model_name_mapping)
        split_and_merge.extend((model, numb_merge, numb_split))
        split_and_merge_rand.extend((model, rand_merge_list, rand_split_list))
        split_and_merge_voi.extend((model, voi_merge_list, voi_split_list))
    output = configs['output']
    generate_error_plot(agglomeration_thresholds, output['config_JSON'], 'number', output['output_path'], 
                        output['markers'], output['colors'], *split_and_merge)
    generate_error_plot(agglomeration_thresholds, output['config_JSON'], 'rand', output['output_path'],
                        output['markers'], output['colors'], *split_and_merge_rand)
    generate_error_plot(agglomeration_thresholds, output['config_JSON'], 'voi', output['output_path'],
                        output['markers'], output['colors'], *split_and_merge_voi)


def construct_graphs_in_parallel(agglomeration_thresholds, segmentation_path,
                                num_processes, configs):
    p = Pool(num_processes)
    return p.map(partial(construct_graph_with_seg_labels,
                        skeleton_path=configs['skeleton_path'],
                        segmentation_path=segmentation_path,
                        with_interpolation=configs['with_interpolation'],
                        step=configs['step'],
                        leaf_node_removal_depth=configs['leaf_node_removal_depth']),
                        ['volumes/'+threshold for threshold in agglomeration_thresholds])


def generate_error_plot(
        agglomeration_thresholds,
        config_file_name,
        chosen_matrice,
        output_path,
        markers,
        colors,
        *split_and_merge):
    fig, ax = plt.subplots(figsize=(8, 6))
    for j in range(int(len(split_and_merge)/3)):
        ax.plot(split_and_merge[j*3+1], split_and_merge[j*3+2],
                label=split_and_merge[j*3], color=colors[j],
                zorder=1, alpha=0.5, linewidth=2.5)
        for a, b, m, l in zip(split_and_merge[j*3+1], split_and_merge[j*3+2],
                              markers, agglomeration_thresholds):
            if j == 0:
                ax.scatter(a, b, marker=m, c=colors[j],
                           label=l.replace('segmentation_', ''),
                           zorder=2, alpha=0.5, s=50)
            else:
                ax.scatter(a, b, marker=m, c=colors[j], zorder=2, alpha=0.5,
                           s=50)
    ax.legend()
    if chosen_matrice == 'number':
        ax.set_ylim(bottom=-0.8)
        ax.set_xlim(left=-0.8)
        plt.xlabel('Merge Error Count')
        plt.ylabel('Split Error Count')
    elif chosen_matrice == 'rand':
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=-0.01)
        plt.xlabel('Merge Rand')
        plt.ylabel('Split Rand')
    elif chosen_matrice == 'voi':
        ax.set_ylim(bottom=0)
        ax.set_xlim(left=-0.01)
        plt.xlabel('Merge VOI')
        plt.ylabel('Split VOI')
    output_file_name = output_path+'/'+config_file_name+'_'+chosen_matrice 
    plt.savefig(output_file_name, dpi=300)


def get_model_name(volume_path, name_dictionary={}):
    if volume_path in name_dictionary:
        return name_dictionary[volume_path]
    if re.search(r'setup[0-9]{2}', volume_path):
        model = re.search(r'setup[0-9]{2}',
                          volume_path).group(0) + \
                          '_'+re.search(r'[0-9]+00',
                                        volume_path).group(0)
    return model


def write_error_files(output_path, file_name, merge_errors, split_errors,
                    graph, voxel_size, write_txt, write_csv):
    try:
        os.makedirs(output_path)
    except FileExistsError:
        pass
    print("output path:", output_path)
    merge_error_rows, split_error_rows = format_errors(merge_errors, split_errors, graph, voxel_size)
    if write_txt:    
        with open(output_path + file_name + '.txt', 'w') as f:
            print("MERGE ERRORS (" + str(len(merge_errors)) + ")", file = f)
            for error in merge_error_rows:
                print("Segment %s" % error[1], file = f)
                print("\t%s and %s merged" % (error[2], error[3]), file = f)
                print("\tCATMAID nodes %s and %s" % (error[4], error[5]), file = f)
                print(file = f)
            print("SPLIT_ERRORS (" + str(len(split_errors)) + ")", file = f)
            for error in split_error_rows:
                print("Skeleton %s" % error[1], file = f)
                print("\t%s and %s split" % (error[2], error[3]), file = f)
                print("\tCATMAID nodes %s and %s" % (error[4], error[5]), file = f)
                print(file = f)
    if write_csv:
        with open(output_path + file_name + '.csv', 'w') as f:
            csvwriter = csv.writer(f)
            fields = ["error type", "ID (segment if merge, skeleton if split)", "coordinate 1", "coordinate 2", "node 1", "node 2"]
            csvwriter.writerow(fields)
            csvwriter.writerows(merge_error_rows)
            csvwriter.writerows(split_error_rows)


def format_errors(merge_errors, split_errors, graph, voxel_size):
    merge_error_rows, split_error_rows = [], []
    for error in merge_errors:
        node1, node2 = error[0], error[1]
        segment_id = str(graph.nodes[node1]['seg_label'])
        coord1 = str(to_pixel_coord_xyz(graph.nodes[node1]['zyx_coord'], voxel_size))
        coord2 = str(to_pixel_coord_xyz(graph.nodes[node2]['zyx_coord'], voxel_size))
        merge_error_rows.append(['M', segment_id, coord1, coord2, node1, node2])
    for error in split_errors:
        node1, node2 = error[0], error[1]
        skeleton_id = str(graph.nodes[node1]['skeleton_id'])
        coord1 = str(to_pixel_coord_xyz(graph.nodes[node1]['zyx_coord'], voxel_size))
        coord2 = str(to_pixel_coord_xyz(graph.nodes[node2]['zyx_coord'], voxel_size))
        split_error_rows.append(['S', skeleton_id, coord1, coord2, node1, node2])
    return merge_error_rows, split_error_rows


def generate_output_path_and_file_name(configs, seg_vol, seg_path):
    output_path = configs['output']['output_path']
    if not output_path.endswith('/'):
        output_path += '/'
    output_path += configs['output']['config_JSON'] + '_error_coords/'
    seg_info = seg_path.split('/')
    file_name = 'error_coords_' + seg_info[-4] + '_' + \
            seg_info[-3] + '_' + seg_info[-2] + '_' + seg_vol
    return output_path, file_name


# following code is to find the coordinate of split or merge error
def to_pixel_coord_xyz(zyx, voxel_size):
    zyx = (Coordinate(zyx) / Coordinate(voxel_size))
    return Coordinate((zyx[2], zyx[1], zyx[0]))
