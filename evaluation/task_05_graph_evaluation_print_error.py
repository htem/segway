import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extract_segId_from_prediction import graph_with_segId_prediction
from evaluation_matrix import splits_error, merge_error, rand_voi_split_merge
from evaluation_matrix import get_rand_voi_gain_after_fix
import re
import daisy
from daisy import Coordinate
from multiprocessing import Pool
from functools import partial
import numpy as np
import os
import csv
matplotlib.use('Agg')

# TODO
# 2) Reduce the number of parameters using config dics for most methods
# 4) Give methods and variables more descriptive names
# compare with lines with any model and cutouts


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
        p = Pool(num_processes)
        graph_list = p.map(partial(graph_with_segId_prediction,
                                   skeleton_path=configs["skeleton"]["skeleton_path"],
                                   segmentation_path=seg_path,
                                   with_interpolation=configs["skeleton"]["with_interpolation"],
                                   step=configs["skeleton"]["step"],
                                   ignore_glia=configs["skeleton"]["ignore_glia"],
                                   leaf_node_removal_depth=configs["skeleton"]["leaf_node_removal_depth"]),
                           ['volumes/'+threshold
                            for threshold in agglomeration_thresholds])
        for graph in graph_list:
            if graph is None:
                numb_split.append(np.nan)
                numb_merge.append(np.nan)
                rand_split_list.append(np.nan)
                rand_merge_list.append(np.nan)
                voi_split_list.append(np.nan)
                voi_merge_list.append(np.nan)
            else:
                split_error_num, split_error_dict = splits_error(graph)
                merge_error_num, merge_error_dict = merge_error(graph)
                if configs['output']['plot_graph']:
                    numb_split.append(split_error_num)
                    numb_merge.append(int(merge_error_num))
                    (rand_split, rand_merge,
                     voi_split, voi_merge) = rand_voi_split_merge(graph)
                    rand_split_list.append(rand_split)
                    rand_merge_list.append(rand_merge)
                    voi_split_list.append(voi_split)
                    voi_merge_list.append(voi_merge)
                if configs['output']['write_CSV'] or configs['output']['write_TXT']:
                    index = graph_list.index(graph)
                    seg_vol = agglomeration_thresholds[index]
                    output_path = configs["output"]["output_path"]
                    if not output_path.endswith("/"):
                        output_path += "/"
                    output_path += configs["output"]["config_JSON"] + "_error_coords/"
                    voxel_size = configs["output"]["voxel_size"]
                    write_TXT = configs['output']['write_TXT']
                    write_CSV = configs['output']['write_CSV']
                    IDs_to_print = configs['output']['skeleton_IDs_to_print']
                    generate_error_coordinates_file(output_path, merge_error_dict, split_error_dict, seg_path, seg_vol, 
                                                    graph, voxel_size, write_TXT, write_CSV, IDs_to_print)
        model = get_model_name(seg_path, model_name_mapping)
        split_and_merge.extend((model, numb_merge, numb_split))
        split_and_merge_rand.extend((model, rand_merge_list, rand_split_list))
        split_and_merge_voi.extend((model, voi_merge_list, voi_split_list))
    
    config_file_name = configs["output"]["config_JSON"]
    output_path = configs["output"]["output_path"]
    markers = configs["output"]["markers"]
    colors = configs["output"]["colors"]
    generate_error_plot(agglomeration_thresholds, config_file_name, 'number', output_path, markers,
                      colors, *split_and_merge)
    generate_error_plot(agglomeration_thresholds, config_file_name, 'rand', output_path, markers,
                      colors, *split_and_merge_rand)
    generate_error_plot(agglomeration_thresholds, config_file_name, 'voi', output_path, markers,
                      colors, *split_and_merge_voi)


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
                           label=l.replace("segmentation_", ""),
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
    output_file_name = output_path+"/"+config_file_name+'_'+chosen_matrice 
    plt.savefig(output_file_name, dpi=300)


def generate_error_coordinates_file(output_path, merge_error_dict, split_error_dict, seg_path, seg_vol,
                                    graph, voxel_size, write_TXT, write_CSV, IDs_to_print):
    print('output folder for error csv files: %s' % output_path)
    try:
        os.makedirs(output_path)
    except FileExistsError:
        pass
    input_info = seg_path.split("/")
    file_name = output_path + "error_coords_" + input_info[-4] + "_" + input_info[-3] + "_" + input_info[-2] + "_" + seg_vol

    fields = ["error type", "ID (segment if merge, skeleton if split)", "coordinate 1", "coordinate 2", "node 1", "node 2"]
    merge_errors = format_merge_error_rows(merge_error_dict, seg_vol, graph, voxel_size)
    split_errors = format_split_error_rows(split_error_dict, seg_path, "volumes/" + seg_vol, graph, voxel_size)
    
    for skeleton_id in IDs_to_print:
        print("Skeleton", skeleton_id, "split errors", "(threshold = " + seg_vol + ")")
        errors = [error for error in split_errors if error[1] == skeleton_id]
        for error in errors:
            print(error[2], error[3], error[4], error[5])
        if len(errors) == 0:
            print("None")
    if write_TXT:
        with open(file_name + ".txt", "w") as f:
            print(seg_vol, file = f)
            print("MERGE ERRORS ", len(merge_errors), file = f)
            for error in merge_errors:
                print("Segment: %s" % error[1], file = f)
                print("\t%s merged to %s" % (error[2], error[3]), file = f)
                print("\tCATMAID nodes %s and %d" % (error[4], error[5]), file = f)
                print("", file = f)            
            print("SPLIT ERRORS ", len(split_errors), file = f)
            for error in split_errors:
                print("Skeleton: %s" % error[1], file = f)
                print("\t%s (%s)" % (error[2], error[3]), file = f)
                print("\tCATMAID nodes %s and %d" % (error[4], error[5]), file = f)
                print("\tSegs %s %s" % (graph.nodes[error[4]]['segId_pred'], graph.nodes[error[5]]['segId_pred']), file=f)
                print("", file = f)                    
    if write_CSV:
        with open(file_name + ".csv", "w") as f:
            csvwriter = csv.writer(f)
            csvwriter.writerow(fields)
            csvwriter.writerows(merge_errors)
            csvwriter.writerows(split_errors)


def format_merge_error_rows(merge_error_dict, seg_vol, graph, voxel_size):
    all_errors = []
    for seg_id in merge_error_dict:
        errors = merge_error_dict[seg_id]
        if len(errors):
            for error in errors:
                all_errors.append({
                    'segid': seg_id,
                    'xyz0': to_pixel_coord_xyz(error[0][0], voxel_size),
                    'xyz1': to_pixel_coord_xyz(error[0][1], voxel_size),
                    'node0': error[1],
                    'node1':error[2]
                    })
    all_errors = sorted(all_errors, key=lambda x: x['segid'])
    formatted = []
    for error in all_errors:
        row = ["m"]
        row.append(error['segid'])
        row.append(error['xyz0'])
        row.append(error['xyz1'])
        row.append(error['node0'])
        row.append(error['node1'])
        formatted.append(row)
    return formatted


def format_split_error_rows(split_error_dict, seg_path, seg_vol, graph, voxel_size):
    segment_ds = daisy.open_ds(seg_path, seg_vol)
    all_errors = []
    for skel_id in split_error_dict:
        errors = split_error_dict[skel_id]
        if len(errors):
            for error in errors:
                tree_node_id = error[2]
                parent_node_id = error[3]
                error = (error[0], error[1])
                xyzx = []
                for point in error:
                    xyzx.append((to_pixel_coord_xyz(point, voxel_size)))
                all_errors.append({
                    'skeleton': skel_id,
                    'xyzs': xyzx,
                    'tree_node_id': tree_node_id,
                    'parent_node_id': parent_node_id
                    })
    all_errors = sorted(all_errors, key=lambda x: x['skeleton'])
    formatted = []
    for error in all_errors:
        entry = {'skeleton': error["skeleton"], 'xyzs': error['xyzs']}
        row = ["s"]
        row.append(error['skeleton'])
        row.append(error['xyzs'][0])
        row.append(error['xyzs'][1])
        row.append(error['tree_node_id'])
        row.append(error['parent_node_id'])
        formatted.append(row)
    return formatted



def get_model_name(volume_path, name_dictionary={}):
    if volume_path in name_dictionary:
        return name_dictionary[volume_path]
    if re.search(r"setup[0-9]{2}", volume_path):
        model = re.search(r"setup[0-9]{2}",
                          volume_path).group(0) + \
                          "_"+re.search(r"[0-9]+00",
                                        volume_path).group(0)
    return model


# following code is to find the coordinate of split or merge error
def to_pixel_coord_xyz(zyx, voxel_size):
    zyx = (daisy.Coordinate(zyx) / daisy.Coordinate(voxel_size))
    return daisy.Coordinate((zyx[2], zyx[1], zyx[0]))

