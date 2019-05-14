import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extract_segId_from_prediction import graph_with_segId_prediction
from extract_segId_from_prediction import graph_with_segId_prediction2
from evaluation_matrix import splits_error, merge_error, rand_voi_split_merge
# import matplotlib.pyplot as plt
import re
import daisy
from daisy import Coordinate
from multiprocessing import Pool
from functools import partial
import numpy as np
matplotlib.use('Agg')


def get_error_dict(skeleton_path, seg_path, threshold_list, num_process,
                   with_interpolation):
    numb_split = []
    numb_merge = []
    split_error_dict = {}
    merge_error_dict = {}
    p = Pool(num_process)
    graph_list = p.map(partial(graph_with_segId_prediction,
                               skeleton_path=skeleton_path,
                               segmentation_path=seg_path,
                               with_interpolation=with_interpolation),
                       ['volumes/'+threshold for threshold in threshold_list])
    # for threshold in threshold_list:
    for graph, threshold in zip(graph_list, threshold_list):
        graph = graph_with_segId_prediction('volumes/'+threshold,
                                            skeleton_path, seg_path,
                                            with_interpolation)
        split_error_num, split_list = splits_error(graph)
        numb_split.append(split_error_num)
        # dict == {segmentation_threshold:{sk_id:(((zyx),(zyx)),....),...} }
        split_error_dict[threshold] = split_list
        merge_error_num, merge_list = merge_error(graph)
        numb_merge.append(int(merge_error_num))
        # dict == {segmentation_threshold:{seg_id:([{(zyx),(zyx)},sk1,sk2])} }
        merge_error_dict[threshold] = merge_list
    return numb_split, numb_merge, split_error_dict, merge_error_dict


def get_rand_voi(skeleton_path, seg_path, threshold_list, num_process,
                 with_interpolation):
    rand_split_list, rand_merge_list = [], []
    voi_split_list, voi_merge_list = [], []
    p = Pool(num_process)
    graph_list = p.map(partial(graph_with_segId_prediction,
                               skeleton_path=skeleton_path,
                               segmentation_path=seg_path,
                               with_interpolation=with_interpolation),
                       ['volumes/'+threshold for threshold in threshold_list])
    for graph in graph_list:
        # for file in threshold_list:
        # graph = graph_with_segId_prediction('volumes/'+ file, skeleton_path,
        #                                     seg_path)
        (rand_split,
         rand_merge,
         voi_split,
         voi_merge) = rand_voi_split_merge(graph)
        rand_split_list.append(rand_split)
        rand_merge_list.append(rand_merge)
        voi_split_list.append(voi_split)
        voi_merge_list.append(voi_merge)
    return rand_split_list, rand_merge_list, voi_split_list, voi_merge_list


# compare with lines with any model and cutouts
def compare_threshold(
        threshold_list,
        filename,
        chosen_matrice,
        output_path,
        markers,
        colors,
        *split_and_merge):

    # split_and_merge should be modelname,merge,split
    fig, ax = plt.subplots(figsize=(8, 6))
    # print(len(split_and_merge))
    # zorder; to make points(markers) over the line
    for j in range(int(len(split_and_merge)/3)):
        ax.plot(split_and_merge[j*3+1], split_and_merge[j*3+2],
                label=split_and_merge[j*3], color=colors[j],
                zorder=1, alpha=0.5, linewidth=2.5)
        for a, b, m, l in zip(split_and_merge[j*3+1], split_and_merge[j*3+2],
                              markers, threshold_list):
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
    plt.savefig(output_path+"/"+filename+'_'+chosen_matrice, dpi=300)


def get_model_name(volume_path, name_dictionary={}):

    if volume_path in name_dictionary:
        return name_dictionary[volume_path]

    if re.search(r"setup[0-9]{2}", volume_path):
        model = re.search(r"setup[0-9]{2}",
                          volume_path).group(0) + \
                          "_"+re.search(r"[0-9]+000",
                                        volume_path).group(0)
    else:
        # model = re.search(r"[0-9]+000",volume_path).group(0)
        model = "cb2_130000"
    return model


def compare_threshold_multi_model(
        threshold_list,
        filename,
        skeleton_path,
        list_seg_path,
        chosen_matrices,
        num_process,
        output_path,
        with_interpolation,
        markers=['.', ',', 'o', 'v', '^', '<', '>', '1', '2'],
        colors=None):

    if colors is None:
        colors = [
            'b',
            'g',
            'r',
            'c',
            'm',
            'y',
            'k',
            'coral',
            'gold',
            'purple',
            'navy',
            'lime',
            'salmon',
            'saddlebrown']

    markers = ['o', '', '^', '', 's', '', 'p', '', 'D']
    works = False
    if 'number' in chosen_matrices:
        split_and_merge = []
        for seg_path in list_seg_path:
            numb_split, numb_merge, _, _ = get_error_dict(skeleton_path,
                                                          seg_path,
                                                          threshold_list,
                                                          num_process,
                                                          with_interpolation)
            model = get_model_name(seg_path)
            split_and_merge.extend((model, numb_merge, numb_split))
        compare_threshold(threshold_list, filename, 'number', output_path,
                          markers, colors, *split_and_merge)
        works = True
    '''
    if 'rand' and 'voi' in chosen_matrices:
        split_and_merge_rand,split_and_merge_voi = [],[]
        for seg_path in list_seg_path:
            (rand_split_list,
             rand_merge_list,
             voi_split_list,
             voi_merge_list) = get_rand_voi(skeleton_path,seg_path,
                                            threshold_list,num_process,
                                            with_interpolation)
            model = get_model_name(seg_path)
            split_and_merge_rand.extend((model,rand_merge_list,rand_split_list))
            split_and_merge_voi.extend((model,voi_merge_list,voi_split_list))
        compare_threshold(threshold_list,filename,'rand',markers,colors,*split_and_merge_rand)
        compare_threshold(threshold_list,filename,'voi',markers,colors,*split_and_merge_voi)
        works = True
    '''
    if 'rand' in chosen_matrices:
        split_and_merge_rand = []
        for seg_path in list_seg_path:
            (rand_split_list,
             rand_merge_list,
             _, _) = get_rand_voi(skeleton_path, seg_path, threshold_list,
                                  num_process, with_interpolation)
            model = get_model_name(seg_path)
            split_and_merge_rand.extend((model, rand_merge_list,
                                         rand_split_list))
        compare_threshold(threshold_list, filename, 'rand', output_path,
                          markers, colors, *split_and_merge_rand)
        works = True
    if 'voi' in chosen_matrices:
        split_and_merge_voi = []
        for seg_path in list_seg_path:
            (_, _,
             voi_split_list,
             voi_merge_list) = get_rand_voi(skeleton_path, seg_path,
                                            threshold_list, num_process,
                                            with_interpolation)
            model = get_model_name(seg_path)
            split_and_merge_voi.extend((model, voi_merge_list, voi_split_list))
        compare_threshold(threshold_list, filename, 'voi', output_path,
                          markers, colors, *split_and_merge_voi)
        works = True
    if not works:
        print("please provide the correct string for chosen matrices from \
              'number','rand' and 'voi'")
#################
# quick compare: after interpolation, the graph is expensive to build, this
# function could save time and space but less parameter option provided


def quick_compare_with_graph(
        threshold_list,
        filename,
        skeleton_path,
        list_seg_path,
        model_name_mapping,
        num_process,
        output_path,
        with_interpolation,
        markers=None,
        colors=None):

    if markers is None:
        markers = ['o', '', '^', '', 's', '', 'p', '', 'D', 'h']

    if colors is None:
        colors = [
            'b',
            'g',
            'r',
            'c',
            'm',
            'y',
            'k',
            'coral',
            'gold',
            'purple',
            'navy',
            'lime',
            'salmon',
            'saddlebrown']

    split_and_merge, split_and_merge_rand, split_and_merge_voi = [], [], []
    for seg_path in list_seg_path:
        numb_split, numb_merge = [], []
        (rand_split_list,
         rand_merge_list,
         voi_split_list,
         voi_merge_list) = [], [], [], []
        p = Pool(num_process)
        graph_list = p.map(partial(graph_with_segId_prediction,
                                   skeleton_path=skeleton_path,
                                   segmentation_path=seg_path,
                                   with_interpolation=with_interpolation),
                           ['volumes/'+threshold
                            for threshold in threshold_list])
        # for threshold in threshold_list:
        # graph = graph_with_segId_prediction(skeleton_path,seg_path,
        #                                     'volumes/'+ threshold)
        for graph in graph_list:
            if graph is None:
                numb_split.append(np.nan)
                numb_merge.append(np.nan)
                rand_split_list.append(np.nan)
                rand_merge_list.append(np.nan)
                voi_split_list.append(np.nan)
                voi_merge_list.append(np.nan)
            else:
                split_error_num, _ = splits_error(graph)
                numb_split.append(split_error_num)
                merge_error_num, _ = merge_error(graph)
                numb_merge.append(int(merge_error_num))
                (rand_split, rand_merge,
                 voi_split, voi_merge) = rand_voi_split_merge(graph)
                rand_split_list.append(rand_split)
                rand_merge_list.append(rand_merge)
                voi_split_list.append(voi_split)
                voi_merge_list.append(voi_merge)

        model = get_model_name(seg_path, model_name_mapping)
        split_and_merge.extend((model, numb_merge, numb_split))
        split_and_merge_rand.extend((model, rand_merge_list, rand_split_list))
        split_and_merge_voi.extend((model, voi_merge_list, voi_split_list))
    compare_threshold(threshold_list, filename, 'number', output_path, markers,
                      colors, *split_and_merge)
    compare_threshold(threshold_list, filename, 'rand', output_path, markers,
                      colors, *split_and_merge_rand)
    compare_threshold(threshold_list, filename, 'voi', output_path, markers,
                      colors, *split_and_merge_voi)


# following code is to find the coordinate of split or merge error
def to_pixel_coord_xyz(zyx):
    zyx = (daisy.Coordinate(zyx) / daisy.Coordinate((40, 4, 4)))
    return daisy.Coordinate((zyx[2], zyx[1], zyx[0]))


def print_split_errors(split_error_dict, seg_path, seg_vol):

    segment_ds = daisy.open_ds(seg_path, seg_vol)
    print(seg_vol)

    for skel_id in split_error_dict:
        errors = split_error_dict[skel_id]
        if len(errors):
            print("Skeleton: ", skel_id)
            for error in errors:
                for point in error:
                    print("\t%s (%s)" % (to_pixel_coord_xyz(point),
                                         segment_ds[Coordinate(point)]))
                    # print('segid is: %d'%segment_ds[Coordinate(point)])


def print_merge_errors(merge_error_dict, seg_vol):
    print(seg_vol)
    for seg_id in merge_error_dict:
        errors = merge_error_dict[seg_id]
        if len(errors):
            print("Segmentation:", seg_id)
            for error in errors:
                print("%s merged to %s" % (
                    to_pixel_coord_xyz(error[0][0]),
                    to_pixel_coord_xyz(error[0][1])))


def get_merge_split_error(
        skeleton_path,
        seg_path,
        seg_vol,
        error_type,
        num_process,
        with_interpolation):

    graph = graph_with_segId_prediction2(
        seg_vol,
        skeleton_path,
        seg_path,
        with_interpolation)
    print(graph)

    if "merge" in error_type or "both" in error_type:
        _, merge_dict = merge_error(graph)
        print('Merge errors:')
        print_merge_errors(merge_dict, seg_vol)

    if "split" in error_type or "both" in error_type:
        _, split_dict = splits_error(graph)
        print('Split errors:')
        print_split_errors(split_dict, seg_path, seg_vol)


def get_multi_merge_split_error(skeleton_path, seg_path_list, threshold_list,
                                error_type, num_process, with_interpolation):
    for seg_path in seg_path_list:
        get_merge_split_error(skeleton_path, seg_path, threshold_list,
                              error_type, num_process, with_interpolation)