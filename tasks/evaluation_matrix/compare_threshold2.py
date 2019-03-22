from extract_segId_from_prediction import easy_process_block, merge_dict,split_dict,graph_with_segId_prediction
from evaluation_matrix import num_splits_or_merges,splits_error,merge_error

import pylab
import matplotlib.pyplot as plt
import os
import re
import daisy
from daisy import Coordinate

def to_pixel_coord_xyz(zyx):
    zyx = (daisy.Coordinate(zyx) / daisy.Coordinate((40, 4, 4)))
    return daisy.Coordinate((zyx[2], zyx[1], zyx[0]))

def compare_threshold_graph_with_dict():
    parent_path = '/home/yh231/segmentation/cb2_segmentation/outputs/2019_03/pl2_yumin/cb2_v2/100000/output.zarr'
    files = [f for f in os.listdir(parent_path+'/volumes') if re.match(r'segmentation.*',f)]
    files.sort()
    numb_merge = []
    numb_split = []
    for file in files:
        seg = easy_process_block(parent_path,'volumes/'+ file) # big dictionary
        numb_merge.append(num_splits_or_merges(merge_dict(*seg)))
        numb_split.append(num_splits_or_merges(split_dict(*seg)))
    #print(numb_merge)
    print(numb_split)

    plt.subplot(211)

    plt.plot(files,numb_merge,color = 'b')
    plt.title('number of merge error')
    plt.subplot(212)
    plt.plot(files,numb_split,color = 'r')
    plt.title('number of split error')
    plt.savefig('test.png')
def compare_threshold_graph():
    skeleton_path = '/n/groups/htem/temcagt/datasets/cb2/segmentation/python_scripts/yh231/cb2_cutout4.csv'
    segment_ds = daisy.open_ds(
            "/home/yh231/segmentation/cb2_segmentation/outputs/2019_03/cb2_synapse_cutout4/cb2/130000/output.zarr",
            "volumes/segmentation_0.500")
    #parent_path = '/home/yh231/segmentation/cb2_segmentation/outputs/2019_03/pl2_yumin/cb2_v2/100000/output.zarr'
    parent_path = '/home/yh231/segmentation/cb2_segmentation/outputs/2019_03/cb2_synapse_cutout4/cb2/130000/output.zarr'
    files = [f for f in os.listdir(parent_path+'/volumes') if re.match(r'segmentation.*',f)]

    #files = ['segmentation_0.500']
    files.sort()
    numb_split = []
    numb_merge = []
    split_error_dict = {}
    merge_error_dict = {}
    for file in files:
        graph = graph_with_segId_prediction(skeleton_path,parent_path,'volumes/'+ file) # graph 
        error_num, split_list = splits_error(graph)
        numb_split.append(error_num)
        split_error_dict[file]=split_list ##dict == {segmentation_threshold:{sk_id:(((zyx),(zyx)),....),...} }   
        merge_error_num, merge_list = merge_error(graph)
        numb_merge.append(merge_error_num)
        merge_error_dict[file]=merge_list ##dict == {segmentation_threshold:{seg_id:(((zyx),(zyx)),....),...} } 
    plt.subplot(211)
    plt.plot(list(map(lambda x: x.replace("segmentation_",""),files)),numb_merge,color = 'b')
    plt.ylabel('merge error')
    plt.subplot(212)
    plt.plot(list(map(lambda x: x.replace("segmentation_",""),files)),numb_split,color = 'r')
    plt.ylabel('split error')
    plt.savefig('cutouts4.png')
    # print (split_error_dict.keys())
    # for skel_id in split_error_dict['segmentation_0.500']:
    #     print("Skeleton: ", skel_id)
    #     errors = split_error_dict['segmentation_0.500'][skel_id]
    #     for error in errors:
    #         for point in error:
    #             #print(point)
    #             print(to_pixel_coord_xyz(point))
    #             print('segid is: %d'%segment_ds[Coordinate(point)])

    return merge_error_dict,split_error_dict

if __name__ == "__main__":
    compare_threshold_graph()
    #test_plot()