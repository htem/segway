from extract_segId_from_prediction import graph_with_segId_prediction
from evaluation_matrix import splits_error,merge_error
import matplotlib.pyplot as plt
import os
import re
import daisy
from daisy import Coordinate



def to_pixel_coord_xyz(zyx):
    zyx = (daisy.Coordinate(zyx) / daisy.Coordinate((40, 4, 4)))
    return daisy.Coordinate((zyx[2], zyx[1], zyx[0]))


def get_error_dict(skeleton_path,seg_path,comparelist):
    #skeleton_path = '/n/groups/htem/temcagt/datasets/cb2/segmentation/python_scripts/yh231/cb2_cutout4.csv'
    # segment_ds = daisy.open_ds(
    #         "/home/yh231/segmentation/cb2_segmentation/outputs/2019_03/cb2_synapse_cutout4/cb2/130000/output.zarr",
    #         "volumes/segmentation_0.500")
    #parent_path = '/home/yh231/segmentation/cb2_segmentation/outputs/2019_03/pl2_yumin/cb2_v2/100000/output.zarr'
    # parent_path = '/home/yh231/segmentation/cb2_segmentation/outputs/2019_03/cb2_synapse_cutout4/cb2/130000/output.zarr'
    # files = [f for f in os.listdir(parent_path+'/volumes') if re.match(r'segmentation.*',f)]

    # #files = ['segmentation_0.500']
    # files.sort()
    numb_split = []
    numb_merge = []
    split_error_dict = {}
    merge_error_dict = {}
    for file in comparelist:
        graph = graph_with_segId_prediction(skeleton_path,seg_path,'volumes/'+ file) # graph 
        split_error_num, split_list = splits_error(graph)
        numb_split.append(split_error_num)
        split_error_dict[file]=split_list ##dict == {segmentation_threshold:{sk_id:(((zyx),(zyx)),....),...} }   
        merge_error_num, merge_list = merge_error(graph)
        numb_merge.append(merge_error_num)
        merge_error_dict[file]=merge_list ##dict == {segmentation_threshold:{seg_id:(((zyx),(zyx)),....),...} } 
    return numb_split,numb_merge,split_error_dict,merge_error_dict


def compare_with_graph(comparelist, numb_split,numb_merge):
    
    plt.subplot(211)
    plt.plot(list(map(lambda x: x.replace("segmentation_",""),comparelist)),numb_merge,color = 'b')
    plt.ylabel('merge error')
    plt.subplot(212)
    plt.plot(list(map(lambda x: x.replace("segmentation_",""),comparelist)),numb_split,color = 'r')
    plt.ylabel('split error')
    plt.savefig('cutouts5.png')

def print_the_split_error(split_error_dict,seg_path):
    segment_ds = daisy.open_ds(
             seg_path,
             "volumes/segmentation_0.300")
    print (split_error_dict.keys())
    for skel_id in split_error_dict['segmentation_0.300']:
        print("Skeleton: ", skel_id)
        errors = split_error_dict['segmentation_0.300'][skel_id]
        for error in errors:
            for point in error:
                #print(point)
                print(to_pixel_coord_xyz(point))
                print('segid is: %d'%segment_ds[Coordinate(point)])

    return merge_error_dict,split_error_dict

if __name__ == "__main__":
    skeleton_path = '/n/groups/htem/temcagt/datasets/cb2/segmentation/python_scripts/yh231/cb2_cutout5.csv'
    seg_path='/home/yh231/segmentation/cb2_segmentation/outputs/2019_03/cb2_synapse_cutout5/cb2/130000/output.zarr'
    comparelist = ['segmentation_0.200','segmentation_0.300','segmentation_0.400','segmentation_0.500','segmentation_0.600','segmentation_0.700']
    numb_split,numb_merge,split_error_dict,merge_error_dict = get_error_dict(skeleton_path,seg_path,comparelist)
    compare_with_graph(comparelist, numb_split,numb_merge)
    print_the_split_error(split_error_dict,seg_path)

    #test_plot()