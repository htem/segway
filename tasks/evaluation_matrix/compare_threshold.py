from extract_segId_from_prediction import easy_process_block, merge_dict,split_dict,graph_with_segId_prediction
from evaluation_matrix import num_splits_or_merges,splits_error,merge_error

import pylab
import matplotlib.pyplot as plt
import os
import re


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
    parent_path = '/home/yh231/segmentation/cb2_segmentation/outputs/2019_03/pl2_yumin/cb2_v2/100000/output.zarr'
    files = [f for f in os.listdir(parent_path+'/volumes') if re.match(r'segmentation.*',f)]
    files.sort()
    numb_split = []
    numb_merge = []
    split_error_dict = {}
    merge_error_dict = {}
    for file in files:
        graph = graph_with_segId_prediction(parent_path,'volumes/'+ file) # graph 
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
    plt.savefig('test2.png')
    return merge_error_dict,split_error_dict

if __name__ == "__main__":
    compare_threshold_graph()
    #test_plot()