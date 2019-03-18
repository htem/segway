from extract_segDict_from_prediction import easy_process_block, merge_dict,split_dict
from evaluation_matrix import num_splits_or_merges
import matplotlib.pyplot as plt
import os
import re


def compare_threshold_graph():
    parent_path = '/home/yh231/segmentation/cb2_segmentation/outputs/2019_03/pl2_yumin/cb2_v2/100000/output.zarr'
    files = [f for f in os.listdir(parent_path+'/volumes') if re.match(r'segmentation.*',f)]
    files.sort()
    numb_merge = []
    numb_split = []
    for file in files:
        seg = easy_process_block(parent_path,'volumes/'+ file)
        numb_merge.append(num_splits_or_merges(merge_dict(*seg)))
        numb_split.append(num_splits_or_merges(split_dict(*seg)))
    plt.figure()
    plt.plot(files,numb_merge,color = 'b',label = 'number of merge error')
    plt.plot(files,numb_split,color = 'r',label = 'number of split error')
    plt.show()


if __name__ == "__main__":
    compare_threshold_graph()