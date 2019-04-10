import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from extract_segId_from_prediction import graph_with_segId_prediction
from evaluation_matrix import splits_error,merge_error,rand_voi_split_merge
# import matplotlib.pyplot as plt
import os
import re
import json
import daisy
from daisy import Coordinate
from multiprocessing import Pool
from functools import partial
import numpy as np

def get_error_dict(skeleton_path,seg_path,threshold_list,num_process,with_interpolation):
    numb_split = []
    numb_merge = []
    split_error_dict = {}
    merge_error_dict = {}
    p = Pool(num_process)
    graph_list = p.map(partial(graph_with_segId_prediction,skeleton_path=skeleton_path,segmentation_path=seg_path,with_interpolation=with_interpolation),['volumes/'+threshold for threshold in threshold_list])
    #for threshold in threshold_list:
    for graph,threshold in zip(graph_list,threshold_list):
        graph = graph_with_segId_prediction('volumes/'+ threshold, skeleton_path, seg_path,with_interpolation) # graph 
        split_error_num, split_list = splits_error(graph)
        numb_split.append(split_error_num)
        split_error_dict[threshold]=split_list ##dict == {segmentation_threshold:{sk_id:(((zyx),(zyx)),....),...} }   
        merge_error_num, merge_list = merge_error(graph)
        numb_merge.append(int(merge_error_num))
        merge_error_dict[threshold]=merge_list ##dict == {segmentation_threshold:{seg_id:([{(zyx),(zyx)},sk1,sk2],....),...} } 
    return numb_split,numb_merge,split_error_dict,merge_error_dict

def get_rand_voi(skeleton_path,seg_path,threshold_list,num_process,with_interpolation):
    rand_split_list, rand_merge_list, voi_split_list, voi_merge_list = [], [], [], []
    p = Pool(num_process)
    graph_list = p.map(partial(graph_with_segId_prediction,skeleton_path=skeleton_path,segmentation_path=seg_path,with_interpolation=with_interpolation),['volumes/'+threshold for threshold in threshold_list])
    for graph in graph_list:
    #for file in threshold_list:
        #graph = graph_with_segId_prediction('volumes/'+ file, skeleton_path,seg_path)
        rand_split, rand_merge, voi_split, voi_merge = rand_voi_split_merge(graph)
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

    #split_and_merge should be modelname,merge,split
    fig, ax = plt.subplots(figsize=(8, 6))
    #print(len(split_and_merge))
    for j in range(int(len(split_and_merge)/3)): # zorder; to make points(markers) over the line
        ax.plot(split_and_merge[j*3+1],split_and_merge[j*3+2],label = split_and_merge[j*3],color = colors[j],zorder = 1,alpha=0.5,linewidth=2.5)
        for a,b,m,l in zip(split_and_merge[j*3+1],split_and_merge[j*3+2],markers,threshold_list):
            if j == 0:
                ax.scatter(a,b,marker=m,c=colors[j],label=l.replace("segmentation_",""),zorder = 2,alpha=0.5,s=50)
            else:
                ax.scatter(a,b,marker=m,c=colors[j],zorder=2,alpha=0.5,s=50)
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

    if re.search(r"setup[0-9]{2}",volume_path):
        model = re.search(r"setup[0-9]{2}",volume_path).group(0)+"_"+re.search(r"[0-9]+000",volume_path).group(0)
    else:
        #model = re.search(r"[0-9]+000",volume_path).group(0)
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
        markers = ['.',',','o','v','^','<','>','1','2'],
        colors=['b','g','r','c','m','y','k','coral','gold','purple']):

    markers = ['o','','^','','s','','p','','D']
    works = False 
    if 'number' in chosen_matrices:
        split_and_merge = []
        for seg_path in list_seg_path: 
            numb_split,numb_merge,_,_ = get_error_dict(skeleton_path,seg_path,threshold_list,num_process,with_interpolation)
            model = get_model_name(seg_path)
            split_and_merge.extend((model,numb_merge,numb_split))
        compare_threshold(threshold_list,filename,'number',output_path,markers,colors,*split_and_merge)
        works = True
    '''
    if 'rand' and 'voi' in chosen_matrices:
        split_and_merge_rand,split_and_merge_voi = [],[]
        for seg_path in list_seg_path:
            rand_split_list, rand_merge_list,voi_split_list, voi_merge_list = get_rand_voi(skeleton_path,seg_path,threshold_list,num_process,with_interpolation)
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
            rand_split_list, rand_merge_list,_,_ = get_rand_voi(skeleton_path,seg_path,threshold_list,num_process,with_interpolation)
            model = get_model_name(seg_path)
            split_and_merge_rand.extend((model,rand_merge_list,rand_split_list))
        compare_threshold(threshold_list,filename,'rand',output_path,markers,colors,*split_and_merge_rand)
        works = True
    if 'voi' in chosen_matrices:
        split_and_merge_voi = []
        for seg_path in list_seg_path:
            _,_, voi_split_list, voi_merge_list = get_rand_voi(skeleton_path,seg_path,threshold_list,num_process,with_interpolation)
            model = get_model_name(seg_path)
            split_and_merge_voi.extend((model,voi_merge_list,voi_split_list))            
        compare_threshold(threshold_list,filename,'voi',output_path,markers,colors,*split_and_merge_voi)
        works = True
    if not works:
        print("please provide the correct string for chosen matrices from 'number','rand' and 'voi'") 
#################quick compare: after interpolation, the graph is expensive to build, this function could save time and space but less parameter option provided


def quick_compare_with_graph(
        threshold_list,
        filename,
        skeleton_path,
        list_seg_path,
        model_name_mapping,
        num_process,
        output_path,
        with_interpolation,
        markers=['o', '', '^', '', 's', '', 'p', '', 'D', 'h'],
        colors=['b', 'g', 'r', 'c', 'm', 'y', 'k', 'coral', 'gold', 'purple']):

    split_and_merge,split_and_merge_rand,split_and_merge_voi = [],[],[]
    for seg_path in list_seg_path:
        numb_split, numb_merge = [],[]
        rand_split_list, rand_merge_list, voi_split_list, voi_merge_list = [], [], [], []
        p = Pool(num_process)
        graph_list = p.map(partial(graph_with_segId_prediction,skeleton_path=skeleton_path,segmentation_path=seg_path,with_interpolation=with_interpolation),['volumes/'+threshold for threshold in threshold_list])
        # for threshold in threshold_list:
        # graph = graph_with_segId_prediction(skeleton_path,seg_path,'volumes/'+ threshold)
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
                rand_split, rand_merge, voi_split, voi_merge = rand_voi_split_merge(graph)
                rand_split_list.append(rand_split)
                rand_merge_list.append(rand_merge)
                voi_split_list.append(voi_split)
                voi_merge_list.append(voi_merge)

        model = get_model_name(seg_path, model_name_mapping)
        split_and_merge.extend((model,numb_merge,numb_split))
        split_and_merge_rand.extend((model,rand_merge_list,rand_split_list))
        split_and_merge_voi.extend((model,voi_merge_list,voi_split_list))
    compare_threshold(threshold_list,filename,'number',output_path,markers,colors,*split_and_merge)
    compare_threshold(threshold_list,filename,'rand',output_path,markers,colors,*split_and_merge_rand)
    compare_threshold(threshold_list,filename,'voi',output_path,markers,colors,*split_and_merge_voi)    


################following code is to find the coordinate of split or merge error 
def to_pixel_coord_xyz(zyx):
    zyx = (daisy.Coordinate(zyx) / daisy.Coordinate((40, 4, 4)))
    return daisy.Coordinate((zyx[2], zyx[1], zyx[0]))

def print_the_split_error(split_error_dict,seg_path,threshold):
    segment_ds = daisy.open_ds(
             seg_path,
             "volumes/"+threshold)
    print (threshold)
    for skel_id in split_error_dict:
        print("Skeleton: ", skel_id)
        errors = split_error_dict[skel_id]
        for error in errors:
            for point in error:
                #print(point)
                print(to_pixel_coord_xyz(point))
                print('segid is: %d'%segment_ds[Coordinate(point)])

    #return merge_error_dict,split_error_dict



def print_the_merge_error(merge_error_dict,threshold):
    print (threshold)
    for seg_id in merge_error_dict:
        print("Segmentation:", seg_id)
        errors = merge_error_dict[seg_id]
        for error in errors:
            print (to_pixel_coord_xyz(error[0][0]))
            print ('sk_id is: %d'%error[1])
            print (to_pixel_coord_xyz(error[0][1]))
            print ('sk_id is: %d'%error[2])

def get_merge_split_error(skeleton_path,seg_path,threshold_list,error_type,num_process,with_interpolation):
    works = False
    #p = Pool(num_process)
    #graph_list = p.map(partial(graph_with_segId_prediction,skeleton_path=skeleton_path,segmentation_path=seg_path,with_interpolation=with_interpolation),['volumes/'+threshold for threshold in threshold_list])
    for threshold in threshold_list:
        graph = graph_with_segId_prediction('volumes/'+ threshold,skeleton_path,seg_path,with_interpolation) # graph
    #for graph in graph_list:
        if 'merge' in error_type:
            works = True
            _,merge_dict = merge_error(graph)
            print('following are merge error')
            print_the_merge_error(merge_dict,threshold)
        if 'split' in error_type:
            works = True
            _,split_dict = splits_error(graph)
            print('following are split error') 
            print_the_split_error(split_dict,seg_path,threshold)
    if not works:
        print("please provide the correct string for error type from 'merge' and 'split'") 

def get_multi_merge_split_error(skeleton_path,seg_path_list,threshold_list,error_type,num_process,with_interpolation):
    for seg_path in seg_path_list:
        get_merge_split_error(skeleton_path,seg_path,threshold_list,error_type,num_process,with_interpolation)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="plot the graph with evaluation matrices num_split_merge_error, rand, voi  || or  just print out the coordinates of split error and merge error")
    # parser.add_argument("configs",help="provide the configs with input information")
    # parser.add_argument("iteration", type=int,help="number of iterations you want to train until") 
    # parser.add_argument("-nc","--num_cores",help="Number of cores you want to use, defualts to 1",type=int,default=1)
    # parser.add_argument("-padz","--z_padding",type=int,default=0,
    #                     help="Amount to pad the z axis, helpful if you have a small stack, for cb3 I used 42 here because the input size was 84 pixels")
    # parser.add_argument("-gt","--groundtruth",action='append', help="the file with image data in /volumes/raw and segmentation in /volumes/labels/neuron_ids. Nonlabeled data can be masked in the /volumes/labels/unlabelled dataset")
    # args = parser.parse_args()

    threshold_list = ['segmentation_0.900']
    skeleton_path= '/n/groups/htem/temcagt/datasets/cb2/segmentation/python_scripts/yh231/cb2_cutout4.csv'
    seg_path = "/n/groups/htem/temcagt/datasets/cb2/segmentation/tri/cb2_segmentation/outputs/2019_03/cb2_synapse_cutout4/setup11/180000/output.zarr"
    #get_merge_split_error(skeleton_path,seg_path,threshold_list,['merge','split'])
    
    
    with open('file_to_evaluate.json','r') as f:
        config = json.load(f)
    threshold_list = config["threshold_list"]
    segment_volumes = config["segment_volumes"]
    chosen_matrices = config["chosen_matrices"]
    skeleton_path = config["OtherInput"]["skeleton_path"]
    filename = config["OtherInput"]["filename"]
    #compare_threshold_multi_model(threshold_list,filename,skeleton_path,chosen_matrices,segment_volumes)
    #quick_compare_with_graph(threshold_list,filename,skeleton_path,segment_volumes)