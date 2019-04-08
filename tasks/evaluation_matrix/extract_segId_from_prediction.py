import daisy
import time
from datetime import timedelta
from daisy import Coordinate, Roi
import pandas as pd
#from networkx import Graph
from build_graph_from_catmaidCSV import add_nodes_from_catmaidCSV_with_interpolation,add_nodes_from_catmaidCSV
#from utility import swap_rows_in_catmaidCSV
import os.path
########################
def add_segId_from_prediction(graph,segmentation_path,segment_dataset):
    print(segmentation_path)
    start_time = time.time()
    print("Start add_segId_from_prediction of %s" % segment_dataset)
    #print(segment_dataset)
    segment_array = daisy.open_ds(
        segmentation_path,
        segment_dataset)
    segment_array = segment_array[segment_array.roi]
    #segment_ndarray = segment_array.to_ndarray()
    #segment_array = daisy.Array(segment_ndarray, roi=segment_array.roi, voxel_size=segment_array.voxel_size)

    for treenode_id, attr in graph.nodes(data=True):
        treenode_zyx = (attr['z'],attr['y'],attr['x'])
        if segment_array.roi.contains(Coordinate(treenode_zyx)):
            seg_id = segment_array[Coordinate(treenode_zyx)]
            #graph.add_nodes_from([treenode_id], seg_pred = seg_id)
            attr['segId_pred'] = seg_id
    #print("Stop add_segId_from_prediction %s" % time.time())
    print("Task add_segId_from_prediction of %s took %s seconds" % (segment_dataset ,time.time()-start_time))

    return graph


def graph_with_segId_prediction(threshold,skeleton_path,segmentation_path,with_interpolation):
    if os.path.isdir(segmentation_path+"/"+threshold):
        #skeleton_data = pd.read_csv('/n/groups/htem/temcagt/datasets/cb2/segmentation/python_scripts/yh231/cb2_cutout4.csv') # skeleton dataset
        skeleton_data = pd.read_csv(skeleton_path)
        skeleton_data.columns = ['skeleton_id','treenode_id','parent_treenode_id','x','y','z','r']
        if with_interpolation == "True":
            gNode = add_nodes_from_catmaidCSV_with_interpolation(skeleton_data)
        else:
            gNode = add_nodes_from_catmaidCSV(skeleton_data)
        #gNodeEdge = add_edges_from_catamaidCSV(skeleton_data,gNode)
        return add_segId_from_prediction(gNode,segmentation_path,threshold)
    else:
        pass


##deprecated functioin
'''
def split_dict(segment_ds,skeleton_data,gNodeEdge):
    ##main idea is, for each skeleton, check how many segments it has, 

    seg_dict = {}
    #get the set of skeleton id
    skeleton_id = set(skeleton_data['skeleton_id'])
    #iterate through each nodes
    for sk_id in skeleton_id:
        seg_id_dict = {}
        for treenode_id, attr in gNodeEdge.nodes(data=True):
            if attr['skeleton_id'] == sk_id :
                treenode_zyx = (attr['z'],attr['y'],attr['x'])
                if segment_ds.roi.contains(Coordinate(treenode_zyx)):
                    seg_id = segment_ds[Coordinate(treenode_zyx)]
                    if seg_id not in seg_id_dict:
                        seg_id_dict[seg_id] = 1
                    else:
                        seg_id_dict[seg_id] += 1
        seg_dict[sk_id] = seg_id_dict
        # if (len(seg_id_array)>0):
        #     seg_dict[s_id] = 1/len(seg_id_array)
        #     print("the acurracy(based on number of splits) of skeleton %d is %f,seg_id is " %(s_id,1/len(seg_id_array)))
        #     print(seg_id_array) 
    return seg_dict

def merge_dict(segment_ds,skeleton_data,gNodeEdge):
    ##main idea is, for each seg_id, check how many skeleton has such seg_id
    seg_dict = {}
    #get the set of skeleton id
    skeleton_id = set(skeleton_data['skeleton_id'])
    #iterate through each nodes
    for sk_id in skeleton_id:
        for treenode_id, attr in gNodeEdge.nodes(data=True):
            if attr['skeleton_id'] == sk_id :
                treenode_zyx = (attr['z'],attr['y'],attr['x'])
                if segment_ds.roi.contains(Coordinate(treenode_zyx)):
                    seg_id = segment_ds[Coordinate(treenode_zyx)]
                    if seg_id not in seg_dict:
                        seg_dict[seg_id] = {}
                        seg_dict[seg_id][sk_id] = 1
                    else:
                        if sk_id not in seg_dict[seg_id]:
                            seg_dict[seg_id][sk_id] = 1
                        else:
                            seg_dict[seg_id][sk_id] += 1
    #print(seg_dict)
    return seg_dict
    
############


def show_networkx(skeleton_data,gNodeEdge):
    # sk_dict = {}
    # #get the set of skeleton id
    # skeleton_id = set(skeleton_data['skeleton_id'])
    # #iterate through each nodes
    # for sk_id in skeleton_id:
    #     sk_list = []
    #     for treenode_id, attr in gNodeEdge.nodes(data=True):
    #         if attr['skeleton_id'] == sk_id :
    #             sk_list.append(treenode_id)
    #     sk_dict[sk_id] = sk_list
    # print (sk_dict)
    sk_list= []
    for treenode_id, attr in gNodeEdge.nodes(data=True):
        sk_list.append(treenode_id)
    print(sk_list)
'''

if __name__ == "__main__":
    # running a quick test

    #logging.basicConfig(level=logging.INFO)

    data = easy_process_skeleton()
    #num_split(*data)
    show_networkx(*data)

