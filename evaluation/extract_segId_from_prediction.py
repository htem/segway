import daisy
import time
from daisy import Coordinate
import pandas as pd
import json
from build_graph_from_catmaid import \
    add_nodes_from_catmaidCSV_with_interpolation, add_nodes_from_catmaidCSV,\
    add_nodes_from_catmaidJson_with_interpolation, add_nodes_from_catmaidJson
import os.path
########################


def add_segId_from_prediction(graph, segmentation_path, segment_dataset):
    print(segmentation_path)
    start_time = time.time()
    print("Start add_segId_from_prediction of %s" % segment_dataset)
    segment_array = daisy.open_ds(
        segmentation_path,
        segment_dataset)
    segment_array = segment_array[segment_array.roi]

    for treenode_id, attr in graph.nodes(data=True):
        treenode_zyx = (attr['z'], attr['y'], attr['x'])
        if segment_array.roi.contains(Coordinate(treenode_zyx)):
            seg_id = segment_array[Coordinate(treenode_zyx)]
            # graph.add_nodes_from([treenode_id], seg_pred = seg_id)
            attr['segId_pred'] = seg_id
    # print("Stop add_segId_from_prediction %s" % time.time())
    print("Task add_segId_from_prediction of %s took %s seconds" %
          (segment_dataset, time.time()-start_time))

    return graph


def graph_with_segId_prediction(threshold, skeleton_path, segmentation_path,
                                with_interpolation,step):
    if os.path.isdir(segmentation_path+"/"+threshold):
        if skeleton_path.endswith('.csv'):
            skeleton_data = pd.read_csv(skeleton_path)
            skeleton_data.columns = ['skeleton_id', 'treenode_id',
                                     'parent_treenode_id', 'x', 'y', 'z', 'r']
            if with_interpolation:
                gNode = add_nodes_from_catmaidCSV_with_interpolation(skeleton_data,step)
            else:
                gNode = add_nodes_from_catmaidCSV(skeleton_data)
            # gNodeEdge = add_edges_from_catamaidCSV(skeleton_data,gNode)
        if skeleton_path.endswith('.json'):
            with open(skeleton_path, 'r') as f:
                skeleton_data = json.load(f)
            if with_interpolation:
                gNode = add_nodes_from_catmaidJson_with_interpolation(skeleton_data,step)
            else:
                gNode = add_nodes_from_catmaidJson(skeleton_data)
        return add_segId_from_prediction(gNode, segmentation_path, threshold)
    else:
        pass


def graph_with_segId_prediction2(
        segmentation_vol,
        skeleton_path,
        segmentation_path,
        with_interpolation,
        step):

    print(skeleton_path)

    if os.path.isdir(segmentation_path+"/"+segmentation_vol):
        if skeleton_path.endswith('.csv'):
            skeleton_data = pd.read_csv(skeleton_path)
            skeleton_data.columns = ['skeleton_id', 'treenode_id',
                                     'parent_treenode_id', 'x', 'y', 'z', 'r']
            if with_interpolation:
                gNode = add_nodes_from_catmaidCSV_with_interpolation(skeleton_data,step)
            else:
      
                gNode = add_nodes_from_catmaidCSV(skeleton_data)
        if skeleton_path.endswith('.json'):
            with open(skeleton_path, 'r') as f:
                skeleton_data = json.load(f)
            if with_interpolation:
                gNode = add_nodes_from_catmaidJson_with_interpolation(skeleton_data,step)
            else:
                gNode = add_nodes_from_catmaidJson(skeleton_data)
        print(gNode)
        return add_segId_from_prediction(gNode, segmentation_path,
                                         segmentation_vol)
    else:
        print("Path %s does not exist!" % (segmentation_path+"/" +
                                           segmentation_vol))
        assert(0)
