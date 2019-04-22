import daisy
import time
from daisy import Coordinate
import pandas as pd
from build_graph_from_catmaidCSV import \
    add_nodes_from_catmaidCSV_with_interpolation, add_nodes_from_catmaidCSV
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
                                with_interpolation):
    if os.path.isdir(segmentation_path+"/"+threshold):
        skeleton_data = pd.read_csv(skeleton_path)
        skeleton_data.columns = ['skeleton_id', 'treenode_id',
                                 'parent_treenode_id', 'x', 'y', 'z', 'r']
        if with_interpolation == "True":
            gNode = add_nodes_from_catmaidCSV_with_interpolation(skeleton_data)
        else:
            gNode = add_nodes_from_catmaidCSV(skeleton_data)
        # gNodeEdge = add_edges_from_catamaidCSV(skeleton_data,gNode)
        return add_segId_from_prediction(gNode, segmentation_path, threshold)
    else:
        pass


def graph_with_segId_prediction2(
        segmentation_vol,
        skeleton_path,
        segmentation_path,
        with_interpolation):

    print(skeleton_path)

    if os.path.isdir(segmentation_path+"/"+segmentation_vol):
        skeleton_data = pd.read_csv(skeleton_path)
        skeleton_data.columns = ['skeleton_id', 'treenode_id',
                                 'parent_treenode_id', 'x', 'y', 'z', 'r']
        if with_interpolation == "True":
            gNode = add_nodes_from_catmaidCSV_with_interpolation(skeleton_data)
        else:
            gNode = add_nodes_from_catmaidCSV(skeleton_data)
        print(gNode)
        return add_segId_from_prediction(gNode, segmentation_path,
                                         segmentation_vol)
    else:
        print("Path %s does not exist!" % (segmentation_path+"/" +
                                           segmentation_vol))
        assert(0)
