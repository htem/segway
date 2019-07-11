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


def add_segId_from_prediction(graph, segmentation_path, segment_dataset, leaf_node_removal_depth):
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
    remove_leaf_nodes(graph, leaf_node_removal_depth)
    return graph


def graph_with_segId_prediction(threshold, skeleton_path, segmentation_path,
                                with_interpolation,step,ignore_glia,leaf_node_removal_depth):
    if os.path.isdir(segmentation_path+"/"+threshold):
        if skeleton_path.endswith('.csv'):
            skeleton_data = pd.read_csv(skeleton_path)
            skeleton_data.columns = ['skeleton_id', 'treenode_id',
                                     'parent_treenode_id', 'x', 'y', 'z', 'r']
            if with_interpolation:
                gNode = add_nodes_from_catmaidCSV_with_interpolation(skeleton_data,step,ignore_glia)
            else:
                gNode = add_nodes_from_catmaidCSV(skeleton_data,ignore_glia)
            # gNodeEdge = add_edges_from_catamaidCSV(skeleton_data,gNode)
        if skeleton_path.endswith('.json'):
            with open(skeleton_path, 'r') as f:
                skeleton_data = json.load(f)
            if with_interpolation:
                gNode = add_nodes_from_catmaidJson_with_interpolation(skeleton_data,step,ignore_glia)
            else:
                gNode = add_nodes_from_catmaidJson(skeleton_data,ignore_glia)
        return add_segId_from_prediction(gNode, segmentation_path, threshold, leaf_node_removal_depth)
    else:
        pass


def graph_with_segId_prediction2(
        segmentation_vol,
        skeleton_path,
        segmentation_path,
        with_interpolation,
        step,
        ignore_glia,
        leaf_node_removal_depth):

    print(skeleton_path)

    if os.path.isdir(segmentation_path+"/"+segmentation_vol):
        if skeleton_path.endswith('.csv'):
            skeleton_data = pd.read_csv(skeleton_path)
            skeleton_data.columns = ['skeleton_id', 'treenode_id',
                                     'parent_treenode_id', 'x', 'y', 'z', 'r']
            if with_interpolation:
                gNode = add_nodes_from_catmaidCSV_with_interpolation(skeleton_data,step,ignore_glia)
            else:      
                gNode = add_nodes_from_catmaidCSV(skeleton_data,ignore_glia)
        if skeleton_path.endswith('.json'):
            with open(skeleton_path, 'r') as f:
                skeleton_data = json.load(f)
            if with_interpolation:
                gNode = add_nodes_from_catmaidJson_with_interpolation(skeleton_data,step,ignore_glia)
            else:
                gNode = add_nodes_from_catmaidJson(skeleton_data,ignore_glia)
        print(gNode)
        return add_segId_from_prediction(gNode, segmentation_path,
                                         segmentation_vol, leaf_node_removal_depth)
    else:
        print("Path %s does not exist!" % (segmentation_path+"/" +
                                           segmentation_vol))
        assert(0)


# This method removes all the leaf nodes (those with 0 or 1 neighbors)
# from a networkx graph. Its aim is to trim the graph representing the catmaid
# skeleton to avoid penalizing the model for small, unimportant misclassifications at
# the ends of a cell.
def remove_leaf_nodes(graph, removal_depth):
    graph = connect_nodes_to_parents(graph)
    for i in range(removal_depth):
        leaf_nodes = []
        for node in graph.nodes:
            is_leaf = graph.degree[node] <= 1
            if is_leaf:
                leaf_nodes.append(node)
        for node in leaf_nodes:
            if graph.degree[node] == 1:     
                adjacent_node = list(graph.adj[node])[0]
                if graph.nodes[adjacent_node]['parent_id'] == node:
                    graph.nodes[adjacent_node]['parent_id'] = None
            graph.remove_node(node)
    return graph

# This method connects every node in the graph to its parent node
# (the node specified by its 'parent_id' attribute).
def connect_nodes_to_parents(graph):
    for node in graph.nodes:
        parent = graph.nodes[node]['parent_id']
        if not parent is None:
            graph.add_edge(node, parent)
    return graph