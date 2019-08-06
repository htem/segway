import daisy
import time
from daisy import Coordinate
import pandas as pd
import json
from build_graph_from_catmaid import \
    add_nodes_from_catmaidCSV_with_interpolation, add_nodes_from_catmaidCSV,\
    add_nodes_from_catmaidJson_with_interpolation, add_nodes_from_catmaidJson
import os.path
import networkx as nx

def add_predicted_seg_labels(graph, segmentation_path, segment_dataset):
    print(segmentation_path)
    start_time = time.time()
    print('Adding segmentation predictions from %s' % segment_dataset)
    segment_array = daisy.open_ds(
        segmentation_path,
        segment_dataset)
    segment_array = segment_array[segment_array.roi]
    nodes = list(graph.nodes)
    for i, treenode_id in enumerate(nodes):
        attr = graph.nodes[treenode_id]
        attr['zyx_coord'] = (attr['z'], attr['y'], attr['x'])
        if segment_array.roi.contains(Coordinate(attr['zyx_coord'])):
            seg_id = segment_array[Coordinate(attr['zyx_coord'])]
            attr['seg_label'] = seg_id
        else:
            graph.remove_node(treenode_id)
        if i % 1500 == 0:
            print("(%s/%s) nodes labelled" % (i, len(nodes)))
    print('Task add_segId_from_prediction of %s took %s seconds' %
          (segment_dataset, round(time.time()-start_time, 3)))
    return assign_skeleton_indexes(graph)


def construct_graph_with_seg_labels(agglomeration_threshold, skeleton_path, segmentation_path,
                                    with_interpolation, step, leaf_node_removal_depth):
    if os.path.isdir(segmentation_path+'/'+agglomeration_threshold):
        if skeleton_path.endswith('.csv'):
            skeleton_data = pd.read_csv(skeleton_path)
            skeleton_data.columns = ['skeleton_id', 'treenode_id',
                                     'parent_treenode_id', 'x', 'y', 'z', 'r']
            if with_interpolation:
                skeleton_nodes = add_nodes_from_catmaidCSV_with_interpolation(skeleton_data,step)
            else:
                skeleton_nodes = add_nodes_from_catmaidCSV(skeleton_data)
        if skeleton_path.endswith('.json'):
            with open(skeleton_path, 'r') as f:
                skeleton_data = json.load(f)
            if with_interpolation:
                skeleton_nodes = add_nodes_from_catmaidJson_with_interpolation(skeleton_data,step)
            else:
                skeleton_nodes = add_nodes_from_catmaidJson(skeleton_data)
        edge_connect_nodes_to_parents(skeleton_nodes)
        remove_leaf_nodes(skeleton_nodes, leaf_node_removal_depth)
        return add_predicted_seg_labels(skeleton_nodes, segmentation_path, agglomeration_threshold)
    else:
        pass


# This method removes all the leaf nodes (those with 0 or 1 neighbors) 
# from the graph in order to avoid penalizing the model for small, unimportant 
# misclassifications at the ends of a cell.
def remove_leaf_nodes(graph, removal_depth):
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
# (the node specified by its 'parent_id' attribute) to facilitate
# removing the leaf nodes.
def edge_connect_nodes_to_parents(graph):
    for node in graph.nodes:
        parent = graph.nodes[node]['parent_id']
        if not parent is None:
            graph.add_edge(node, parent)
            # Line below might be unnecessary
            graph[node][parent]['error_type'] = ''
    return graph
    

# Assign unique ids to each cluster of connected nodes. This is to
# differentiate between sets of nodes that are discontinuous in the
# ROI but actually belong to the same skeleton ID, which is necessary
# because the network should not be penalized for incorrectly judging
# that these processes belong to different neurons.
def assign_skeleton_indexes(graph):
    skeleton_index_to_id = {}
    skel_clusters = nx.connected_components(graph)
    for i, cluster in enumerate(skel_clusters):
        for node in cluster:
            graph.nodes[node]['skeleton_index'] = i
        skeleton_index_to_id[i] = graph.nodes[cluster.pop()]['skeleton_id']
    return graph
