import pandas as pd
import networkx as nx
import numpy as np
import daisy
from daisy import Coordinate, Roi
#data = pd.read_csv('skeleton_coordinates.csv')
#data.columns = ['skeleton_id','treenode_id','parent_treenode_id','x','y','z','r']
##data = [['skeleton_id','treenode_id','parent_treenode_id','x','y','z','r']]


def add_nodes_from_catmaidCSV(CSVdata):
   skeleton_graph = nx.Graph()
   #print(data.columns.values)
   for i, nrow in CSVdata.iterrows():
       skeleton_graph.add_nodes_from([nrow['treenode_id']],skeleton_id = nrow['skeleton_id'])
       skeleton_graph.add_nodes_from([nrow['treenode_id']], x = nrow['x'])
       skeleton_graph.add_nodes_from([nrow['treenode_id']], y = nrow['y'])
       skeleton_graph.add_nodes_from([nrow['treenode_id']], z = nrow['z'])
       skeleton_graph.add_nodes_from([nrow['treenode_id']], parent_id = nrow['parent_treenode_id'])
       skeleton_graph.add_nodes_from([nrow['treenode_id']], segId_pred = -1)
       #print(nrow['treenode_id'])
       #print(nrow[['skeleton_id','x','y','z']].to_dict())
   return skeleton_graph
def add_edges_from_catamaidCSV(CSVdata,graph):
   for i, nrow in CSVdata.iterrows():
       if np.isnan(nrow['parent_treenode_id']):
           continue
       else:
           graph.add_edge(nrow['parent_treenode_id'],nrow['treenode_id'])
   return graph

def add_segId_from_prediction(graph,segmentation_path,threshold):
    segment_ds = daisy.open_ds(
    segmentation_path,
    threshold,
    mode='r+')
    for treenode_id, attr in graph.nodes(data=True):
        treenode_zyx = (attr['z'],attr['y'],attr['x'])
        if segment_ds.roi.contains(Coordinate(treenode_zyx)):
            seg_id = segment_ds[Coordinate(treenode_zyx)]
            #graph.add_nodes_from([treenode_id], seg_pred = seg_id)
            attr['segId_pred'] = seg_id
    
    return graph
