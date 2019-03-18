import pandas as pd
import networkx as nx
import numpy as np

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

