import networkx as nx
import numpy as np


def interpolation_sections_CSV(graph, CSVdata, current_row, id_to_start,
                               step):
    if np.isnan(current_row['parent_treenode_id']):
        graph = add_nodes(graph, current_row['z'], current_row['y'],
                          current_row['x'], current_row['treenode_id'],
                          current_row['parent_treenode_id'],
                          current_row['skeleton_id'])
        next_id_to_start = id_to_start
    else:
        parent_id = current_row['parent_treenode_id']
        parent_row = CSVdata[CSVdata['treenode_id'] == parent_id].squeeze()
        skeleton_id = current_row['skeleton_id']
        gap = abs(parent_row['z'] - current_row['z'])
        if gap <= step:
            graph = add_nodes(graph, current_row['z'], current_row['y'],
                              current_row['x'], current_row['treenode_id'],
                              current_row['parent_treenode_id'],
                              current_row['skeleton_id'])
            next_id_to_start = id_to_start
        else:
            graph = add_nodes(graph, current_row['z'], current_row['y'],
                              current_row['x'], current_row['treenode_id'],
                              id_to_start, current_row['skeleton_id'])
            gap_z = step*(parent_row['z'] - current_row['z'])/gap
            gap_y = step*(parent_row['y'] - current_row['y'])/gap
            gap_x = step*(parent_row['x'] - current_row['x'])/gap
            next_node_z = gap_z + current_row['z']
            next_node_y = gap_y + current_row['y']
            next_node_x = gap_x + current_row['x']
            next_sk_id = skeleton_id
            next_treenode_id = id_to_start
            next_parent_id = id_to_start + 1
            graph = add_nodes(graph, next_node_z, next_node_y, next_node_x,
                              next_treenode_id, next_parent_id, next_sk_id)
            while (next_node_z + gap_z)*gap_z < parent_row['z']*gap_z:
                next_node_z += gap_z
                next_node_y += gap_y
                next_node_x += gap_x
                next_treenode_id += 1
                next_parent_id += 1
                graph = add_nodes(graph, next_node_z, next_node_y, next_node_x,
                                  next_treenode_id, next_parent_id, next_sk_id)
            graph = add_nodes(graph, next_node_z+gap_z, next_node_y+gap_y,
                              next_node_x+gap_x, next_treenode_id+1, parent_id,
                              next_sk_id)
            next_id_to_start = next_treenode_id+2
    return graph, next_id_to_start


def interpolation_sections_JSON(graph, sk_id, tr_id, sk_dict, current_dict,
                                id_to_start, step):
    if current_dict['parent_id'] is None:
        graph = add_nodes(graph,
                          current_dict['location'][2],
                          current_dict['location'][1],
                          current_dict['location'][0],
                          int(tr_id),
                          current_dict['parent_id'],
                          int(sk_id))
        next_id_to_start = id_to_start
    else:
        parent_id = current_dict['parent_id']
        parent_dict = sk_dict[str(parent_id)]
        gap = abs(parent_dict['location'][2] - current_dict['location'][2])
        if gap <= step:
            graph = add_nodes(graph,
                              current_dict['location'][2],
                              current_dict['location'][1],
                              current_dict['location'][0],
                              int(tr_id),
                              current_dict['parent_id'],
                              int(sk_id))
            next_id_to_start = id_to_start
        else:
            graph = add_nodes(graph,
                              current_dict['location'][2],
                              current_dict['location'][1],
                              current_dict['location'][0],
                              int(tr_id),
                              id_to_start,
                              int(sk_id))
            gap_z = step*(parent_dict['location'][2] -
                          current_dict['location'][2])/gap
            gap_y = step*(parent_dict['location'][1] -
                          current_dict['location'][1])/gap
            gap_x = step*(parent_dict['location'][0] -
                          current_dict['location'][0])/gap
            next_node_z = gap_z + current_dict['location'][2]
            next_node_y = gap_y + current_dict['location'][1]
            next_node_x = gap_x + current_dict['location'][0]
            next_sk_id = int(sk_id)
            next_treenode_id = id_to_start
            next_parent_id = id_to_start + 1
            graph = add_nodes(graph, next_node_z, next_node_y, next_node_x,
                              next_treenode_id, next_parent_id, next_sk_id)
            while(next_node_z + gap_z)*gap_z < parent_dict['location'][2]*gap_z:
                next_node_z += gap_z
                next_node_y += gap_y
                next_node_x += gap_x
                next_treenode_id += 1
                next_parent_id += 1
                graph = add_nodes(graph, next_node_z, next_node_y, next_node_x,
                                  next_treenode_id, next_parent_id, next_sk_id)
            graph = add_nodes(graph, next_node_z+gap_z, next_node_y+gap_y,
                              next_node_x+gap_x, next_treenode_id+1, parent_id,
                              next_sk_id)
            next_id_to_start = next_treenode_id+2
    return graph, next_id_to_start


def add_nodes(graph, node_z, node_y, node_x, treenode_id, parent_treenode_id,
              sk_id):
    graph.add_nodes_from([treenode_id], skeleton_id=sk_id)
    graph.add_nodes_from([treenode_id], x=node_x)
    graph.add_nodes_from([treenode_id], y=node_y)
    graph.add_nodes_from([treenode_id], z=node_z)
    graph.add_nodes_from([treenode_id], parent_id=parent_treenode_id)
    graph.add_nodes_from([treenode_id], segId_pred=-1)
    return graph


def add_nodes_from_catmaidCSV_with_interpolation(CSVdata, step, ignore_glia = True):
    graph = nx.Graph()
    glia_cells_sk = glia_cells_sk_id_CSV(CSVdata)
    id_to_start = max(CSVdata['treenode_id'])+1
    for _, current_row in CSVdata.iterrows():
        if ignore_glia and current_row['skeleton_id'] in glia_cells_sk:
            continue
        else:
            graph, id_to_start = interpolation_sections_CSV(graph,
                                                            CSVdata,
                                                        current_row,
                                                        id_to_start,
							step)
    return graph


def add_nodes_from_catmaidJson_with_interpolation(JSONdata,step,ignore_glia = True):
    graph = nx.Graph()
    glia_cells_sk = glia_cells_sk_id_Json(JSONdata)

    # sanity check the provided input
    for skelid in JSONdata['skeletons']:
        skel = JSONdata['skeletons'][skelid]
        if len(skel['treenodes'].keys()) == 0:
            assert False, "Error in JSON file: skeleton %s has no treenodes" % skelid

    id_to_start = int(max(max(list(i['treenodes'].keys()))
                          for i in JSONdata['skeletons'].values()))+1
    for sk_id, sk_dict in JSONdata['skeletons'].items():
        if len(sk_dict['treenodes']) < 2:
            continue
        if ignore_glia and int(sk_id) in glia_cells_sk:
            continue
        for tr_id, tr_dict in sk_dict['treenodes'].items():
            (graph,
             id_to_start) = interpolation_sections_JSON(graph,
                                                        sk_id,
                                                        tr_id,
                                                        sk_dict['treenodes'],
                                                        tr_dict,
                                                        id_to_start,
							step)
    return graph


def add_nodes_from_catmaidCSV(CSVdata, ignore_glia=True):
    skeleton_graph = nx.Graph()
    # print(data.columns.values)
    glia_cells_sk = glia_cells_sk_id_CSV(CSVdata)
    for i, nrow in CSVdata.iterrows():
        if ignore_glia and current_row['skeleton_id'] in glia_cells_sk:
            continue
        else: 
            skeleton_graph.add_nodes_from([nrow['treenode_id']],
                                      skeleton_id=nrow['skeleton_id'])
            skeleton_graph.add_nodes_from([nrow['treenode_id']], x=nrow['x'])
            skeleton_graph.add_nodes_from([nrow['treenode_id']], y=nrow['y'])
            skeleton_graph.add_nodes_from([nrow['treenode_id']], z=nrow['z'])
            skeleton_graph.add_nodes_from([nrow['treenode_id']],
                                      parent_id=nrow['parent_treenode_id'])
            skeleton_graph.add_nodes_from([nrow['treenode_id']], segId_pred=-1)
            # print(nrow['treenode_id'])
            # print(nrow[['skeleton_id','x','y','z']].to_dict())
    return skeleton_graph


def add_nodes_from_catmaidJson(JSONdata,ignore_glia = True):
    skeleton_graph = nx.Graph()
    glia_cells_sk = glia_cells_sk_id_Json(JSONdata)
    for sk_id, sk_dict in JSONdata['skeletons'].items():
        if len(sk_dict['treenodes']) < 2:
            continue
        sk_id = int(sk_id)
        if ignore_glia and sk_id in glia_cells_sk:
            continue
        for tr_id, tr_dict in sk_dict['treenodes'].items():
            tr_id = int(tr_id)
            skeleton_graph.add_nodes_from([tr_id], skeleton_id=sk_id)
            skeleton_graph.add_nodes_from([tr_id], x=tr_dict['location'][0])
            skeleton_graph.add_nodes_from([tr_id], y=tr_dict['location'][1])
            skeleton_graph.add_nodes_from([tr_id], z=tr_dict['location'][2])
            skeleton_graph.add_nodes_from([tr_id],
                                          parent_id=tr_dict['parent_id'])
            skeleton_graph.add_nodes_from([tr_id], segId_pred=-1)
    return skeleton_graph

def glia_cells_sk_id_CSV(CSVdata):
    glia_cell_sk_id = set() #store the sk_id which are glia cells
    parent_dict = {} # store the parent_treenode_id and their child 
    sk_id = -1 
    for i, current_row in CSVdata.iterrows():
        if current_row['skeleton_id'] != sk_id: #find the end of this skeleton with sk_id
            for child_set in parent_dict.values():
                if len(child_set) > 5:
                    child_count = 0
                    for child in child_set: 
                        if child not in parent_dict:
                            child_count += 1
                    if child_count > 5:
                        glia_cell_sk_id.add(sk_id)
            sk_id = current_row['skeleton_id']
            parent_dict.clear()

        if np.isnan(current_row['parent_treenode_id']): 
            continue
        elif current_row['parent_treenode_id'] not in parent_dict:
            parent_dict[current_row['parent_treenode_id']] = set()
            parent_dict[current_row['parent_treenode_id']].add(current_row['treenode_id'])
        else:  
            parent_dict[current_row['parent_treenode_id']].add(current_row['treenode_id'])
    return glia_cell_sk_id
	
def glia_cells_sk_id_Json(JSONdata):
    glia_cell_sk_id = set() #store the sk_id which are glia cells
    parent_dict = {} # store the parent_treenode_id and their child 
    for sk_id, sk_dict in JSONdata['skeletons'].items():
        parent_dict.clear()
        for tr_id, tr_dict in sk_dict['treenodes'].items():
            if tr_dict['parent_id'] is None:
                continue
            elif tr_dict['parent_id'] not in parent_dict:
                parent_dict[tr_dict['parent_id']] = set()
                parent_dict[tr_dict['parent_id']].add(int(tr_id)) 
            else:
                parent_dict[tr_dict['parent_id']].add(int(tr_id))
        for child_set in parent_dict.values():
            if len(child_set) > 5:
                child_count = 0
                for child in child_set: 
                    if child not in parent_dict:
                        child_count += 1
                if child_count > 5:
                    glia_cell_sk_id.add(int(sk_id))  
    return glia_cell_sk_id 
	
	

# def add_edges_from_catamaidCSV(CSVdata, graph):
#     for i, nrow in CSVdata.iterrows():
#         if np.isnan(nrow['parent_treenode_id']):
#             continue
#         else:
#             graph.add_edge(nrow['parent_treenode_id'], nrow['treenode_id'])
#     return graph


# def add_segId_from_prediction(graph,segmentation_path,threshold):
#     print(segmentation_path)
#     print(threshold)
#     segment_ds = daisy.open_ds(
#     segmentation_path,
#     threshold)
#     for treenode_id, attr in graph.nodes(data=True):
#         treenode_zyx = (attr['z'],attr['y'],attr['x'])
#         if segment_ds.roi.contains(Coordinate(treenode_zyx)):
#             seg_id = segment_ds[Coordinate(treenode_zyx)]
#             #graph.add_nodes_from([treenode_id], seg_pred = seg_id)
#             attr['segId_pred'] = seg_id

#     return graph
