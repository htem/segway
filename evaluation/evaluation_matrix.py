import math
from daisy import Coordinate
import networkx as nx
import copy
from utility import shortest_euclidean_bw_two_sk, to_pixel_coord_xyz

## here we consider the situation that network to sometimes miss small internal areas of larger processes without breakingtheir overall continuity. 
## include_breaking_error=True means we didn't consider such situation. include_breaking_error=False means we will consider such situation.  
## Thus the numb of splits error is decreasing and we will provide dict of breaking_errors will be ignored in counting the numb of split error.

# more information https://arxiv.org/pdf/1611.00421.pdf  see 4.3 and 5 
def splits_error(graph, include_breaking_error=False):  # dict === {sk_id_1:(((zyx),(zyx)),....),sk_id_2:(),...}
    error_count = 0
    error_dict = {}
    sk_id = -1
    breaking_error_dict = {}
    for treenode_id, attr in graph.nodes(data=True):
        if attr['skeleton_id'] != sk_id:
            sk_id = attr['skeleton_id']
            error_dict[sk_id] = set()
            breaking_error_dict[sk_id] = set()
        if attr['parent_id'] is None or math.isnan(attr['parent_id']):
            continue
        parent_node = graph.node[attr['parent_id']]
        if attr['segId_pred'] != parent_node['segId_pred']:
            if include_breaking_error:
                error_count += 1
                error_dict[sk_id].add((Coordinate(attr['zyx_coord']),
                                        Coordinate(parent_node['zyx_coord']),
                                        treenode_id, attr['parent_id']))
            else:
                ancestor_node = parent_node
                while not (ancestor_node['parent_id'] is None or math.isnan(ancestor_node['parent_id'])):
                    ancestor_node = graph.node[ancestor_node['parent_id']]
                    if ancestor_node['segId_pred'] == attr['segId_pred']:
                        breaking_error_dict[sk_id].add((Coordinate(attr['zyx_coord']),
                                                        Coordinate(parent_node['zyx_coord']),
                                                        Coordinate(ancestor_node['zyx_coord'])))
                        break
                if ancestor_node['segId_pred'] != attr['segId_pred']:
                    error_count += 1
                error_dict[sk_id].add((Coordinate(attr['zyx_coord']),
                                        Coordinate(parent_node['zyx_coord']),
                                        treenode_id, attr['parent_id']))
    if include_breaking_error:
        return error_count, (error_dict)
    else:
        return error_count, (error_dict, breaking_error_dict)
###breaking_error_dict stores three coordinate, first two is the same as error_dict, third one is the ancestor node save it from split error.
###for example, for node with segId A, if it's parent segId is B, it is split error. But if it's ancstor has same segId A, it's not split error.
### A....BA, BA is not split error any more. ....CBBA, here BA is split eror   
 

# A split error occurs when a pair of adjacent nodes receive different predicted segmentation IDs
# despite belong to the same neuron. 
def get_split_errors(graph):
    for edge in graph.edges:
        node1 = edge[0]
        node2 = edge[1]
        if graph.nodes[node1]['segId_pred'] != graph.nodes[node2]['segId_pred']:
            breaks = fomd_local_breaking_errors(edge, graph)
            if len(breaks):
                for break_edge in breaks:
                    graph.edges[break_edge]['error_type'] = "breaking"
            else:
                graph.edges[edge]['error_type'] = "split"
    split_edges = {edge for edge in graph.edges if graph.edges[edge]['error_type'] == "split"}
    breaking_edges = {edge for edge in graph.edges if graph.edges[edge]['error_type'] == "breaking"}
    return split_edges, breaking_edges


# A breaking error is a split error in which a mislabeled node or group of mislabeled nodes does not break the continuity of the skeleton it
# belongs to. For example, ...-X-X-X-Y-X-X would be an example of a breaking error of size 1, whereas ...-X-X-X-Y1-Y2-X-X would be an
# example of a breaking error of size 2. Note that Y1 and Y2 need not have the same label, so long as their label or labels differ from
# that of X. If Y1 and Y2 have different ids, Y1-Y2 will also be labelled a breaking error to prevent it from being counted as a split error.
# Breaking errors typically occur when a large organelle is labeled as background, preventing correct agglomeration at that location.
def find_local_breaking_errors(edge, graph, max_size = 1):
    if graph.edges[edge]['error_type'] == "breaking":
        return {edge}
    node1, node2 = edge[0], edge[1]
    for i in range(2):
        # Begins by constructing the subgraph constituting the max allowable break in the skeleton continuity
        subgraph = {node2}
        for i in range(max_size - 1):
            hits = set()
            for node in subgraph:
                for neighbor in graph.neighbors(node):
                    if graph.nodes[neighbor]['segId_pred'] != graph.nodes[node1]['segId_pred']:
                        hits.add(neighbor)
            subgraph = subgraph.union(hits)
        # Retrieves all the nodes at the periphery of the subgraph and checks that they all have the correct ID
        borders = set()
        for node in subgraph:
            for neighbor in graph.neighbors(node):
                if neighbor not in subgraph:
                    borders.add(neighbor)
        breaking_error = len(borders) and all(graph.nodes[node]['segId_pred'] ==
                                                graph.nodes[node1]['segId_pred'] for node in borders)

        # If the error is a breaking error, all the other edges at the periphery of the subgraph (and the adjacent nodes
        # with differing segmentation ID prediction within the subgraph) will be labelled breaking errors)
        if breaking_error == True:
            breaks = set()
            for node in borders:
                for neighbor in graph.neighbors(node):
                    if neighbor in subgraph:
                        breaks.add((node, neighbor))
            for node in subgraph:
                for neighbor in set(graph.neighbors(node)) - borders:
                    if graph.nodes[node]['segId_pred'] != graph.nodes[neighbor]['segId_pred']:
                        breaks.add((node, neighbor))
            return breaks
        else:
            # if a breaking error is not detected on the first pass, the process is repeated with node1 and node2 reversed. 
            # For example, in ...-X-X-X-Y-X-X-.., if Y was node1 and X node2, the breaking error would not be found until pass 2.
            node1, node2 = node2, node1
    return set()


def merge_error(graph,z_weight_multiplier=1):  # dict === {seg_id:([{(zyx),(zyx)},sk1,sk2],....),...}
    seg_dict = {}
    seg_error_dict = {}
    # build the {seg_id:{sk_id:[((zyx),(zyx),...]}}
    for treenode_id, attr in graph.nodes(data=True):
        if attr['segId_pred'] == -1:
            continue
        elif attr['segId_pred'] not in seg_dict:
            # collections.dafaultdict(dict) did the same thing
            seg_dict[attr['segId_pred']] = {}
            seg_dict[attr['segId_pred']][attr['skeleton_id']] = set()
            seg_dict[attr['segId_pred']][attr['skeleton_id']].add(attr['zyx_coord'])
        elif attr['skeleton_id'] not in seg_dict[attr['segId_pred']]:
            seg_dict[attr['segId_pred']][attr['skeleton_id']] = set()
            seg_dict[attr['segId_pred']][attr['skeleton_id']].add(attr['zyx_coord'])
        else:
            seg_dict[attr['segId_pred']][attr['skeleton_id']].add(attr['zyx_coord'])
    error_counts = 0
    for seg_id, seg_skeleton in seg_dict.items():
        seg_error_dict[seg_id] = []
        sk_id_list = [sk_id for sk_id in seg_skeleton.keys()] 
        for pos1 in range(len(sk_id_list)):
            for pos2 in range(len(sk_id_list)):
                if pos1 < pos2:
                    seg_error_dict[seg_id].append([shortest_euclidean_bw_two_sk(seg_skeleton[sk_id_list[pos1]], seg_skeleton[sk_id_list[pos2]], z_weight_multiplier), sk_id_list[pos1], sk_id_list[pos2]])
                    error_counts += 1
    return error_counts, seg_error_dict


# 2. rand and voi
# i : gt(skeleton) , j : prediction(segmentation)
# new rand = 1-rand, within the range 0-1, lower is better
# voi could be higher than 1, lower is better
# transform the script below to python code
# https://github.com/funkelab/funlib.evaluate/blob/master/funlib/evaluate/impl/rand_voi.hpp

def rand_voi_split_merge(graph, return_cluster_scores=False):
    p_ij = {}
    p_i = {}
    p_j = {}
    total = 0
    for treenode_id, attr in graph.nodes(data=True):
        if attr['segId_pred'] == -1:
            continue
        sk_id = attr['skeleton_id']
        seg_id = attr['segId_pred']
        total += 1
        if sk_id not in p_i:
            p_i[sk_id] = 1
        else:
            p_i[sk_id] += 1

        if seg_id not in p_j:
            p_j[seg_id] = 1
        else:
            p_j[seg_id] += 1

        if sk_id not in p_ij:
            p_ij[sk_id] = {}
            p_ij[sk_id][seg_id] = 1
        elif seg_id not in p_ij[sk_id]:
            p_ij[sk_id][seg_id] = 1
        else:
            p_ij[sk_id][seg_id] += 1
    # sum of squares in p_ij
    sum_p_ij = 0
    for i_dict in p_ij.values():
        for freq_label in i_dict.values():
            sum_p_ij += freq_label * freq_label
    # sum of squres in p_i
    sum_p_i = 0
    for freq_label in p_i.values():
        sum_p_i += freq_label * freq_label
    # sum of squres in p_j
    sum_p_j = 0
    for freq_label in p_j.values():
        sum_p_j += freq_label * freq_label
    # we have everything we need for RAND, normalize histograms for VOI
    for sk_id, i_dict in p_ij.items():
        for seg_id in i_dict:
            p_ij[sk_id][seg_id] /= total
    for sk_id in p_i:
        p_i[sk_id] /= total
    for seg_id in p_j:
        p_j[seg_id] /= total
    # compute entropies
    voi_split_i = {}
    voi_merge_j = {}
    if return_cluster_scores:
        for sk_id, prob in p_i.items():
            voi_split_i[sk_id] = prob * math.log2(prob)
        for seg_id, freq_label in p_j.items():
            voi_merge_j[seg_id] = prob * math.log2(prob)
    # H(a,b)
    H_ab = 0
    for sk_id, i_dict in p_ij.items():
        for seg_id, prob in i_dict.items():
            H_ab -= prob * math.log2(prob)
            if return_cluster_scores:
                voi_split_i[sk_id] -= prob * math.log2(prob)
                voi_merge_j[seg_id] -= prob * math.log2(prob)
    # H(a)
    H_a = 0
    for prob in p_i.values():
        H_a -= prob * math.log2(prob)
    # H(b)
    H_b = 0
    for prob in p_j.values():
        H_b -= prob * math.log2(prob)
    rand_split = 1-sum_p_ij/sum_p_i
    rand_merge = 1-sum_p_ij/sum_p_j
    # H(b|a)
    voi_split = H_ab - H_a
    # H(a|b)
    voi_merge = H_ab - H_b
    return rand_split, rand_merge, voi_split, voi_merge


def get_rand_voi_gain_after_fix(
        graph,
        error_type,
        error,
        origin_scores,
        segment_ds=None,
        seg_id=None):

    if error_type == "split":
        graph_fix = copy.deepcopy(graph)
        seg_id = segment_ds[Coordinate(error[0])]
        replace_seg_id = segment_ds[Coordinate(error[1])]
        for treenode_id, attr in graph_fix.nodes(data=True):
            if attr['segId_pred'] == replace_seg_id:
                attr['segId_pred'] = seg_id
        return get_diff(origin_scores, graph_fix)
    if error_type == "merge":
        graph_fix_0 = copy.deepcopy(graph)
        next_seg_id = max([attr['segId_pred'] for _, attr in graph_fix_0.nodes(data=True)])+1
        replace_sk_id_0 = error[1]
        for treenode_id, attr in graph_fix_0.nodes(data=True):
            if attr['segId_pred'] == seg_id:
                if attr['skeleton_id'] == replace_sk_id_0:
                    attr['segId_pred'] = next_seg_id
        return get_diff(origin_scores, graph_fix_0)


def get_diff(origin_scores, graph_fix):
    rand_split_fix, rand_merge_fix, voi_split_fix, voi_merge_fix = rand_voi_split_merge(graph_fix)
    return {
        'rand_split': abs(origin_scores[0] - rand_split_fix),
        'rand_merge': abs(origin_scores[1] - rand_merge_fix),
        'voi_split': abs(origin_scores[2] - voi_split_fix),
        'voi_merge': abs(origin_scores[3] - voi_merge_fix),
        }
