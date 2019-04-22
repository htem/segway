# from statistics import mean
import math
from daisy import Coordinate
# import networkx as nx
from utility import shortest_euclidean_bw_two_sk

# 1. number of splits or merges errors and the coordiante of error

'''
##deprecated function
def num_splits_or_merges(dic):
    ##input should be dictionary embedded with dictionary {seg_id:{skeleton_id:
    # counts}} or {skeleton_id:{seg_id:counts}}
    #find the number of split or merges
    num_merge_split = 0
    for split_or_merge in dic.values():
        num_merge_split += len(split_or_merge)-1
    #print(num_split_or_merge)
    #print(num_split_or_merge)
    return num_merge_split

##deprecated function
def num_splits(graph):
    #input should be skeleton graph with attr segId
    sk_id = -1
    seg_id = -1
    sk_dict = {}
    for treenode_id, attr in graph.nodes(data=True):
        if attr['skeleton_id'] != sk_id:
            tree
            sk_id = attr['skeleton_id']
            seg_id = attr['segId_pred']
            seg_dict = {}
            seg_dict[seg_id] = treenode_id
            sk_dict[sk_id] = seg_dict
        else:
            if attr['segId_pred'] != seg_id :
                seg_list = []
                seg_list.append(treenode_id)
                seg_dict[seg_id] = seg_list
                seg_id = attr['segId_pred']
                seg_list = [treenode_id]
            else:
                seg_list.append(treenode_id)
'''


def splits_error(graph):  # dict === {sk_id:(((zyx),(zyx)),....),...}
    error_count = 0
    error_dict = {}
    sk_id = -1
    for treenode_id, attr in graph.nodes(data=True):
        # print (attr)
        if attr['skeleton_id'] != sk_id:
            sk_id = attr['skeleton_id']
            error_dict[sk_id] = set()
        if attr['segId_pred'] == -1:
            continue
        elif math.isnan(attr['parent_id']):
            continue
        else:
            parent_node = graph.node[attr['parent_id']]
            if parent_node['segId_pred'] == -1:
                continue
            if attr['segId_pred'] != parent_node['segId_pred']:
                error_count += 1
                error_dict[sk_id].add((Coordinate((attr['z'],
                                                   attr['y'],
                                                   attr['x'])),
                                       Coordinate((parent_node['z'],
                                                   parent_node['y'],
                                                   parent_node['x']))))
    return error_count, error_dict


def merge_error(graph):  # dict === {seg_id:([{(zyx),(zyx)},sk1,sk2],....),...}
    seg_dict = {}
    seg_error_dict = {}
    for treenode_id, attr in graph.nodes(data=True):
        if attr['segId_pred'] == -1:
            continue
        elif attr['segId_pred'] not in seg_dict:
            # collections.dafaultdict(dict) did the same thing
            seg_dict[attr['segId_pred']] = {}
            seg_dict[attr['segId_pred']][attr['skeleton_id']] = set()
            seg_dict[attr['segId_pred']][attr['skeleton_id']].add((attr['z'],
                                                                   attr['y'],
                                                                   attr['x']))
        elif attr['skeleton_id'] not in seg_dict[attr['segId_pred']]:
            seg_dict[attr['segId_pred']][attr['skeleton_id']] = set()
            seg_dict[attr['segId_pred']][attr['skeleton_id']].add((attr['z'],
                                                                   attr['y'],
                                                                   attr['x']))
        else:
            seg_dict[attr['segId_pred']][attr['skeleton_id']].add((attr['z'],
                                                                   attr['y'],
                                                                   attr['x']))
    # build the {seg_id:{sk_id:[((zyx),(zyx),...]}}
    error_counts = 0
    for seg_id, seg_skeleton in seg_dict.items():
        seg_error_dict[seg_id] = []
        sk_list = [sk_zyx for sk_zyx in seg_skeleton.values()]
        for pos1 in range(len(sk_list)):
            for pos2 in range(len(sk_list)):
                if pos1 < pos2:
                    seg_error_dict[seg_id].append([shortest_euclidean_bw_two_sk(sk_list[pos1], sk_list[pos2]), pos1, pos2])
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
    # print (sum_p_i)
    # print (sum_p_j)
    # print (sum_p_ij)
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
