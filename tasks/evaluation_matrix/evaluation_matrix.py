
from statistics import mean
import math
from daisy import Coordinate
import networkx as nx 
from utility import shortest_euclidean_bw_two_sk

##1. number of splits or merges 
def num_splits_or_merges(dic):
    ##input should be dictionary embedded with dictionary {seg_id:{skeleton_id: counts}} or {skeleton_id:{seg_id:counts}}
    #find the number of split or merges 
    num_merge_split = 0
    for split_or_merge in dic.values():
        num_merge_split += len(split_or_merge)-1
    #print(num_split_or_merge)
    #print(num_split_or_merge)
    return num_merge_split

'''
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
def splits_error(graph):
    error_count = 0
    error_dict = {}
    sk_id = -1 
    seg_id = -1
    for treenode_id, attr in graph.nodes(data=True): 
        #print (attr)
        if attr['segId_pred'] == -1:
            continue
        elif attr['skeleton_id'] != sk_id:
            sk_id = attr['skeleton_id']
            error_dict[sk_id] = set()
        else:
            parent_node = graph.nodes[attr['parent_id']]
            if attr['segId_pred'] != parent_node['segId_pred']:
                error_count += 1
                error_dict[sk_id].add( (Coordinate((attr['z'],attr['y'],attr['x'])), Coordinate((parent_node['z'],parent_node['y'],parent_node['x']))) )
    return error_count, error_dict
def merge_error(graph):
    seg_dict = {}
    seg_error_dict = {}
    for treenode_id, attr in graph.nodes(data=True):
        if attr['segId_pred'] == -1:
            continue
        elif attr['segId_pred'] not in seg_dict:
            seg_dict[attr['segId_pred']] = {}
            seg_dict[attr['segId_pred']][attr['skeleton_id']] = set()
            seg_dict[attr['segId_pred']][attr['skeleton_id']].add((attr['z'],attr['y'],attr['x']))
        elif attr['skeleton_id'] not in seg_dict[attr['segId_pred']]:
            seg_dict[attr['segId_pred']][attr['skeleton_id']] = set()
            seg_dict[attr['segId_pred']][attr['skeleton_id']].add((attr['z'],attr['y'],attr['x']))
        else:
            seg_dict[attr['segId_pred']][attr['skeleton_id']].add((attr['z'],attr['y'],attr['x']))
    ##build the {seg_id:{sk_id:[((zyx),(zyx),...]}}
    error_counts = 0 
    for seg_id , seg_skeleton in seg_dict.items():
        seg_error_dict[seg_id] = set()
        sk_list = [sk_zyx for sk_zyx in seg_skeleton.values()]
        for pos1 in range(len(sk_list)):
            for pos2 in range(len(sk_list)):
                if pos1 < pos2:
                    seg_error_dict[seg_id].add(shortest_euclidean_bw_two_sk(sk_list[pos1],sk_list[pos2]))
                    error_counts +=1
    return error_counts, seg_error_dict
        

        



##2. purity = avg(max(Si) / N)   where Si: number of nodes with seg_id i  ; N: total number of nodes in skeleton  
def purity(dic):
    purity = []
    for sk_or_seg_dict in dic.values():
        total_count = 0
        max = 0
        for counts in sk_or_seg_dict.values():
            total_count += counts 
            if counts > max:
                max = counts
        purity.append(max/total_count)
    return mean(purity)




##3. Information Entropy H(N) = avg(-Σ P(Si|N) log2 P(Si|N)) 

def entropy(dic):
    for sk_or_seg_dict in  dic.values():
        
        math.log2(num)
        pass

##4. rand index
def rand_index():
    pass

if __name__ == "__main__":
    #quick testing
    dic = {123:{123,234},789:{123,456},334:{123,123,123}}
    #num_splits_or_merges(dic)
    purity()
