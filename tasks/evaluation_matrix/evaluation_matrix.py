
from statistics import mean
import math
##input should be dictionary embedded with dictionary {seg_id:{skeleton_id: counts}} or {skeleton_id:{seg_id:counts}}
##1. number of splits or merges 
def num_splits_or_merges(dic):
    #find the number of split or merges 
    num_split_or_merge = 0
    for _, split_or_merge in dic.items():
        num_split_or_merge += len(split_or_merge)
    #print(num_split_or_merge)
    return num_split_or_merge

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
'''
def entropy(dic):
    for sk_or_seg_dict in  dic.values():
        total_count = 0
        math.log2(num)
        pass

##4. rand index
def rand_index():
    pass

if __name__ == "__main__":
    #quick testing
    #dic = {123:{123,234},789:{123,456},334:{123,123,123}}
    #num_splits_or_merges(dic)
'''