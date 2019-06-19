import math
from daisy import Coordinate

## distance based on raw data (nm)
def shortest_euclidean_bw_two_sk_raw(set1, set2):
    shortest_len = math.inf
    shortest_datapoint = set()
    for point1 in set1:
        for point2 in set2:
            distance = math.sqrt(sum([(a-b)**2 for a, b in zip(point1,
                                                               point2)]))
            if distance < shortest_len:
                shortest_len = distance
                shortest_datapoint = (Coordinate(point1), Coordinate(point2))
    return shortest_datapoint

## distance based on pixel data (zyx)/(40,4,4) 
def shortest_euclidean_bw_two_sk_pixel(set1, set2):
    multiplier = (40,4,4)
    shortest_len = math.inf
    shortest_datapoint = set()
    for point1 in set1:
        for point2 in set2:
            distance = math.sqrt(sum([(a-b)**2 for a, b in zip(map(lambda c,d: c/d ,point1,multiplier),
                                                               map(lambda c,d: c/d ,point2,multiplier) )]))
            if distance < shortest_len:
                shortest_len = distance
                shortest_datapoint = (Coordinate(point1), Coordinate(point2))
    return shortest_datapoint

## distance after multipling z_weight
## higher z_weight, more penalty in distance. In other word, difference of z is prone to be small 
def shortest_euclidean_bw_two_sk(set1, set2,z_weight_multiplier):
    multiplier = (z_weight_multiplier,1,1)
    shortest_len = math.inf
    shortest_datapoint = set()
    for point1 in set1:
        for point2 in set2:
            distance = math.sqrt(sum([(a-b)**2 for a, b in zip(map(lambda c,d: c*d ,point1,multiplier),
                                                               map(lambda c,d: c*d ,point2,multiplier) )]))
            if distance < shortest_len:
                shortest_len = distance
                shortest_datapoint = (Coordinate(point1), Coordinate(point2))
    return shortest_datapoint

# following code is to find the coordinate of split or merge error
def to_pixel_coord_xyz(zyx):
    zyx = (Coordinate(zyx) / Coordinate((40, 4, 4)))
    return Coordinate((zyx[2], zyx[1], zyx[0]))


# deprecated: graph_with_segId_prediction() is now more robust, no restriction
# in csv files, among the treenode with the same skeleton_id, the treenode \
# with no parent should be above others
'''
def swap_rows_in_catmaidCSV(CSVdata):
    # CSVdata = pd.read_csv('/n/groups/htem/temcagt/datasets/cb2/segmentation/
    # python_scripts/yh231/cb2_cutout4.csv')
    # CSVdata.columns = ['skeleton_id','treenode_id','parent_treenode_id','x',
    # 'y','z','r']
    skeleton_id = -1
    startrow_skeleton = 0
    for i, nrow in CSVdata.iterrows():
        if nrow['skeleton_id'] != skeleton_id:
            skeleton_id = nrow['skeleton_id']
            startrow_skeleton = i
            if np.isnan(nrow['parent_treenode_id']):
                continue
        elif np.isnan(nrow['parent_treenode_id']):
            temp = CSVdata.iloc[i]
            CSVdata.iloc[i] = CSVdata.iloc[startrow_skeleton]
            CSVdata.iloc[startrow_skeleton] = temp
        else:
            continue
    print(CSVdata.head(80))
    return CSVdata
'''
