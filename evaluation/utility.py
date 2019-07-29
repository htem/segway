import math
from daisy import Coordinate
import networkx as nx


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
def to_pixel_coord_xyz(zyx, voxel_size):
    zyx = (Coordinate(zyx) / Coordinate(voxel_size))
    return Coordinate((zyx[2], zyx[1], zyx[0]))
