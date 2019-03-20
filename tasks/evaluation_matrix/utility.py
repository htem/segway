import math 
from daisy import Coordinate
def shortest_euclidean_bw_two_sk(set1,set2):
    shortest_len = math.inf
    shortest_datapoint = set()
    for point1 in set1:
        for point2 in set2:
            distance = math.sqrt(sum([(a-b)**2 for a,b in zip(point1,point2)]))
            if distance < shortest_len:
                shortest_len = distance
                shortest_datapoint = (Coordinate(point1),Coordinate(point2))
    return shortest_datapoint
    
    