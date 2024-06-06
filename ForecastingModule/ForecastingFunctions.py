#!/usr/bin/env python3
from Dyson.Recursion import *
from Dyson.Forecasting import *
import numpy as np

def get_Starting_Partition(origin, point):
    partition = findPoint(origin, point)
    if partition != None:
        return partition
    smallest_containing = findClosest(origin, point)

    return findClosest_kid(smallest_containing, point)

def findClosest(origin, point):
    if origin.inPartition(point):
        if len(origin.kids) == 0 or sum([kid.inPartition(point) for kid in origin.kids]) == 0:
            return origin
        else:
            for kid in origin.kids:
                if kid.inPartition(point):
                    return findClosest(kid, point)

def findClosest_kid(parent, point):
        if len(parent.kids) == 0:
            return parent
        else:
            dist = np.inf
            closest = None
            for kid in parent.kids:
                d = distToBeinCube(kid, point)
                if d < dist:
                    dist = d
                    closest = kid
            return findClosest_kid(closest, point)
