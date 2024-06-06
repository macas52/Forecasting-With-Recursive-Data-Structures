#!/usr/bin/env python3
import dill as pickle
import numpy as np
import numpy.random as rd
from scipy.spatial.distance import euclidean
from itertools import combinations
# import matplotlib.pyplot as plt
from Dyson.Recursion import *
from Dyson.Num_Integration import *
from tqdm import tqdm




def randSample(cube, numEnsembles):
    filt = rd.choice(len(cube.distDiffs), numEnsembles,replace=True)
    selection = cube.distDiffs[filt]
    return np.mean(selection,axis=0)

def distToBeinCube(neighbour, point):
    vals = [neighbour.size * (-1)**i for i in range(8)]
    points = np.unique(np.array(list(combinations(vals,3))), axis=0) + neighbour.point
    points = np.vstack((points, neighbour.point))
    minDist = lambda arr, point: min([euclidean(val,point) for val in arr])
    return minDist(points, point)

"""
Has the trajectory diverged from the true trajectory. Returns False when it has departed sufficiently from the true trajectory.
"""
def shadowing_condition(x,y,theta):
    return euclidean(x,y) <= theta


"""Returns the neighbouring partition that is closest to the trajectory"""
def findClosest(origin, point):
    if origin.inPartition(point):
        if len(origin.kids) == 0 or sum([kid.inPartition(point) for kid in origin.kids]) == 0:
            return origin
        else:
            for kid in origin.kids:
                if kid.inPartition(point):
                    return findClosest(kid, point)

def findClosest_kid(parent, point, K):
        if len(parent.kids) == 0 and parent in K.children:
            return parent
        else:
            dist = np.inf
            closest = None
            for kid in parent.kids:
                d = distToBeinCube(kid, point)
                if d < dist:
                    dist = d
                    closest = kid
            return findClosest_kid(closest, point,K)

def findClose(K, prev, point):
    close_neighbour = None
    dist = np.inf

    if len(prev.neighbours) == 0:
        closest_parent = findClosest(K.origin(), point)
        return findClosest_kid(closest_parent, point,K)

    for neigh in prev.neighbours:
        d = distToBeinCube(K.children[neigh], point)
        if dist > d:
            close_neighbour = K.children[neigh]
            dist = d

    return close_neighbour


