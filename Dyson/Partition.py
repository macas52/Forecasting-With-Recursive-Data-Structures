#!/usr/bin/env python3

import numpy as np
from itertools import combinations
from Dyson.Bools import *
from Dyson.Recursion import *
from Dyson.Num_Integration import *
import multiprocessing as mp

# from Dyson.Forecasting import *

from tqdm import tqdm




class partition:
    """
    Defines a cube partition centred at the centre_point with sidelength
    """

    def __init__(self, centre_point, sidelength, parent=None):
        self.point = centre_point
        self.size = sidelength
        self.pointsIn = []
        self.indices = []
        self.npoints = 0 # The number of points contained in the partition.

        self.neighbours = []

        self.distDiffs = []

        self.parent=parent
        self.kids = []

    # Returns true or false whether a point is contained in a partition
    def inPartition(self, coord):
        return inCube(self.point,self.size, coord)

    # Counts how many points are in all the partitions
    def pointsInPartition(self, points):
        points = np.array(points)
        self.pointsIn = points[[self.inPartition(coord) for coord in points]]
        self.npoints = len(self.pointsIn)

    # Spawns children -> Don't touch
    def spawnChildren(self, pointRange):
        """
        Need 8 Children, kill the empty ones. WORKS, DON'T Touch
        """
        # Define values to add to the centre to get points
        vals = [self.size/4 * (-1)**i for i in range(8)]

        # Define positions for centres of the 8 cubes
        points = np.unique(np.array(list(combinations(vals, 3))), axis=0) + self.point

        # Define an array of children (8).
        children = [partition(p, self.size/2, parent=self) for p in points]

        # Stores all the children that have partitions with 1-5 points
        survivors = []

        # Iterate through all children
        for child in children:
            # Calc points in children
            child.pointsInPartition(self.pointsIn) # Takes parent points and stores ones that fit
            # If the number of points is within the point range. Don't split anymore.
            if child.npoints >= pointRange[0] and child.npoints <= pointRange[1]:
                # This partitions kid includes this child
                self.kids += [child]
                # The final children includes this child
                survivors += [child]
            elif child.npoints > pointRange[1]:
                self.kids += [child]
                # It is outside the range, so we split another generation.
                survivors += [child.spawnChildren(pointRange)]
        return flattenList(survivors)

    def buildDist(self, systemU, ds,dt):
        """
        Builds distribution of corrections
        """
        # Add ds time steps to the points in the partition.
        neighPoints = np.array(self.indices + ds * np.ones(self.npoints, dtype=int))
        # Only use ones that are contained in  the time series
        # Apply filter
        Y1index = neighPoints

        Y0index = np.array(self.indices)

        Y0 = systemU[Y0index]
        Y1 = systemU[Y1index]
        # Integrate our model over that time frame
        X1 = np.array([EulerMaruyama(lorenzSTD,y0, dt,ds)[-1] for y0 in Y0])
        # subtract the respective points
        D1 = np.array(Y1 - X1)
        self.distDiffs = D1
        return self.distDiffs

    def clearPartition(self):
        self.npoints = 0
        self.pointsIn = []
        self.indices = []

    def clearPoints(self):
        # empty cube
        self.clearPartition()

        if  len(self.kids) == 0:
            return

        for kid in self.kids:
            kid.clearPoints()

class partitionFamily:
    """
    Groups together partitions
    """
    def __init__(self):
        self.children = []
        self.Scale = None

    def origin(self):
        return original(self.children[0])

    def findPartition(self,point):
        origin = self.origin()
        for kid in origin.kids:
            if kid.inPartition(point):
                return findPoint(kid, point)

    def getKids(self,pointRange,u, centre, maxvar):
        """
        Start with a partition that encompases the entire data set, centered at the median
        Spawn children and get rid of the empty ones.
        """
        head = partition(centre, maxvar)
        head.pointsInPartition(u)
        self.children = head.spawnChildren(pointRange)

    def buildDistributions(self, system, ds,dt):
        """
        Calls the buildDist function for child nodes. # There was a tiny mistake.
        """
        for child in tqdm(self.children):
            child.buildDist(system, ds, dt)

    def clearPoints(self):
        """
        Deletes all the points in all partitions
        """
        self.origin().clearPoints()


    # def sortPoints(self, us):
    #     """
    #     Clears the partitions and populates them with a longer time series of points
    #     """
    #     self.clearPoints()
    #     # find starting point
    #     origin = self.origin()
    #     # Every other point will be in terms of the first
    #     # If we can't find a point for one we keep head the same.
    #     prev = origin
    #     i=-1
    #     for u in tqdm(us):
    #         i+=1
    #         # child = findPoint_Sorting(origin, u, i)
    #         child = findPoint_Sort_Improved(origin, u, prev, i,self)
    #         prev = child
    #         if child == None:
    #             continue

    # def getNeighbours(self, sysTimeSeries, ds):
    #     origin = self.origin()
    #     for child in tqdm(self.children):
    #         if child.npoints == 0:
    #             continue
    #         # Store the indices of the points in the neighbouring nodes
    #         neighbourPoints = np.array(child.indices + ds * np.ones(child.npoints,dtype=int))
    #         filterArr = np.array([point < len(sysTimeSeries) for point in neighbourPoints])
    #         neighs = np.array([findPoint_Improved(origin,sysTimeSeries[point],child,self) for point in neighbourPoints[filterArr]])
    #         child.neighbours = list(neighs[neighs != None])


    # def neighbour_To_Index(self):
    #     for child in tqdm(self.children):
    #         neighbours = child.neighbours
    #         child.neighbours = np.unique([self.children.index(child) for child in neighbours])

    def getCubes(self):
        """
        Exports the partition locations and dimensions.
        """
        cubes = np.array([[child.size, child.point] for child in self.children],dtype=object)
        np.save("./Blender/cubesPlotting", cubes)


