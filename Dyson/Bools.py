#!/usr/bin/env python3

import numpy as np

def inCube(centre, sidelength, coord):
        x,y,z = centre
        coord = np.array(coord)
        s = sidelength/2

        c1 = x-s <= coord[0] <= x+s
        c2 = y-s <= coord[1] <= y+s
        c3 = z-s <= coord[2] <= z+s

        return c1 and c2 and c3
