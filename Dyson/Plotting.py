#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib
from itertools import combinations, product
import numpy as np

def axisIn3D(title="", figsize=(16,9),fontsize=16):
    import matplotlib.style as style
    style.use('seaborn-v0_8-colorblind')
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    ax = plt.figure(figsize=figsize).add_subplot(projection='3d')
    ax.set_xlabel("X Axis", fontsize = int(fontsize * 0.6))
    ax.set_ylabel("Y Axis", fontsize = int(fontsize * 0.6))
    ax.set_zlabel("Z Axis", fontsize = int(fontsize * 0.6))
    ax.set_title(title,fontsize = fontsize)
    return ax

def Plot3D(ax,x,label="",color="", len_w=0.5):
    if color == "":
        ax.plot(*x.T, lw=len_w,label=label)
    else:
        ax.plot(*x.T,lw=len_w,label=label,color=color)
    return ax

"""
Add cubes to the plot (DON'T TOUCH)
"""
def addCubes(ax,points, sidelength):  # Plotting
    for point in points:
        ax = addCube(ax,point, sidelength)
    return ax

"""
Add a single cube (DON'T TOUCH)
"""
def addCube(ax,point,sidelength,color="green"):
    r = np.array([-1,1]) * sidelength / 2
    for s, e in combinations(np.array(list(product(r, r, r))), 2):
        s = s + point
        e = e + point
        if np.sum(np.abs(s-e)) == r[1]-r[0]:
            try:
                ax.plot3D(*zip(s, e), color=color)
            except:
                raise Exception('The axis must be a 3D projection')
    return ax
