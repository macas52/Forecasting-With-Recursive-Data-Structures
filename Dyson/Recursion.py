#!/usr/bin/env python3

def recurseParent(point, child):
    # Returns the smallest parent that contains the point\
    # Stops when the child of the desired parent (allows for 1 less operation).
    if child == None:
        return None
    parent = child.parent
    if parent == None: # Go out of bounds
        return None
    elif parent.inPartition(point):
        return parent

    else:
        return recurseParent(point, parent)

def recurseChild(point, parent, K):
    if parent == None:
        return None
    for child in parent.kids:
        if child.inPartition(point):
            if len(child.kids) == 0 and child in K.children:
                return child
            else:
                return recurseChild(point,child, K)
        else:
            continue

"""
Returns a 1D array (LIST) from a multidimensional nested array
Eg.
[[1,2,[3,7]],1,4] -> [1,2,3,7,1,4]
(DON'T TOUCH)
"""
def flattenList(lst):
    D1 = []
    for el in lst:
        if type(el) == list:
            D1 += flattenList(el)
        else:
            D1 += [el]
    return D1

def original(child):
    if child.parent == None:
        return child
    else:
        return original(child.parent)

def findPoint(origin, point):
    if origin.inPartition(point):
        if len(origin.kids) == 0:
            return origin
        else:
            for kid in origin.kids:
                if kid.inPartition(point):
                    return findPoint(kid,point)
def findPoint_Sorting(origin, point, index):
    if origin == None:
        return None

    if origin.inPartition(point):
        if len(origin.kids) == 0:
            origin.npoints += 1
            origin.pointsIn += [point]
            origin.indices += [index]
            return origin
        else:
            for kid in origin.kids:
                if kid.inPartition(point):
                    return findPoint_Sorting(kid,point,index)

def findPoint_Sort_Improved(origin, point,prev, index,K):
    if prev != None:
        ancestor = recurseParent(point, prev)
        child = recurseChild(point,ancestor, K)
    else:
        prev = origin
        child = recurseChild(point, origin, K)
    if child != None:
        child.npoints += 1
        child.pointsIn += [point]
        child.indices += [index]
    return child

def findPoint_Improved(origin, point,prev,K):
    if prev != None:
        ancestor = recurseParent(point, prev)
        child = recurseChild(point,ancestor,K)
    else:
        prev = origin
        child = recurseChild(point, origin,K)
    return child
