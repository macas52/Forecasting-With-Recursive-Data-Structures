#!/usr/bin/env python3
from itertools import combinations
import numpy as np
import numpy.linalg as la
import os
import multiprocessing as mp
from Dyson.Recursion import findPoint_Improved, findPoint_Sort_Improved
from tqdm import tqdm
import dill as pickle
"""
Save in a separate folder varying based on N_2.

DS{N1} - partFamily.pkl, Sortings -> {somethingN2, somethingN2,...}, Neighbours -> {somethingN2...}
, {dists}, indices...

"""
def sortPoints_MP(K, C,j):
        """
        Clears the partitions and populates them with a longer time series of points
        """
        # find starting point
        origin = K.origin()
        # Every other point will be in terms of the first
        # If we can't find a point for one we keep head the same.
        prev = origin
        i=-1
        for u in tqdm(C[j]):
            i+=1
            child = findPoint_Sort_Improved(origin, u, prev, i, K) # Problem
            """
            When the index is being added it does not consider the previous points.
            """
            prev = child
            if child == None:
                continue
        filled_partitions = np.array([i for i in range(len(K.children))  if K.children[i].npoints != 0])
        points_n_index = np.array([
            [i, K.children[i].indices] for i in filled_partitions
        ],dtype='object')

        np.save(f'./Sortings/{j}sort.npy',points_n_index)

def sort_save_partition(K,N_1, N_2, sortingPoints, Num_Processes = 10):
    # Assumed to run if sortings for N_1 and N_2 do not exist.
    # K.sortPoints(sortingPoints)
    K.clearPoints()

    C = split_list(sortingPoints, Num_Processes)

    processes = [mp.Process(target=sortPoints_MP, args=(K, C, i)) for i in range(Num_Processes)]

    # P1, P2 = split_list(processes, 2)
    P1 = processes

    for p in P1:
        p.start()
    for p in P1:
        p.join()

    """FIXING SORTING"""
    def combine_sorts(load_arr, C):
        cumsum_len = np.cumsum([len(c) for c in C])
        cumsum_len = np.append([0], cumsum_len)
        B = [None for _ in range(len(K.children))]
        for j in range(len(load_arr)):
            for a in load_arr[j]:
                if B[a[0]] == None:
                    B[a[0]] = [aa + cumsum_len[j] for aa in a[1]]
                else:
                    B[a[0]] += [aa + cumsum_len[j] for aa in a[1]]
        return B

    sorted_arrays = [
        np.load(f'./Sortings/{i}sort.npy',allow_pickle=True) for i in range(Num_Processes)
    ]
    
    for i in range(Num_Processes):
    	os.remove(f'./Sortings/{i}sort.npy')

    Indices = combine_sorts(sorted_arrays,C)
    Indices = np.array(Indices, dtype='object')

    if sum([ind == None for ind in Indices]) == 0:
        path = f'./DataStructures/DS{N_1}/Sortings'
        if not os.path.exists(path):
            os.mkdir(path)
        np.save(''.join([path, f'/{N_1}sorted{N_2}.npy']), Indices)

        print("Finished_Sorting")
        path = f'./DataStructures/DS{N_1}'
        K.clearPoints()
        pickle.dump(K, open(''.join([path,f'/partFamily{N_2}.pkl']),'wb'))

    else:
        Indices = delete_Partitions(K,Indices)

        path = f'./DataStructures/DS{N_1}/Sortings'
        if not os.path.exists(path):
            os.mkdir(path)
        np.save(''.join([path, f'/{N_1}sorted{N_2}.npy']), Indices)

        print("Finished_Sorting")
        K.clearPoints()
        path = f'./DataStructures/DS{N_1}'
        pickle.dump(K, open(''.join([path,f'/partFamily{N_2}.pkl']),'wb'))
    # path = f'./DataStructures/DS{N_1}/Sortings'
    # if not os.path.exists(path):
    #     os.mkdir(path)

    # np.save(''.join([path, f'/{N_1}sorted{N_2}.npy']), Indices)
    # print("Finished_Sorting")

def disown_patricide(cube):
    parent = cube.parent
    if len(parent.kids) > 1:
        parent.kids.remove(cube)
    else:
        parent.kids.remove(cube)
        disown_patricide(parent)

def delete_Partitions(K, Indices):
    ind2 = list(Indices.copy())
    Partitions_to_remove = []
    for i in range(len(Indices)):
        if Indices[i] == None or len(Indices[i]) < 2:
            Partitions_to_remove += [K.children[i]]
            disown_patricide(K.children[i])
            ind2.remove(Indices[i])
    for empty_Partition in Partitions_to_remove:
        K.children.remove(empty_Partition)
    return np.array(ind2, dtype='object')

def split_list(arr,n):
    ls_len = len(arr)
    sizes = ls_len//n
    return [arr[sizes * i:sizes*(i+1)] for i in range(n-1)] + [arr[sizes * (n-1):]]

def getNeighbours_MP(K,sysTimeSeries, ds,C, i): # old
        print(f'Process {i}')
        origin = K.origin()
        neighs_arr = []
        for child in tqdm(C[i]):
            if child.npoints == 0:
                continue
        # Store the indices of the points in the neighbouring nodes
            neighbourPoints = np.array(child.indices + ds * np.ones(child.npoints,dtype=int))
            # filterArr = np.array([point < len(sysTimeSeries) for point in neighbourPoints])
            neighs = np.array([findPoint_Improved(origin,sysTimeSeries[point],child,K) for point in neighbourPoints])
            neighs = neighs[neighs != None]

            neighs_int = np.unique([K.children.index(neigh) for neigh in neighs])
            neighs_arr += [neighs_int]

        np.save(f'./Neighbours/{i}neigh.npy',np.array(neighs_arr, dtype='object'))

def get_Neighbours(K, N_1,N_2, sortingPoints,ds, Num_Processes=10):
    C = split_list(K.children, Num_Processes)
    processes = [mp.Process(target=getNeighbours_MP, args=(K,sortingPoints, ds, C,i)) for i in range(Num_Processes)]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    neighs = np.concatenate([np.load(f'./Neighbours/{i}neigh.npy', allow_pickle=True) for i in range(Num_Processes)])

    for i in range(Num_Processes):
        os.remove(f'./Neighbours/{i}neigh.npy')

    path = f'./DataStructures/DS{N_1}/Neighbours'
    if not os.path.exists(path):
        os.mkdir(path)
    np.save(''.join([path, f'/{N_1}neighs{N_2}.npy']), neighs)

def buildDistributions_MP(C,i,system, ds,dt):
        """
        Calls the buildDist function for child nodes. # There was a tiny mistake.
        """
        dists = []
        for child in tqdm(C[i]):
            child.buildDist(system, ds, dt)
            dists += [child.distDiffs]
        dists = np.array(dists, dtype='object')
        np.save(f'./Corrections/{i}dist.npy', dists)


def get_Corrections(K,N_1,N_2, sortingPoints,ds, Num_Processes):
    dt = 1e-3

    C = split_list(K.children, Num_Processes)

    processes = [mp.Process(target=buildDistributions_MP, args=(C, i,sortingPoints, ds,dt)) for i in range(Num_Processes)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()

    dists = np.concatenate([np.load(f'./Corrections/{i}dist.npy',allow_pickle=True) for i in range(Num_Processes)])

    for i in range(Num_Processes):
        os.remove(f'./Corrections/{i}dist.npy')

    path = f'./DataStructures/DS{N_1}/Corrections'
    if not os.path.exists(path):
        os.mkdir(path)
    np.save(''.join([path, f'/{N_1}dists{N_2}.npy']), dists)

def MP_covmatrices(children_split, N_1, N_2, i):
    PointsIn = []
    for child in tqdm(children_split):
        PointsIn += [[np.cov(np.array(child.distDiffs).T),
                      np.mean([la.norm(diff) for diff in child.distDiffs]),
                      np.var([la.norm(diff) for diff in child.distDiffs])]]

    np.save(f'./CovMatrices/{i}Covs.npy',np.array(PointsIn, dtype='object'))


def get_covMatrices(K, N_1, N_2, Num_Processes):
    C = split_list(K.children, Num_Processes)

    processes = [
            mp.Process(target = MP_covmatrices, args=(C[i], N_1, N_2, i) )
            for i in range(Num_Processes)
    ]

    for p in processes:
        p.start()

    for p in processes:
        p.join()

    COVS = np.concatenate([np.load(f'./CovMatrices/{i}Covs.npy', allow_pickle=True) for i in range(Num_Processes)])

    for i in range(Num_Processes):
        os.remove(f'./CovMatrices/{i}Covs.npy')

    path = f'./DataStructures/DS{N_1}/CovMatrices'
    if not os.path.exists(path):
        os.mkdir(path)
    np.save(''.join([path, f'/{N_1}Covs{N_2}.npy']), COVS)
