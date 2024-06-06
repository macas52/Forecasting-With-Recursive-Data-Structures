#!/usr/bin/env python3
import os
import numpy as np
import dill as pickle

from BuildingDS.buildData import getData
from BuildingDS.sortPoints import *
from BuildingDS.partitioning import generatePartition

"""
Get long data
"""
N_proc = 12
ds = 10

def buildDataStructure(N_1, N_2):
   path = './DataStructures/LongData.npy'
   if os.path.isfile(path):
       DATA = np.load(path)
   else:
       getData()
       DATA = np.load(path)
       print("Data Acquired")

   path=f'./DataStructuresPF/DS{N_1}'
   if os.path.isfile(''.join([path, '/partFamily.pkl'])):
       K = pickle.load(open(''.join([path,'/partFamily.pkl']),'rb'))
   else:
       K = generatePartition(5, N_1, 100_000, 1e-3) # N_transient = 100_000, dt= 1e-3

   """
   Sorting
   - We keep the data sorted in at partition.
   """
   DATA = DATA * K.Scale

   sortingPoints = DATA[N_1 : 2*N_1 + N_2] # N_2 points. have ds apart. (length=N_1 + N_2)

   if not os.path.isfile(f'./DataStructures/DS{N_1}/Sortings/{N_1}sorted{N_2}.npy'):
       sort_save_partition(K, N_1, N_2, sortingPoints[0:-ds], Num_Processes=N_proc)
       print(f"Sorted Generated: {N_1}_{N_2}")
   else:
       print(f"Sorted Opened: {N_1}_{N_2}")

   print(f"______________________Sorting Acquired_{N_1}_{N_2}____________________")

   """
   Neighbours
   """
   N_1Dir = f'./DataStructures/DS{N_1}'
   K = pickle.load(open(''.join([N_1Dir,f'/partFamily{N_2}.pkl']),'rb'))
   Indices = np.load(''.join([N_1Dir,f'/Sortings/{N_1}sorted{N_2}.npy']), allow_pickle=True)
   for i in range(len(Indices)):
       K.children[i].indices = Indices[i]
       K.children[i].npoints = len(Indices[i])

   if not os.path.isfile(f'./DataStructures/DS{N_1}/Neighbours/{N_1}neighs{N_2}.npy'):
       get_Neighbours(K, N_1, N_2, sortingPoints, ds, N_proc)
   else:
       print(f"already generated {N_1}neighs{N_2}")

   print(f"______________________Neighbours Acquired_{N_1}_{N_2}____________________")

   """
   Corrections
   """
   if not os.path.isfile(f'./DataStructures/DS{N_1}/Corrections/{N_1}dists{N_2}.npy'):
       get_Corrections(K,N_1, N_2, sortingPoints,ds, N_proc)
       print(f"Corrections generated N_2 = {N_2}, N_1 = {N_1}")
   else:
       print(f"Corrections Exists N_2 = {N_2}, N_1 = {N_1}")
   print(f"______________________Corrections Acquired_{N_1}_{N_2}____________________")

   """ Cov Matrices"""
   if not os.path.isfile(f'./DataStructures/DS{N_1}/CovMatrices/{N_1}Covs{N_2}.npy'):
      dist = np.load(f'./DataStructures/DS{N_1}/Corrections/{N_1}dists{N_2}.npy',allow_pickle = True)
      for i in range(len(K.children)):
         K.children[i].distDiffs = dist[i]
      get_covMatrices(K, N_1, N_2, N_proc)
   print(f"______________________Covs Acquired_{N_1}_{N_2}____________________")


# Maximum I'll go up to.
N_1, N_2 = int(1e5), int(7.5*1e6)
buildDataStructure(N_1,N_2)
# Done Building
