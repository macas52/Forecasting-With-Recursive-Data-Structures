#!/usr/bin/env python3

import numpy as np
import dill as pickle
import os
from BuildingDS.buildData import getData
from BuildingDS.buildData import getTest_Validate

path = './DataStructures/LongData.npy'
if os.path.isfile(path):
    DATA = np.load(path)
else:
    DATA = getData()


N_1 = 100_000
N_2 = int(1e6 * 7.5)
ds = 10
K = pickle.load(open(f'./DataStructures/DS{N_1}/partFamily{N_2}.pkl','rb'))
Indices = np.load(f'./DataStructures/DS{N_1}/Sortings/{N_1}sorted{N_2}.npy', allow_pickle=True)

sortingPoints = DATA[N_1: 2*N_1 + N_2] * K.Scale # N_2 points. have ds apart. (length=N_1 + N_2)

neighbours = np.load(f'./DataStructures/DS{N_1}/Neighbours/{N_1}neighs{N_2}.npy', allow_pickle=True)
dist = np.load(f'./DataStructures/DS{N_1}/Corrections/{N_1}dists{N_2}.npy', allow_pickle=True)

Covs = np.load(f'./DataStructures/DS{N_1}/CovMatrices/{N_1}Covs{N_2}.npy', allow_pickle=True)


for i in range(len(Indices)):
    K.children[i].indices = np.array(Indices[i])
    K.children[i].npoints = len(Indices[i])
    K.children[i].distDiffs = dist[i]
    K.children[i].neighbours = neighbours[i]
    K.children[i].PointsIn = Covs[i]

from ForecastingModule.Forecasting import *

"""
Generate Results
"""

N_Processes = 16
test_split, validate_split = getTest_Validate(K,ds,Num_Processes=N_Processes)

for N_E in [1,5,10,15,20,100,150]: # 50 has already been calculated.
    path = f'./Results/{N_1}_Unadjusted/{N_1}Unadjusted{N_2}.npy'
    if not os.path.exists(path):
        avgHorizonForecast_UnAdjusted_Model(K,N_2,N_1,ds, N_Processes)
    print('UnAdjusted Model Done')

    path = f'./Results/{N_1}_Adjusted/{N_1}Adjusted{N_2}_{N_E}.npy'
    if not os.path.exists(path):
        avgHorizonForecast_Adjusted_Model(K, N_E, N_2, N_1, test_split,validate_split,ds, N_Processes)
    print('Adjusted Model Done')

    print(f"{N_E} Ensemble Done")


for N_E in [1,5,10,15,20, 50, 100, 150]:
    path = f'./Results/{N_1}_DynamicPF/{N_1}PF_Dynamic{N_2}_{N_E}.npy'
    if not os.path.exists(path):
        avgHorizonForecast_Particle_Filter_Model_Dynamic(K,N_E, N_2, N_1, test_split,validate_split, sortingPoints,ds, N_Processes)
    print('Dynamic PF Model Done')

    path = f'./Results/{N_1}_StaticPF/{N_1}PF_STATIC{N_2}_{N_E}.npy'
    if not os.path.exists(path):
        avgHorizonForecast_Particle_Filter_Model_STATIC(K,N_E, N_2, N_1, test_split,validate_split, sortingPoints,ds, N_Processes)
    print('STATIC PF Model Done')

    print(f"{N_E} Ensemble Done")

