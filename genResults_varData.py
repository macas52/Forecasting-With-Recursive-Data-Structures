#!/usr/bin/env python3
from BuildingDS.buildData import getTest_Validate
import numpy as np
import dill as pickle
import os

N_E = 50
N_proc = 12
ds = 10

from ForecastingModule.Forecasting import avgHorizonForecast_Adjusted_Model
DATA = np.load('./DataStructures/LongData.npy')

def generate_var_data(N_1, N_2,N_E):
     K = pickle.load(open(f'./DataStructures/DS{N_1}/partFamily{N_2}.pkl','rb'))
     Indices = np.load(f'./DataStructures/DS{N_1}/Sortings/{N_1}sorted{N_2}.npy', allow_pickle=True)
     neighbours = np.load(f'./DataStructures/DS{N_1}/Neighbours/{N_1}neighs{N_2}.npy', allow_pickle=True)
     dist = np.load(f'./DataStructures/DS{N_1}/Corrections/{N_1}dists{N_2}.npy', allow_pickle=True)

     for i in range(len(Indices)):
         K.children[i].indices = Indices[i]
         K.children[i].npoints = len(Indices[i])
         K.children[i].distDiffs = dist[i]
         K.children[i].neighbours = neighbours[i]
     test_split, validate_split = getTest_Validate(K,ds,Num_Processes=N_proc)
     avgHorizonForecast_Adjusted_Model(K, N_E, N_2, N_1, test_split,validate_split,ds, N_proc)
     print("_"*10 + f"{N_1}:{N_2} Done" + "_" * 10)

def GenerateResults_Var_DATA(N1s, N2s, N_E):
    for N_1 in N1s:
        for N_2 in N2s:
            K = pickle.load(open(f'./DataStructures/DS{N_1}/partFamily{N_2}.pkl','rb'))
            origin = K.origin()

            test_split, _ = getTest_Validate(K,ds,Num_Processes=N_proc)
            testing = np.concatenate(test_split)

            if  sum([not origin.inPartition(p) for p in testing]) != 0:
                print('FAILURE')
                if not os.path.exists(f'./Results/{N_1}_Adjusted/'):
                    os.mkdir(f'./Results/{N_1}_Adjusted/')
                np.save(f'./Results/{N_1}_Adjusted/{N_1}Adjusted{N_2}_50.npy',np.zeros(len(testing)))
                continue
            if not os.path.isfile(f'./Results/{N_1}_Adjusted/{N_1}Adjusted{N_2}_50.npy'):
                generate_var_data(N_1,N_2,N_E)
        print("_"*10 + f"{N_1} Done Completely" + "_" * 10)
