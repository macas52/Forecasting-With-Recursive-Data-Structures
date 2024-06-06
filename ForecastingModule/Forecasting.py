#!/usr/bin/env python3
from multiprocessing import Process
import numpy as np
from numpy import random as rd
from tqdm import tqdm
import os
from scipy.spatial.distance import euclidean


from Dyson.Forecasting import findClose, randSample, shadowing_condition, distToBeinCube
from Dyson.Num_Integration import STDlorenzTimeSeries
from BuildingDS.sortPoints import split_list
from Dyson.Recursion import findPoint_Improved, findPoint
from ForecastingModule.ForecastingFunctions import get_Starting_Partition

from BuildingDS.buildData import getTest_Validate

"""
Adjusted Model
"""
def MP_avgHorizonForecast_Adjusted_Model(K, Num_ENS, N_2, N_1,  validate, test, ind):
   ds = 10
   dt = 1e-3

   T_horizons = []
   # I need to be careful in how I split the data for the test data and validation data.
   # Define Y
   N2 = len(validate)
   
   for p in tqdm(range(len(test))):
       X = np.empty((N2, 3))

       X[0] = test[p]
       origin = K.origin()
       current  = get_Starting_Partition(origin, X[0])

       prev = current

       i=0
       while shadowing_condition(X[i], validate[i+p], 5) and origin.inPartition(X[i]): # Fail if go out of partition
           if current == None:
               current = findClose(K, prev, X[i])
           X[i+1] = STDlorenzTimeSeries(ds , X[i], dt, "E")[-1] + randSample(current, Num_ENS)
           prev = current
           current = findPoint_Improved(origin, X[i+1], prev, K)
           i+= 1

       T_horizons += [i * dt * ds]

   N2 = len(validate)
   path = f'./Results/{N_1}_Adjusted/'
   if not os.path.exists(path):
       os.mkdir(path)
   np.save(''.join([path, f'{ind}forecast.npy']), np.array(T_horizons))


def avgHorizonForecast_Adjusted_Model(K, Num_ENS, N_2,N_1, test_split,validate_split,ds, Num_Processes = 10):

    processes = [
        Process(
            target=MP_avgHorizonForecast_Adjusted_Model,
            args=(
                K, Num_ENS, N_2, N_1, validate_split[i], test_split[i], i
            )
        )
        for i in range(Num_Processes)
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    path = f'./Results/{N_1}_Adjusted/'

    if not os.path.exists(path):
        os.mkdir(path)

    RESULTS = np.concatenate(
        [
            np.load(''.join([path, f'{ind}forecast.npy'])) for ind in range(Num_Processes)
        ]
    )
    for i in range(Num_Processes):
        os.remove(''.join([path, f'{i}forecast.npy']))

    np.save(''.join([path, f'{N_1}Adjusted{N_2}_{Num_ENS}.npy']), RESULTS)


"""
Unadjusted Model
"""
def MP_avgHorizonForecast_UnAdjusted_Model(N_2, N_1,  validate, test,ds, ind):
   dt = 1e-3

   T_horizons = []
   # I need to be careful in how I split the data for the test data and validation data.
   # Define Y
   N2 = len(validate)


   for p in tqdm(range(len(test))):
       X = np.empty((N2, 3))

       X[0] = test[p]

       i=0
       while shadowing_condition(X[i], validate[i+p], 5):
           X[i+1] = STDlorenzTimeSeries(ds , X[i], dt, "E")[-1]
           i+= 1
       T_horizons += [i * dt * ds]

   N2 = len(validate)
   path = f'./Results/{N_1}_Unadjusted/'
   if not os.path.exists(path):
       os.mkdir(path)
   np.save(''.join([path, f'{ind}forecast.npy']), np.array(T_horizons))


def avgHorizonForecast_UnAdjusted_Model(K,N_2,N_1, ds, Num_Processes = 10):

    test_split, validate_split = getTest_Validate(K,ds, Num_Processes)

    processes = [
        Process(
            target=MP_avgHorizonForecast_UnAdjusted_Model,
            args=(
                N_2, N_1, validate_split[i], test_split[i], i
            )
        )
        for i in range(Num_Processes)
    ]

    for p in processes:
        p.start()
    for p in processes:
        p.join()

    path = f'./Results/{N_1}_Unadjusted/'

    if not os.path.exists(path):
        os.mkdir(path)

    RESULTS = np.concatenate(
        [
            np.load(''.join([path, f'{ind}forecast.npy'])) for ind in range(Num_Processes)
        ]
    )
    for i in range(Num_Processes):
        os.remove(''.join([path, f'{i}forecast.npy']))

    np.save(''.join([path, f'{N_1}Unadjusted{N_2}.npy']), RESULTS)

"""
Particle Filter Static
"""

def Particlefilter_Original(child, ensemble_members, X,ds):
    dt = 1e-3

    # 1. Define a uniform distribution of points in the ensemble
    ztm1 = X[child.indices] # System
    stm1 = ztm1

    zt = X[child.indices + ds] # System realization at time t

    w = np.ones(len(zt)) / len(zt) # weighting of each point

    # Define parameters for multivariate normal distribution
    # Params from On Two Localized Particle Filter Methods

    eta = 0.5 # STD of observation error
    sigma = 12  # Variance of distance between points.

    Sig = np.identity(3) * eta

    Xis2 = []
    omegas2 = []
    # 50 uniform random samples
    for _ in range(200):
        # randomly select z_t and its associated weighting
        rIndex = rd.randint(0,int(len(zt)), dtype=int)
        Xi = zt[rIndex]
        omega = w[rIndex]

        # Sample from multivariate Normal distribution centered at the reality (observation).
        # z_t is the system state at time t. s_t is an observation of the system (the model).
        z = rd.default_rng().multivariate_normal(Xi, Sig)

        rho = lambda z, sigma: np.e**(-z**2/(2*sigma**2))
         # I treat my imperfect model as observer

        # St is the observation (what our model thinks it is)
        st = STDlorenzTimeSeries(ds, stm1[rIndex], dt, "E")[-1]

        p = rho(euclidean(z,st), sigma)

        r = rd.uniform(0,1)

        if p > r:
            Xis2 += [z]
            omegas2 += [omega * p]

    # Pick the most likely ones then average them. -> That is the result.
    Xis2, omegas2 = np.array(Xis2), np.array(omegas2)
    random_index = np.array([rd.randint(0,int(len(Xis2)), dtype=int) for _ in range(ensemble_members)])

    # normalizing
    weights = omegas2[random_index]/sum(omegas2[random_index])
    Xis2 = Xis2[random_index]
    # weighted average
    return np.sum([Xis2[i]*weights[i] for i in range(len(random_index))],axis=0)


def Changing_Parameters_Particlefilter(child, ensemble_members, X,ds):
    dt = 1e-3

    # 1. Define a uniform distribution of points in the ensemble
    ztm1 = X[child.indices] # System
    stm1 = ztm1

    zt = X[child.indices + ds] # System realization at time t

    w = np.ones(len(zt)) / len(zt) # weighting of each point


    Sig = child.PointsIn[0]
    mu = child.PointsIn[1]
    sigma2 = child.PointsIn[2]

    Xis2 = []
    omegas2 = []
    # 50 uniform random samples
    for _ in range(200):
        # randomly select z_t and its associated weighting
        rIndex = rd.randint(0,int(len(zt)), dtype=int)
        Xi = zt[rIndex]
        omega = w[rIndex]

        # Sample from multivariate Normal distribution centered at the reality (observation).
        # z_t is the system state at time t. s_t is an observation of the system (the model).
        z = rd.default_rng().multivariate_normal(Xi, Sig)

        rho = lambda z, sigma, mu: np.e**(-(z-mu)**2/(2*sigma))
         # I treat my imperfect model as observer

        # St is the observation (what our model thinks it is)
        st = STDlorenzTimeSeries(ds, stm1[rIndex], dt, "E")[-1]

        p = rho(euclidean(z,st), sigma2, mu)

        r = rd.uniform(0,1)

        if p > r:
            Xis2 += [z]
            omegas2 += [omega * p]

    Xis2, omegas2 = np.array(Xis2), np.array(omegas2)

    # uniformly random sample.
    random_index = np.array([rd.randint(0,int(len(Xis2)), dtype=int) for _ in range(ensemble_members)])

    # normalizing
    weights = omegas2[random_index]/sum(omegas2[random_index])
    Xis2 = Xis2[random_index]
    # weighted average
    return np.sum([Xis2[i]*weights[i] for i in range(len(random_index))],axis=0)

def MP_avgHorizonForecast_Particle_Filter_Model_Dynamic(K,Num_ENS, sortingPoints, N_2, N_1, validate, test,ds,ind):
    dt = 1e-3
    T_horizons = []

    N2 = len(validate)

    for p in tqdm(range(len(test))):
        X = np.empty((N2,3))

        X[0] = test[p]

        origin = K.origin()
        current = get_Starting_Partition(origin, X[0])
        prev = current

        i=0
        while shadowing_condition(X[i],validate[i+p],5) and origin.inPartition(X[i]):
            if current == None:
                current = findClose(K,prev, X[i])
            X[i+1] = Changing_Parameters_Particlefilter(current,Num_ENS, sortingPoints,ds)
            prev = current
            current = findPoint_Improved(origin, X[i+1], prev, K)
            i += 1

        T_horizons += [i*dt*ds]

    N2 = len(validate)
    path = f'./Results/{N_1}_DynamicPF/'
    if not os.path.exists(path):
         os.mkdir(path)
    np.save(''.join([path, f'{ind}forecast.npy']), np.array(T_horizons))


def MP_avgHorizonForecast_Particle_Filter_Model_STATIC(K,Num_ENS,sortingPoints, N_2, N_1, validate, test,ds,ind):
    dt = 1e-3
    T_horizons = []
    N2 = len(validate)

    for p in tqdm(range(len(test))):
        X = np.empty((N2,3))

        X[0] = test[p]

        origin = K.origin()
        current = get_Starting_Partition(origin, X[0])
        prev = current

        i=0
        while shadowing_condition(X[i],validate[i+p],5) and origin.inPartition(X[i]):
            if current == None:
                current = findClose(K,prev, X[i])
            X[i+1] = Particlefilter_Original(current,Num_ENS, sortingPoints,ds)
            prev = current
            current = findPoint_Improved(origin, X[i+1], prev, K)
            i += 1

        T_horizons += [i*dt*ds]

    N2 = len(validate)
    path = f'./Results/{N_1}_StaticPF/'
    if not os.path.exists(path):
         os.mkdir(path)
    np.save(''.join([path, f'{ind}forecast.npy']), np.array(T_horizons))



def avgHorizonForecast_Particle_Filter_Model_STATIC(K,Num_ENS, N_2, N_1,test_split,validate_split, sortingPoints,ds, Num_Processes = 10):

    processes = [
       Process(
          target=MP_avgHorizonForecast_Particle_Filter_Model_STATIC,
          args=(
             K, Num_ENS, sortingPoints, N_2, N_1, validate_split[i], test_split[i],ds, i
          )
       )
       for i in range(Num_Processes)
    ]
    for p in processes:
       p.start()
    for p in processes:
       p.join()

    path = f'./Results/{N_1}_StaticPF/'

    if not os.path.exists(path):
       os.mkdir(path)

    RESULTS = np.concatenate(
       [
            np.load(''.join([path, f'{ind}forecast.npy'])) for ind in range(Num_Processes)
       ]
    )
    for i in range(Num_Processes):
        os.remove(''.join([path, f'{i}forecast.npy']))

    np.save(''.join([path, f'{N_1}PF_STATIC{N_2}_{Num_ENS}.npy']), RESULTS)

def avgHorizonForecast_Particle_Filter_Model_Dynamic(K,Num_ENS, N_2, N_1, test_split, validate_split,sortingPoints,ds, Num_Processes = 10):

    processes = [
       Process(
          target=MP_avgHorizonForecast_Particle_Filter_Model_Dynamic,
          args=(
             K, Num_ENS, sortingPoints, N_2, N_1, validate_split[i], test_split[i],ds, i
          )
       )
       for i in range(Num_Processes)
    ]
    for p in processes:
       p.start()
    for p in processes:
       p.join()

    path = f'./Results/{N_1}_DynamicPF/'

    if not os.path.exists(path):
       os.mkdir(path)

    RESULTS = np.concatenate(
       [
            np.load(''.join([path, f'{ind}forecast.npy'])) for ind in range(Num_Processes)
       ]
    )
    for i in range(Num_Processes):
        os.remove(''.join([path, f'{i}forecast.npy']))

    np.save(''.join([path, f'{N_1}PF_Dynamic{N_2}_{Num_ENS}.npy']), RESULTS)
