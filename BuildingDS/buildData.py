#!/usr/bin/env python3

import numpy as np
from Dyson.Initialise_ODE import varianceScaling
from Dyson.Num_Integration import Rho42lorenzTimeSeries, STDlorenzTimeSeries
from BuildingDS.sortPoints import split_list

def getData():
    N_DATA = int(2*1e7)
    N_transient = int(1e5)
    dt = 1e-3
    u0=[0,1,1.05]

    DATA_UnScaled = Rho42lorenzTimeSeries(
        N_DATA + N_transient - 1,
        u0,
        dt,
        "E"
    )[N_transient:]

    path = './DataStructures/LongData.npy'
    np.save(path, DATA_UnScaled)

    LONG_DATA_test_valid = Rho42lorenzTimeSeries(
        N_DATA + int(2 * 1e5) + N_transient,
        u0,
        dt,
        "E"
    )[(N_transient + N_DATA):]

    test = LONG_DATA_test_valid[0:100_000]
    validate = LONG_DATA_test_valid
    np.save('./DataStructures/test.npy',test)
    np.save('./DataStructures/validate.npy',validate)
    return DATA_UnScaled


def getTest_Validate(K, ds, Num_Processes):
    test = np.load('./DataStructures/test.npy')[::ds] * K.Scale
    validate = np.load('./DataStructures/validate.npy')[::ds] * K.Scale
    indices = list(range(0, len(test)))

    indices_split = split_list(indices, Num_Processes)
    test_split = split_list(test,Num_Processes)
    validate_split = [validate[split[0]::] for split in indices_split]
    return test_split, validate_split
