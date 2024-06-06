

"""
Dependencies
"""
import numpy as np
import matplotlib.pyplot as plt

def generatePartition(N_points_To_Split,N_1 , N_transient, dt):
    """
    Generate Realisations of the system and model using
    the Euler-Maruyama method of integration.
    - take N_1 size realisations
    """
    from Dyson.Num_Integration import Rho42lorenzTimeSeries, STDlorenzTimeSeries
    u0 = [0,1,1.05]

    l42 = np.load('./DataStructures/LongData.npy')[0:N_1]

    l_model = STDlorenzTimeSeries(
       N_1 + N_transient - 1,
       u0,
       dt,
       "E"
    )[N_transient:]

    """
    Scaling the system Realisation
    """
    from Dyson.Initialise_ODE import varianceScaling, initPartitionParmams
    hat_l42 = varianceScaling(l42,l_model)
    diameter, _ = initPartitionParmams(hat_l42)
    center_point = getSeparatrix(42) * np.std(l_model)/np.std(l42)
    """
    Partitioning
    - I need the starting box to be the same size.
    - Therefore they will all be scaled by the same amount & have the same initial bounding box.
    """
    from Dyson.Partition import partitionFamily
    K = partitionFamily()
    K.getKids([1,N_points_To_Split],hat_l42, center_point, diameter)
    K.Scale = np.std(l_model) / np.std(l42)
    K.clearPoints()


    import dill as pickle
    import os
    path = f'./DataStructures/DS{N_1}'
    if not os.path.exists(path):
        os.mkdir(path)
    pickle.dump(K, open(''.join([path,'/partFamily.pkl']),'wb'))

    print('-'*10 + '__FINISHED_MAKING_PARTITION__' + '-'*10)

    return K

def getSeparatrix(rho):
    return np.array([0,0,rho-1])
