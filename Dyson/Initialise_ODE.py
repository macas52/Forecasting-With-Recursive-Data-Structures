#!/usr/bin/env python3
import numpy as np

def initPartitionParmams(u):
    return 4 * np.std(u), np.median(u,axis=0)
    # Use 3.5 standard deviations - This is enough to hold the test data.

def varianceScaling(uSys, uMod):
    model_std, sys_std = np.std(uMod), np.std(uSys)
    return uSys * model_std / sys_std
