import numpy as np

def MAD(array):
    return np.median(np.abs(array - np.median(array)))

def S_MAD(array):
    return MAD(array)/0.6745
