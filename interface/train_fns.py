### Author: Devin Whitten
### Date: 1/5/18

### This script serves as the interface for creating training sets
### from photometric catalogs.

import pandas as pd
import numpy as np
##### Interface for main training set generation procedure
import os, sys
sys.path.append("interface")
import io_functions
#import temperature_functions
#import param_teff as param
import itertools
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.ion()


def MAD(input_vector):

    return np.median(np.abs(input_vector - np.median(input_vector)))

def MAD_finite(input_vector):
    ### Special case of MAD where we need to remove np.nans from consideration
    ### intended especially for the is_interpolating() self.flag in network_array
    input_vector = input_vector[np.isfinite(input_vector)]
    return np.median(np.abs(input_vector - np.median(input_vector)))

def weighted_error(input_vector, scores):
    ### Weight the error estimate by the individual network scores
    #### len(input_vector) == len(scores) !!
    ## remove np.nans from is_interpolating() check
    scores = scores[np.isfinite(input_vector)]
    input_vector = input_vector[np.isfinite(input_vector)]
    #####

    ### First weighted average
    average = np.dot(input_vector, scores)/(scores.sum())

    TOP = np.dot(scores, np.power(input_vector-average, 2))
    MID = (float(len(scores[scores != 0.0])) - 1.0) * scores.sum()
    BOT = float(len(scores[scores != 0.0]))


    return np.sqrt(TOP/MID/BOT)



def GAUSS(x, a, b, c):
    return a * np.exp(-(x - b)**2.0 / (2 * c**2))


def write_catalogs(catalog, train_fct, mode="SEGUE"):
    #### Shuffles the catalog, writes out the appropriate fractions
    #### to datafiles
    shuffle = catalog.iloc[np.random.permutation(len(catalog))]

    ### write to cache
    shuffle.iloc[0:int(len(shuffle)*train_fct)].to_csv("cache/"+mode+"_training.csv", index=False)
    shuffle.iloc[int(len(shuffle)*train_fct):].to_csv("cache/"+mode+"_testing.csv", index=False)


def gaussian_sigma(residuals, clip=5.0, bins=20, normed=True):
    #return the mean and sigma of the distribution according to a histogram fit
    working = residuals[residuals.between(np.mean(residuals) - clip*np.std(residuals),
                                          np.mean(residuals) + clip*np.std(residuals),
                                          inclusive=True)]

    HIST = np.histogram(working, bins=bins, normed=normed)

    ### get bin centers
    xbins = [0.5 * (HIST[1][i] + HIST[1][i+1]) for i in range(len(HIST[1])-1)]
    ybins = HIST[0]
    #print("Numpy xbins:  ", xbins)
    #print("Numpy ybins:  ", ybins)
    popt, pcov = curve_fit(GAUSS,xbins, ybins,p0=[max(ybins),np.mean(xbins),np.std(xbins)])
    popt, pcov = curve_fit(GAUSS,xbins, ybins,p0=popt)
    #print(popt)
    return popt, pcov

def Linear_Scale(input_vector, mean, scale):
    ### centers and scales input_vector according to the mean/scale
    return np.divide((input_vector-mean),scale)

def unscale(input_vector, mean, scale):
    ### transforms input_vector to original scale
    return scale * input_vector + mean



def random_layer(hidden_layers):
    ### return the necessary hidden_layer version
    ### given specified type.. basically int or tuple

    #handle int vrs tuple
    if type(hidden_layers) == int:
        return np.random.randint(1, hidden_layers)

    elif (type(hidden_layers) == tuple) or (type(hidden_layers) == list):
        left, right = hidden_layers[0], hidden_layers[1]

        return (np.random.randint(1, left),np.random.randint(1,right))

################################################################################

class train_pool():
    ### just little wrapper to assist the multiprocess training
    def __init__(self, training_set, network_array):
        ### all it really needs is the training set,
        ### so it doesn't have to be distributed to all networks as a copy
        self.training_set = training_set
        self.network_array = network_array

        return

    def train_network(self, input_tuple):
        ### I think I need this function to handle the multiprocessing
        net, ID = input_tuple
        net.train_on(self.training_set, ID=ID)

        return net


################################################################################







##### Master Class for the Training set, I need to solve to the problem of an overcomplicated main.py
