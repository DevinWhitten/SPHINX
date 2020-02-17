## author: devin Whitten
## I need to store statistics functions somewhere

import numpy as np
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d
from statsmodels.nonparametric.kde import KDEUnivariate
import MAD


def MAD(input_vector):

    return np.median(np.abs(input_vector - np.median(input_vector)))

def MAD_finite(input_vector):
    ### Special case of MAD where we need to remove np.nans from consideration
    ### intended especially for the is_interpolating() self.flag in network_array
    input_vector = input_vector[np.isfinite(input_vector)]
    return np.median(np.abs(input_vector - np.median(input_vector)))


def S_MAD(array):
    return MAD_finite(array)/0.6745


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

def gaussian_sigma(residuals, clip=5.0, bins=20, normed=True):
    ### DEPRECIATED
    #return the mean and sigma of the distribution according to a histogram fit
    working = residuals[residuals.between(np.mean(residuals) - clip*np.std(residuals),
                                          np.mean(residuals) + clip*np.std(residuals),
                                          inclusive=True)]

    HIST = np.histogram(working, bins=bins, density=normed)

    ### get bin centers
    xbins = [0.5 * (HIST[1][i] + HIST[1][i+1]) for i in range(len(HIST[1])-1)]
    ybins = HIST[0]
    popt, pcov = curve_fit(GAUSS,xbins, ybins,p0=[max(ybins),np.mean(xbins),np.std(xbins)])
    popt, pcov = curve_fit(GAUSS,xbins, ybins,p0=popt)
    return popt, pcov

def linear_scale(input_vector, mean, scale):
    ### centers and scales input_vector according to the mean/scale
    return np.divide((input_vector-mean),scale)

def unscale(input_vector, mean, scale):
    ### transforms input_vector to original scale
    return scale * input_vector + mean
