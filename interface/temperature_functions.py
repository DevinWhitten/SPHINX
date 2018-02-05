import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.optimize import curve_fit, minimize

### Author: Devin Whitten
### Date: 11/11/17

### This script serves as the interface for all functions associated with the
### temperature training sets.

def Linear_Scale(input_vector, mean, scale):
    ### centers and scales input_vector according to the mean/scale
    return np.divide((input_vector-mean),scale)

def MAD(input_vector):
    return np.median(np.abs(input_vector - np.median(input_vector)))

def GAUSS(x, a, b, c):
    return a * np.exp(-(x - b)**2.0 / (2 * c**2))


def gen_uniform(frame, TMIN, TMAX, BIN_NUMBER=20, SIZE=200):
    ### Accepts pandas dataframe of candidate training sample
    ### approximates uniform distribution in temperature
    ## BIN: number of temperature partitions
    ## SIZE: Number of stars per bin
    Bounds = np.linspace(TMIN, TMAX,BIN_NUMBER)
    BIN = []

    for i in range(len(Bounds) - 1):
        Current_Chunk = frame[(frame.TEFF > Bounds[i]) & (frame.TEFF < Bounds[i+1])]
        BIN.append(Current_Chunk.iloc[np.random.permutation(len(Current_Chunk))][0:SIZE])

    Reform = pd.concat(BIN)
    output_frame = Reform.iloc[np.random.permutation(len(Reform))]
    return output_frame


def force_normal(frame, column, bin, fixed_mean, fixed_std):
    ### Precondition: Force a gaussian distribution from the training set_scale_frame
    ###               according to the desired mean and std
    ### This procedure attempts to maximize the number of stars obtained, while
    ### maintaining a gaussian distribution
    ### Postcondition: Returns the resulting selection frame

    # first generate the x and y bins of the training data
    distro = np.array(frame[column][np.isfinite(frame[column])])
    hist = np.histogram(distro, bin)

    xbins =  [0.5 * (hist[1][i] + hist[1][i+1]) for i in range(len(hist[1])-1)]
    xedges = hist[1]
    ybins =  hist[0]

    #### a is the true variable
    def forced_gauss(x, a):
        return a * np.exp(-(x - fixed_mean)**2.0 / (2 * fixed_std**2))

    error_fun_root = lambda a : (np.power(np.abs(ybins - forced_gauss(xbins, a)), 1./3.)).sum()

    root_res = minimize(error_fun_root, 100, method='Nelder-Mead')

    BINS = []

    for i in range(len(xedges)-1):
        current = frame[frame[column].between(xedges[i], xedges[i+1], inclusive=True)]
        shuffle = current.iloc[np.random.permutation(len(current))]

        BINS.append(shuffle.iloc[0:int(forced_gauss(xbins[i], root_res.x))])

        print("x_cent:      ", xbins[i])
        print("max(x_cent): ", int(GAUSS(xbins[i], root_res.x)))
        print("Obtained:    ", len(BINS[i]))


    selection = pd.concat(BINS)

    custom = selection


def distribution_stats(input_vector):
    ### computes the necessary scale estimates for the input variable.
    ### standard deviation
    ### median absolute deviation
    ### gaussian to residuals

    SD = np.std(input_vector)
    MAD = np.median(np.abs(input_vector - np.median(input_vector)))

    mean,std=norm.fit(input_vector)  #### This function is the same as the np.std().. dumb

    return SD, MAD, std

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



################################################################################
def finite_frame(frame):
    ### Just remove any weird numbers in the target frame
    inputs = ["J0378", "J0395", "J0410", "J0430", "J0515", "J0660", "J0861", "gSDSS", 'rSDSS', 'iSDSS']

    for band in inputs:
        frame=frame[np.isfinite(frame[band])]

    return frame


def set_scale_frame(frame, inputs):
    ### Assume JPLUS_calibrated_catalog
    ### Generates Scale_Frame containing mean and std of input column distributions

    calibration = pd.DataFrame()

    for band in inputs:
        popt,pcov = gaussian_sigma(frame[band])
        calibration.loc[:,band] = [popt[1], popt[2]]

    return calibration

def set_interp_frame(frame, filters):
    ### returns a DataFrame with max/min limits for each specified input column
    ### Precondition: erroneous values in frame must be eliminated prior
    interp_frame = pd.DataFrame()

    for band in filters:
        interp_frame.loc[:, band] = [min(frame[band]), max(frame[band])]

    return interp_frame

def format_frame(frame, params):
    ### return dataframe of the specified column format
    if len(params['target_bands']) != len(params['format_bands']):
        print("Length mismatch in format_frame")

    length0 = len(frame)

    output = pd.DataFrame()

    ## Rename columns of output frame
    for i, band in enumerate(params['format_bands']):
        output.loc[:, params['format_bands'][i]] = frame.loc[:, params['target_bands'][i]]

    #### faint/bright
    for band in params['format_bands']:
        output = output[np.isfinite(output[band])]
        output = output[output[band].between(params['mag_bright_lim'], params['mag_faint_lim'],  inclusive=True)]
        print("Current length after ", band, len(output))

    print(length0 - len(output), " removed from formatting")

    return output

def scale_photometry(frame, coefficient_frame, columns):
    ### performs linear scaling on the input frame according to predefine column
    ### list
    ### Precondition: Assumes both frames have identical column names
    ### returns deep copied, scaled frame
    working = frame.copy(deep=True)  # I don't want to funk with the original frame
    for column_name in columns:
        print(column_name, coefficient_frame[column_name].iloc[0], coefficient_frame[column_name].iloc[1])
        #print(coefficient_frame[column_name])
        working.loc[:,column_name] = Linear_Scale(working[column_name], coefficient_frame[column_name].iloc[0], coefficient_frame[column_name].iloc[1])

    return working

def scale_frame_custom_column(frame, coef_frame, target_columns, coef_columns):
    ### performs linear scaling on the input frame according to predefine column
    ### list
    ### Precondition: Assumes both frames have identical column names
    ### returns deep copied, scaled frame
    working = frame.copy(deep=True)  # I don't want to funk with the original frame
    for i in range(len(target_columns)):
        print(target_columns[i], coef_columns[i])
        #print(coefficient_frame[column_name])
        working.loc[:,target_columns[i]] = Linear_Scale(working[target_columns[i]],
                                                        coef_frame[coef_columns[i]].iloc[0], coef_frame[coef_columns[i]].iloc[1])

    return working


################################################################################

def bad_spectra(frame):
    ### Remove spectra identified as faulty.
    errors = pd.read_csv("performance/bad_spectra.csv", header=None)

    return frame[~frame.SPSPEC.isin(errors[0])]

def teff_dispersion(frame, estimators, dispersion_limit=100.):
    ### rejects stars whose teff estimates defined by estimators
    ### exceeds dispersion limit
    dispersion_array = []
    for i,row in frame.iterrows():
        teff_array = np.array(row[estimators].values, dtype=float)
        dispersion_array.append(MAD(teff_array[(np.isfinite(teff_array)) & (teff_array > 0.0)]))

    dispersion_array = np.array(dispersion_array)
    subselection = frame[dispersion_array < dispersion_limit]
    subselection.loc[:,'teff_dis'] = dispersion_array[dispersion_array < dispersion_limit]
    return subselection
