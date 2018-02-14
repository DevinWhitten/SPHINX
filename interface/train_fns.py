### Author: Devin Whitten
### Date: 1/5/18

### This script serves as the interface for creating training sets
### from photometric catalogs.

import pandas as pd
import numpy as np
##### Interface for main training set generation procedure
import os, sys
sys.path.append("interface")
import temperature_functions
import param_IDR as param
import itertools
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.ion()
def span_window():
    print("-" * os.get_terminal_size().columns)
    return

def MAD(input_vector):
    return np.median(np.abs(input_vector - np.median(input_vector)))

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

##### Master Class for the Training set, I need to solve to the problem of an overcomplicated main.py


class Dataset():
    ### Trying to solve the problem of overcomplicating the main.py with training functions

    def __init__(self,path, variable, mode="SEGUE", params = param.params,
                scale_frame=pd.DataFrame(), interp_frame=pd.DataFrame()):
        ### path should be set from param file
        ### variable: "TEFF" or "FEH" dictates the behavior of training
        ### base set from which we process everything
        print("... Reading database:  ", path)
        self.master = pd.read_csv(path)
        # generate ID for postprocessing remerge
        self.master.loc[:, "SPHINX_ID"] = np.arange(0, len(self.master), 1)

        self.variable=variable

        ### this is the version of master that we will modify
        self.custom = self.master.copy(deep=True)

        self.mode = mode

        self.scale_frame = scale_frame

        self.interp_frame = interp_frame

        self.params = params

        self.colors = []

        if mode == 'SEGUE':
            self.error_bands = params['segue_sigma']

        elif mode == "IDR_SEGUE":
            self.error_bands = params['idr_segue_sigma']

        elif mode == 'TARGET':  ### For use of Dataset with the target list
            self.error_bands = params['target_sigma']

        else:
            print("I haven't implemented that set yet.")

        print("Initial Network Size of: ", self.mode, len(self.custom))

    ############################################################################
    def remove_duplicates(self):
        #### intended to trim target set of duplicate names.
        #### should be run before format names
        for name in param.params['format_bands']:
            try:
                print(name, "deleted")
                del self.custom[name]
            except:
                pass

    def remove_discrepant_variables(self, threshold):
        ### Remove stars whose ADOP and BIW estimates differ more than threshold
        print("remove_discrepant_variables():  ", threshold)

        if self.variable == "FEH":
            self.custom = self.custom[np.abs(self.custom['FEH_ADOP'] - self.custom['FEH_BIW']) < threshold]

        elif self.variable == "TEFF":
            print("I've not implemented this feature")

    def SNR_threshold(self, SNR_limit = 30):
        ### Remove sources with SNR below the SNR_limit from self.custom
        ### This will probably greatly improve training
        print("SNR_threshold")
        original_length = len(self.custom)
        self.custom = self.custom[self.custom['SNR'] > SNR_limit]
        print("Stars removed:  ", original_length - len(self.custom))

    def EBV_threshold(self, EBV_limit):
        ## Remove stars above the EBV limit in param.params
        print("EBV_threshold")
        original_length = len(self.custom)
        self.custom[self.custom['EBV_SFD'] < EBV_limit]
        print("Stars removed:  ", original_length - len(self.custom))

    def format_names(self):
        span_window()
        print("format_names()")

        if self.mode == "SEGUE":
            #df.rename(columns=dict(zip(old_names, new_names)), inplace=True)
            self.custom.rename(columns=dict(zip(self.params['segue_bands'], self.params['format_bands'])), inplace=True)
            self.custom.rename(columns={"TEFF_ADOP": "TEFF", "TEFF_ADOP_ERR": "TEFF_ERR"}, inplace=True)
            self.custom.rename(columns={"FEH_BIW": "FEH", "FEH_BIW_ERR":"FEH_ERR"}, inplace=True)

        elif self.mode == "IDR_SEGUE":
            self.custom.rename(columns=dict(zip(self.params['idr_segue_bands'], self.params['format_bands'])), inplace=True)
            self.custom.rename(columns={"TEFF_ADOP": "TEFF", "TEFF_ADOP_ERR": "TEFF_ERR"}, inplace=True)
            self.custom.rename(columns={"FEH_BIW": "FEH", "FEH_BIW_ERR": "FEH_ERR"}, inplace=True)

        elif self.mode == "TARGET": ### For use of Dataset with the target list
            self.custom.rename(columns=dict(zip(self.params['target_bands'], self.params['format_bands'])), inplace=True)

        else:
            print("I haven't implemented that catalog yet")


    def faint_bright_limit(self):
        span_window()
        print("faint_bright_limit()")

        for band in self.params['format_bands']:
            #print(self.custom)
            #print(self.custom[self.custom[band].between(self.params['mag_bright_lim'], self.params['mag_faint_lim'],  inclusive=True)])
            print("minimum in:", band, min(self.custom[band]))
            self.custom = self.custom[np.isfinite(self.custom[band])]
            self.custom = self.custom[self.custom[band].between(self.params['mag_bright_lim'], self.params['mag_faint_lim'],  inclusive=True)]
            print("Current length after:", band, len(self.custom))




    def error_reject(self):
        span_window()
        print("error_reject()")
        #### Reject observations above the input error threshold
        for band in self.error_bands:
            self.custom = self.custom[self.custom[band] < self.params['mag_err_max']]

        if self.variable == "TEFF":
            self.custom = self.custom[self.custom['TEFF_ERR'] < self.params['T_ERR_MAX']]

        elif self.variable == "FEH":
            self.custom = self.custom[self.custom['FEH_ERR'] < self.params['FEH_ERR_MAX']]


    def format_colors(self):
        ### generate color combinations corresponding to each of the params['format_bands']
        print("format_colors()")
        color_combinations = list(itertools.combinations(param.params['format_bands'], 2))
        self.colors = []
        for each in color_combinations:
            self.custom.loc[:, each[0] + "_" + each[1]] = self.custom[each[0]] - self.custom[each[1]]
            self.colors.append(each[0] + "_" + each[1])
        return

    def set_bounds(self, run=False):
        if run == True:
            self.custom = self.custom[self.custom["TEFF"].between(self.params['TMIN'], self.params['TMAX'], inclusive=True)]
            self.custom = self.custom[self.custom["FEH"].between(self.params['FEH_MIN'], self.params['FEH_MAX'], inclusive=True)]


    ### Generators
    def gen_scale_frame(self, input_frame, method="median"):
        ### method: "gauss", "median"
        ### "median" makes use of the fscale and (max - min)/2.0 for center
        ### "gauss" is what we're used to

        span_window()
        print("gen_scale_frame()")
        ### Generate scale_frame from input_frame and inputs
        calibration = pd.DataFrame()

        if input_frame == "self":
            input_frame = self.custom

        ########################################################################
        if method == "gauss":
            for band in self.params['format_bands'] + self.colors:
                popt, pcov = gaussian_sigma(input_frame[band])
                calibration.loc[:, band] = [popt[1], popt[2]]

        elif method == "median":
            for band in self.params['format_bands'] + self.colors:
                p_min, p_max = np.percentile(input_frame[band], 5), np.percentile(input_frame[band], 95)

                calibration.loc[:, band] = [(p_min + p_min)/2.0, np.abs(p_max - p_min)]

        else:
            print("I haven't implemented that scaling method")

        self.scale_frame = calibration
        return self.scale_frame


    def gen_interp_frame(self, input_frame):
        ### Create interpolation frame, likely based on the target set.
        ### Need to be aware of the state of inputs, whether they are normalized or not
        span_window()
        print("gen_interp_frame()")
        calibration = pd.DataFrame()

        if input_frame == "self":
            input_frame = self.custom

        for band in self.params['format_bands'] + self.colors:
            calibration.loc[:, band] = [np.percentile(input_frame[band], 1), np.percentile(input_frame[band], 99)]

        self.interp_frame = calibration

        return calibration

    def outlier_rejection(self, interp_frame):
        ##### INCORRECT IMPLEMENTATION, REVISION NEEDED
        ### Just reject magnitudes outside of limits defined by interp_frame
        ### this only needs to be run on relevant network inputs...
        print("outlier_rejection()")

        for band in self.params['format_bands']:
                self.custom = self.custom[self.custom[band].between(interp_frame[band].iloc[0], interp_frame[band].iloc[1], inclusive=True)]


    def force_normal(self, columns, bins=20, verbose=False, show_plot=True):
        ##### Force a gaussian distribution of the custom set according to input mean and std
        ####   simultaneously maximize the number of stars possible
        ####   default pivot column set to F515 for now.
        ### Precondition:  Must have self.scale_frame defined

        span_window()


        print("force_normal()")
        #print(self.custom.columns)
        for column in columns:

            #print("Pivot Column:  ", column)
            ### remove erroneous
            #print(column, " : prior finite", len(self.custom[column]))

            distro = np.array(self.custom[column][np.isfinite(self.custom[column])])
            hist = np.histogram(distro, bins)
            print(column, " : ", len(distro))

            xedges = hist[1]
            xbins = [0.5 * (hist[1][i] + hist[1][i+1]) for i in range(len(hist[1])-1)]
            ybins = hist[0]
            #print("HERE")
            fixed_mean, fixed_std = self.scale_frame[column].iloc[0], self.scale_frame[column].iloc[1]
            print("Forced Normal mean:  ", fixed_mean)
            print("Forced Normal std:   ", fixed_std)

            ## use interpolation to estimate the initial a value
            linear = interp1d(xbins, ybins)

            def forced_gauss(x, a):
                return a * np.exp(-(x - fixed_mean)**2.0 / (2 * fixed_std**2))

            #error_fun_root = lambda a : (np.power(np.abs(ybins - forced_gauss(xbins, a)), 1./2.)).sum()
            error_fun_root = lambda a : (np.abs(ybins - forced_gauss(xbins, a)) + 0.7*forced_gauss(xbins, a) - ybins).sum()
            x0 = [linear(fixed_mean)]
            root_res = minimize(error_fun_root, x0,
                                method='SLSQP', bounds=[(0, None)])


            BINS = []
            print("a:  ", root_res.x)
            if show_plot==True:
                fig,ax = plt.subplots(1,2, figsize=(10,4))
                ax[0].hist(self.custom[column], bins=bins)
                ax[0].plot(np.linspace(14, 20, 50), forced_gauss(np.linspace(14, 20, 50), root_res.x), linestyle="--")
                ax[0].set_xlabel(column, fontname="Times New Roman")

                ax[1].plot(np.linspace(10, max(ybins),100), [error_fun_root(a) for a in np.linspace(10, max(ybins),100)], color="blue", linestyle="-")
                ax[1].axvline(root_res.x, linestyle="--")
                ax[1].set_ylabel(r"$\sum | y_i - \phi(x_i, a) | ^{1/3}$", fontname="Times New Roman")
                ax[1].set_xlabel("$a$", fontsize=14, fontname="Times New Roman")
                [label.tick_params(direction="in") for label in ax]
                plt.show()
                input()
                plt.close()


            for i in range(len(xedges)-1):
                current = self.custom[self.custom[column].between(xedges[i], xedges[i+1], inclusive=True)]
                shuffle = current.iloc[np.random.permutation(len(current))]
                BINS.append(shuffle.iloc[0:int(forced_gauss(xbins[i], root_res.x))])
                if verbose==True:
                    print("x_cent:      ", xbins[i])
                    print("max(x_cent): ", int(forced_gauss(xbins[i], root_res.x)))
                    print("Obtained:    ", len(BINS[i]))

            selection = pd.concat(BINS)

            self.custom = selection


            popt, pcov = gaussian_sigma(self.custom[column])
            print("Resulting MEAN:  ", popt[1])
            print("Result    STD:   ", popt[2])

            print("---------------------------------------------------------------")

    def scale_photometry(self):
        span_window()
        print("scale_photometry()")

        #### Precondition: We need self.scale_frame to be set
        ### performs scaling from inputs defined in params.format_bands

        working = self.custom.copy(deep=True)  # I don't want to funk with the original frame
        for band in self.params['format_bands'] + self.colors:

            working.loc[:, band] = Linear_Scale(working[band], self.scale_frame[band].iloc[0], self.scale_frame[band].iloc[1])

        self.custom = working

    def scale_variable(self, mean=None, std=None, variable=None, method="median"):
        span_window()
        print("scale_variable()")
        if (mean == None) and (std==None):
            if method == "gauss":
                try:
                    print("scaling", self.variable, "on gaussian")
                    popt, pcov = gaussian_sigma(self.custom[self.variable])
                    self.scale_frame[self.variable] = [popt[1], popt[2]]
                except:
                    print("scaling", self.variable, "on mean/std")
                    self.scale_frame[self.variable] = [np.mean(self.custom[self.variable]), np.std(self.custom[self.variable])]

            elif method == "median":
                p_min, p_max = np.percentile(self.custom[self.variable], 5), np.percentile(self.custom[self.variable], 95)
                self.scale_frame[self.variable] = [(p_min + p_max)/2.0, np.abs(p_max - p_min)]

            #self.scale_frame[self.variable] = [popt[1], popt[2]]
            self.custom.loc[:, self.variable] = Linear_Scale(self.custom[self.variable],
                                                    self.scale_frame[self.variable].iloc[0],
                                                    self.scale_frame[self.variable].iloc[1])

        else:
            print("I have not implemented this yet")

        print(self.variable, " mean: ", self.scale_frame[self.variable].iloc[0])
        print(self.variable, " std:  ", self.scale_frame[self.variable].iloc[1])
        return

    def unscale_frame(self):
        ### this will be an important function for network performance analysis.
        ### until then we'll work with unscale()
        return

    def uniform_sample(self, bin_number=20, size=100):
        ### Might want to revisit the size critera
        ### Just implement a way of maximizing the binsize

        span_window()
        print("... uniform_sample()")
        if self.variable == "TEFF":
            Bounds = np.linspace(self.params["TMIN"], self.params['TMAX'], bin_number)
            variable = "TEFF"

        elif self.variable == "FEH":
            Bounds = np.linspace(self.params["FEH_MIN"], self.params['FEH_MAX'], bin_number)
            variable = "FEH"

        else:
            print("I haven't implemented that variable yet")

        BIN = []

        for i in range(len(Bounds) - 1):
            Current_Chunk = self.custom[(self.custom[variable] > Bounds[i]) & (self.custom[variable] < Bounds[i+1])]
            BIN.append(Current_Chunk.iloc[np.random.permutation(len(Current_Chunk))][0:size])


        reform = pd.concat(BIN)
        self.custom = reform.iloc[np.random.permutation(len(reform))]

        return



    #### Mutators
    def set_scale_frame(self, input_frame):
        self.scale_frame = input_frame

    def set_interp_frame(self, input_frame):
        self.interp_frame = input_frame
    ############################################################################

    def get_input_stats(self, inputs="both"):
        span_window()
        print(self.variable, " input statistics: ")
        if inputs == "magnitudes":
            input_array = self.params['format_bands']

        elif inputs == "colors":
            input_array = self.colors

        elif inputs == "both":
            input_array = self.params['format_bands'] + self.colors

        else:
            print("Error in input specification")

        for band in input_array:
            popt, pcov = gaussian_sigma(self.custom[band])
            print('{:>9}'.format(band), " : ", '%.3f' %popt[1], '%.3f' %popt[2])

    def get_length(self):
        return len(self.custom)


    def process(self, scale_frame, threshold, SNR_limit=25, EBV_limit = param.params['EBV_MAX'], normal_columns=None, set_bounds=False, bin_number=20,
                bin_size =100, verbose=False, show_plot=False):
        #### just run all of the necessary procedures on the training database
        ## normal_columns: columns that subject to force_normal()
        print("... Processing ", self.variable, " training set")
        self.remove_discrepant_variables(threshold)
        self.SNR_threshold(SNR_limit)
        self.EBV_threshold(EBV_limit)
        self.format_names()
        #self.set_scale_frame(scale_frame)
        self.faint_bright_limit()
        self.error_reject()
        self.format_colors()
        self.set_bounds(set_bounds)

        ####### SCALE_FRAME section #######
        if type(scale_frame) == str:  ### We'll have to come back to this
            self.gen_scale_frame(scale_frame)
        else:
            self.set_scale_frame(scale_frame)


        if normal_columns != None : self.force_normal(columns=normal_columns, verbose=verbose, show_plot=show_plot)
        #self.outlier_rejection(interp_frame)

        self.uniform_sample(bin_number = bin_number, size=bin_size)


        self.scale_photometry()
        ###################################

        self.gen_interp_frame("self")

        print("MAX TEFF:  ", max(self.custom["TEFF"]))
        print("MIN TEFF:  ", min(self.custom["TEFF"]))

        print("MAX FEH:  ", max(self.custom["FEH"]))
        print("MIN FEH:  ", min(self.custom["FEH"]))

        self.scale_variable(method="median")
        return

    ############################################################################

    def merge_master(self):
        ### Precondition: Network estimates are completed on self.custom,
        ### Postcondition: Merges the network estimates with self.master using SPHINX_ID
        self.custom = pd.merge(self.master,self.custom[["NET_" + self.variable, "NET_"+self.variable + "_ERR", "NET_ARRAY_FLAG",'SPHINX_ID']], on="SPHINX_ID")

    def save(self, filename=None):
        if filename == None:
            filename = param.params['output_filename']

        print("... Saving training set to ", filename)
        self.custom.to_csv(self.params["output_directory"] + filename, index=False)
