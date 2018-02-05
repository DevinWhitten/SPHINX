import pandas as pd
import numpy as np
##### Interface for main training set generation procedure
import os, sys
sys.path.append("/Users/MasterD/Google Drive/JPLUS/Pipeline3.0/Temperature/interface")
import temperature_functions
import param, itertools
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


def gen_scaled_catalogs(target, scale_frame, interp_frame, params,
                 mode="SEGUE"):
    ### mode: determines training population, {"EDR", "IDR", "SEGUE"}, default="SEGUE"
    ###       we will at some point need to match SEGUE to IDR for more objects
    ### Still open question of how best to handle the definition of scale_frame
    ### for single stars it might be best to define this from the training set,
    ### whereas for large target set we might define differently.
    ###
    print('gen_catalogs()')
    ############################################################################
    if   mode == "SEGUE":   ### Read SEGUE in
        print("mode == SEGUE")
        Batch = pd.read_csv("/Users/MasterD/Google Drive/JPLUS/Pipeline3.0/data/catalogs/SEGUE_calibrated_catalog.csv")

        filter_names = params["segue_bands"]
        error_names =  params['segue_sigma']

    elif mode == "EDR":
        print("mode == EDR")
        Batch = pd.read_csv("/Users/MasterD/Google Drive/JPLUS/Pipeline2.0/data/catalogs/JPLUS_calibrated_catalog.csv")

        ### ILL COME BACK TO THIS
        filter_names  = ['gSDSS_0', 'rSDSS_0', 'iSDSS_0', 'J0395_0', 'J0410_0', 'J0430_0', 'J0515_0', 'J0660_0', 'J0861_0']
        error_names   = ['gSDSS_ERR', 'rSDSS_ERR', 'iSDSS_ERR', 'J0395_ERR', 'J0410_ERR', 'J0430_ERR', 'J0515_ERR', 'J0660_ERR', 'J0861_ERR']
    else:
        print("Error. Give me a training batch I can use.")
        sys.quit()
    ############################################################################

    print("... Training rejection")

    ### temperature rejection

    Cut = Batch[(Batch['TEFF_ADOP'].between(params['TMIN'], params['TMAX'], inclusive=True)) &
                (Batch['TEFF_ADOP_ERR'] < params['T_ERR_MAX'])]

    ### photometric rejection
    print("... faint/bright rejection")
    for band in filter_names:
        Cut = Cut[Cut[band].between(params["mag_bright_lim"], params["mag_faint_lim"], inclusive=True)]

    print("... mag error rejection")
    for band in error_names:
        Cut = Cut[Cut[band] < params["mag_err_max"]]

    print("Training size after photo rejection: ", len(Cut))
    ##### Interpolation before sampling
    ##### We need to approximate the target distributions
    for i,train_band in enumerate(filter_names):
        Cut = Cut[Cut[train_band].between(interp_frame[params['format_bands'][i]][0], interp_frame[params['format_bands'][i]][1], inclusive=True)]

    ############################################################################
    print("Stars in Training:  ", len(Cut))

    Shuffle = Cut.iloc[np.random.permutation(len(Cut))]

    return Shuffle

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


            self.custom = self.custom[np.isfinite(self.custom[band])]
            self.custom = self.custom[self.custom[band].between(self.params['mag_bright_lim'], self.params['mag_faint_lim'],  inclusive=True)]
            print("Current length after ", band, len(self.custom))

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

    def gen_scale_frame(self, input_frame):
        span_window()
        print("gen_scale_frame()")
        ### Generate scale_frame from input_frame and inputs
        calibration = pd.DataFrame()

        if input_frame == "self":
            input_frame = self.custom

        for band in self.params['format_bands'] + self.colors:
            popt, pcov = gaussian_sigma(input_frame[band])
            calibration.loc[:, band] = [popt[1], popt[2]]

        self.scale_frame = calibration
        return self.scale_frame


    def gen_interp_frame(self, input_frame):
        ### Create interpolation frame, likely based on the target set.
        span_window()
        print("gen_interp_frame()")
        calibration = pd.DataFrame()

        if input_frame == "self":
            input_frame = self.custom

        for band in self.params['format_bands']:
            calibration.loc[:, band] = [np.percentile(input_frame[band], 2), np.percentile(input_frame[band], 98)]

        return calibration

    def outlier_rejection(self, interp_frame):
        ##### INCORRECT IMPLEMENTATION, REVISION NEEDED
        ### Just reject magnitudes outside of limits defined by interp_frame
        ### this only needs to be run on relevant network inputs...
        print("outlier_rejection()")

        for band in self.params['format_bands']:
                self.custom = self.custom[self.custom[band].between(interp_frame[band].iloc[0], interp_frame[band].iloc[1], inclusive=True)]


    def force_normal(self, columns, bins=20, verbose=False, show_plot=True):
        span_window()
        print("force_normal()")
        print(self.custom.columns)
        ##### Force a gaussian distribution of the custom set according to input mean and std
        ####   simultaneously maximize the number of stars possible
        ####   default pivot column set to F515 for now.
        for column in columns:
            print("Pivot Column:  ", column)
            ### remove erroneous
            print(column, " : prior finite", len(self.custom[column]))

            distro = np.array(self.custom[column][np.isfinite(self.custom[column])])
            hist = np.histogram(distro, bins)
            print(column, " : ", len(distro))

            xedges = hist[1]
            xbins = [0.5 * (hist[1][i] + hist[1][i+1]) for i in range(len(hist[1])-1)]
            ybins = hist[0]
            print("HERE")
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

    def scale_variable(self, mean=None, std=None, variable=None):
        span_window()
        print("scale_variable()")
        if (mean == None) and (std==None):
            try:
                print("scaling", self.variable, "on gaussian")
                popt, pcov = gaussian_sigma(self.custom[self.variable])
                self.scale_frame[self.variable] = [popt[1], popt[2]]
            except:
                print("scaling", self.variable, "on mean/std")
                self.scale_frame[self.variable] = [np.mean(self.custom[self.variable]), np.std(self.custom[self.variable])]

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


    def process(self, scale_frame, normal_columns=None, set_bounds=False, bin_number=20,
                bin_size =100, verbose=False, show_plot=False):
        #### just run all of the necessary procedures on the training database
        ## normal_columns: columns that subject to force_normal()
        print("... Processing ", self.variable, " training set")
        self.format_names()
        #self.set_scale_frame(scale_frame)
        self.faint_bright_limit()
        self.error_reject()
        self.format_colors()
        self.set_bounds(set_bounds)
        #if normal_columns != None : self.force_normal(columns=normal_columns, verbose=verbose, show_plot=show_plot)
        #self.outlier_rejection(interp_frame)
        self.uniform_sample(bin_number = bin_number, size=bin_size)
        self.gen_scale_frame("self")
        self.scale_photometry()

        print("MAX TEFF:  ", max(self.custom["TEFF"]))
        print("MIN TEFF:  ", min(self.custom["TEFF"]))

        print("MAX FEH:  ", max(self.custom["FEH"]))
        print("MIN FEH:  ", min(self.custom["FEH"]))
        self.scale_variable()
        return self.scale_frame



    def save(self, filename="training.csv"):
        print("... Saving training set to ", filename)
        self.custom.to_csv(self.params["output_directory"] + filename, index=False)

    #### Mutators
    def set_scale_frame(self, input_frame):
        self.scale_frame = input_frame

    def set_interp_frame(self, input_frame):
        self.interp_frame = input_frame
    ############################################################################

    def get_input_stats(self):
        span_window()
        print(self.variable, " input statistics: ")
        for band in self.params['format_bands'] +  self.colors:
            popt, pcov = gaussian_sigma(self.custom[band])
            print('{:>5}'.format(band), " : ", '%.3f' %popt[1], '%.3f' %popt[2])

    def get_length(self):
        return len(self.custom)
