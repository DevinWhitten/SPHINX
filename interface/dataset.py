### Author: Devin Whitten
### Date: Feb, 2020

### This is a streamlined version of train_fns.py
### Hoping to remove clutter

##################################################################

import pandas as pd
import numpy as np
import os, sys
sys.path.append("interface")
from io_functions import span_window
import stat_functions
import itertools
from scipy.optimize import curve_fit, minimize
from scipy import integrate
from statsmodels.nonparametric.kde import KDEUnivariate
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pickle as pkl
plt.style.use('ggplot')
plt.ion()


###################################################################


class Dataset():

    def __init__(self, path, variable, params,
                 mode="TRAINING", scale_frame=pd.DataFrame(),
                 interp_frame=pd.DataFrame()):

        ### path should be set from the param file
        ### variable dictates the behavior of many member functions
        self.params = params
        print("... Reading database:  ", path)

        try:
            self.master = pd.read_csv(params['SPHINX_path'] + path)
        except:
            print("... catalog was compressed.")
            self.master = pd.read_csv(params['SPHINX_path'] + path, compression="gzip")

        # generate ID for postprocessing remerge
        self.master.loc[:, "SPHINX_ID"] = np.arange(0, len(self.master), 1)

        ### this is the version of database that we will modify throughout
        self.custom = self.master.copy(deep=True)

        self.variable = variable
        self.mode = mode

        self.scale_frame = scale_frame
        self.interp_frame = interp_frame


        self.colors = []
        print("... setting error bands")
        self.set_error_bands()
        return

    def set_error_bands(self):
        if self.mode == 'TRAINING':
            self.error_bands = self.params['training_sigma']

        elif self.mode == 'TARGET':
            self.error_bands = self.params['target_sigma']

        else:
            raise Exception("current mode not implemented:  {}".format(self.mode))


    def remove_duplicate_names(self):
        #### intended to trim target set of duplicate names.
        #### should be run before format names
        print("... remove_duplicates()")
        for name in self.params['format_bands']:
            try:
                print("\t", name, "deleted")
                del self.custom[name]
            except:
                pass

    def format_names(self, band_s_n=None):
        ### renames the variables in self.custom to the proper format for processing.
        span_window()
        print("... format_names()")

        if self.mode == "TRAINING":

            self.custom.rename(columns=dict(zip(self.params['training_bands'], self.params['format_bands'])), inplace=True)
            self.custom.rename(columns={"TEFF_FINAL": "TEFF", "TEFF_FINAL_ERR": "TEFF_ERR"}, inplace=True)
            self.custom.rename(columns={"FEH_FINAL": "FEH", "FEH_FINAL_ERR":"FEH_ERR"}, inplace=True)
            self.custom.rename(columns={"CFE_FINAL": "CFE", "CFE_FINAL_ERR": "CFE_ERR"}, inplace=True)
            self.custom.rename(columns={"AC_FINAL": "AC", "AC_FINAL_ERR": "AC_ERR"}, inplace=True)



        elif self.mode == "TARGET": ### For use of Dataset with the target list
            print("Replacing:  ", self.params['target_bands'])
            print("With:       ", self.params['format_bands'])
            self.custom.rename(columns=dict(zip(self.params['target_bands'],
                                                self.params['format_bands'])),
                                                inplace=True)

        else:
            raise Exception("mode {} not yet implemented in format_names")


    def SNR_threshold(self, SNR_limit = 30):
        ### Remove sources with SNR below the SNR_limit from self.custom
        ### This will probably greatly improve training
        print("... SNR_threshold:  ", SNR_limit)
        original_length = len(self.custom)
        self.custom = self.custom[self.custom['SN'] > SNR_limit]
        print("\tStars removed:  ", original_length - len(self.custom))

    def EBV_threshold(self, EBV_limit):
        ## Remove stars above the EBV limit in param.params
        print("... EBV_threshold:  ", EBV_limit)
        original_length = len(self.custom)
        self.custom[self.custom['EBV_SFD'] < EBV_limit]
        print("\tStars removed:  ", original_length - len(self.custom))

    def faint_bright_limit(self):
        span_window()
        print(".... faint_bright_limit:  ", self.params['mag_bright_lim'], "<-->", self.params['mag_faint_lim'],)
        #print("custom columns:     ", self.custom.columns)
        for band in self.params['format_bands']:

            #print(self.custom[self.custom[band].between(self.params['mag_bright_lim'], self.params['mag_faint_lim'],  inclusive=True)])
            #print("\tminimum in:", band, min(self.custom[band]))
            self.custom = self.custom[np.isfinite(self.custom[band])]
            self.custom = self.custom[self.custom[band].between(self.params['mag_bright_lim'], self.params['mag_faint_lim'],  inclusive=True)]
            print("\tCurrent length after:", band, len(self.custom))

    def error_reject(self, training=True):
        span_window()
        print("... error_reject()")
        #### Reject observations above the input error threshold
        print("\t Rejection with max err:  ", self.params['mag_err_max'])
        original = len(self.custom)
        for band in self.error_bands:
            self.custom = self.custom[self.custom[band] < self.params['mag_err_max']]

        if training:
            print('\t Rejecting on input variable:  ', self.variable)
            if self.variable == "TEFF":
                self.custom = self.custom[self.custom['TEFF_ERR'] < self.params['TEFF_ERR_MAX']]

            elif self.variable == "FEH":
                self.custom = self.custom[self.custom['FEH_ERR'] < self.params['FEH_ERR_MAX']]

            elif self.variable == 'CFE':
                self.custom = self.custom[self.custom['CFE_ERR'] < self.params['CFE_ERR_MAX']]

            elif self.variable == 'AC':
                self.custom = self.custom[self.custom['AC_ERR'] < self.params['AC_ERR_MAX']]

            else:
                raise Exception("Could not find variable listed:  {}".format(self.variable))

        print("\t Rejected:   ", original - len(self.custom))

    def set_variable_bounds(self, run=False, soften=0.2, lowpass=False):
        print("...set_variable_bounds()")

        ## set run to False if you want this function to do nothing..
        if run == True:
            var_range  = self.params[self.variable + "_MAX"] - self.params[self.variable + "_MIN"]
            var_center = (self.params[self.variable + "_MAX"] + self.params[self.variable + "_MIN"]) * 0.5

            left  = var_center - (0.5 * var_range * (1 + soften))
            right = var_center + (0.5 * var_range * (1 + soften))

            print('\t softening variable:  ', self.variable, left, right)


            if lowpass == True:
                print("\t implementing lowpass on input variable")
                self.custom = self.custom[self.custom[self.variable] < right]
            else:
                self.custom = self.custom[self.custom[self.variable].between(left, right, inclusive=True)]

        return



    def specify_variable_bounds(self, variable, bounds):
        ### Quick fix to hard cut the input variable.
        print("\t specify_variable_bounds()  ", variable)
        self.custom = self.custom[self.custom[variable].between(*bounds, inclusive=True)].copy()

        return

    def build_colors(self):
        ### generate color combinations corresponding to each of the params['format_bands']
        print("...build_colors()")
        color_combinations = list(itertools.combinations(self.params['format_bands'], 2))
        self.colors = []
        for each in color_combinations:
            self.custom.loc[:, each[0] + "_" + each[1]] = self.custom[each[0]] - self.custom[each[1]]
            self.colors.append(each[0] + "_" + each[1])
        return


    def gen_scale_frame(self, method="norm"):
        ### method: "gauss", "median"
        ### "median" makes use of the fscale and (max - min)/2.0 for center
        ### "gauss" is what we're used to

        span_window()
        print("... gen_scale_frame()")
        print("\t method:  ", method)
        ### Generate scale_frame from input_frame and inputs
        calibration = pd.DataFrame()


        input_frame = self.custom

        ########################################################################
        if method == "norm":
            for band in self.params['format_bands'] + self.colors:
                #popt, pcov = stat_functions.gaussian_sigma(input_frame[band])
                median = np.median(input_frame[band])
                smad = stat_functions.S_MAD(input_frame[band])
                std  = np.std(input_frame[band])

                if (smad/std > 1.25) or (smad/std < 0.75):
                    print("\t warning: scale discrepancy in band:  ", band)

                ### load the location and scale
                calibration.loc[:, band] = [median, smad]

        elif method == "median":
            for band in self.params['format_bands'] + self.colors:
                p_min, p_max = np.percentile(input_frame[band], 5), np.percentile(input_frame[band], 95)

                calibration.loc[:, band] = [(p_min + p_min)/2.0, np.abs(p_max - p_min)]

        else:
            print("I haven't implemented that scaling method")

        self.scale_frame = calibration
        return self.scale_frame


    def set_scale_frame(self, input_frame):
        self.scale_frame = input_frame



    def uniform_kde_sample(self, cut=True, p_scale = 0.07):
        ### updated uniform sample function to
        ### homogenize the distribution of the training variable.
        ### if the distribution is even close to normal, the algorithm shouldn't
        ### penalize edge values, just the rest of the sample.

        print("... uniform_kde_sample")
        print('\t preuniform sample size:  ', len(self.custom))
        variable = self.variable

        if variable == 'TEFF':
            kde_width = 100
        else:
            kde_width = 0.15

        bounds = list([self.params[variable + "_MIN"], self.params[variable + "_MAX"]])


        ### Basics
        var_min, var_max = min(self.custom[variable]), max(self.custom[variable])

        distro = np.array(self.custom[variable])

        ### Handle boundary solution

        lower = var_min - abs(distro - var_min)
        upper = var_max + abs(distro - var_max)
        merge = np.concatenate([lower, upper, distro])

        ### KDE

        KDE_MERGE = KDEUnivariate(merge)
        KDE_MERGE.fit(bw = kde_width)

        #### interp KDE_MERGE for computation speed
        span = np.linspace(var_min, var_max, 100)
        KDE_FUN = interp1d(span, KDE_MERGE.evaluate(span))

        ### Rescale
        full_c = len(distro) / integrate.quad(KDE_MERGE.evaluate, var_min, var_max)[0]

        ### respan, because I don't want to be penalized for low counts outide variable range
        respan = np.linspace(self.params[self.variable + "_MIN"], self.params[self.variable + "_MAX"], 100)

        scale = np.percentile(KDE_MERGE.evaluate(respan)*full_c, p_scale * 100.)

        ### Accept-Reject sampling
        sample = np.random.uniform(0, 1, len(distro)) * KDE_FUN(distro) * full_c
        boo_array = sample < scale

        if cut:
            self.custom = self.custom.iloc[boo_array].copy()
            self.custom = self.custom.iloc[np.random.permutation(len(self.custom))].copy()

            print("\t uniform sample size:  ", len(self.custom))

        else: #just return the uniform sample
            print('\t returning KDE sample')

            return self.custom.iloc[boo_array].copy()


    def supplement_synthetic(self, iterations=2):
        ### this is an important function, basically create mock observations by
        ### sampling within the errors.
        ### mainly to build up low metallicity stars


        print('... supplementing with gaussian variates')
        self.custom.loc[:, 'sampled'] = [False] * len(self.custom)
        ### get that uniform sample, that's what we'll sample
        sample = self.uniform_kde_sample(cut=False)

        sample = sample.iloc[np.random.permutation(len(sample))].copy()

        final = []
        for i in range(iterations):
            resample = sample.copy()
            ## pretty crucial to ensure that the format_bands corresponds to error_bands

            ### MAGNITUDE SAMPLING
            for band, err in zip(self.params['format_bands'], self.error_bands):

                resample.loc[:, band] = np.random.normal(resample.loc[:, band], resample.loc[:, err])

            #### MAIN VARIABLE SAMPLING

            resample.loc[:, self.variable] = np.random.normal(resample.loc[:, self.variable], resample.loc[:, self.variable + "_ERR"])

            resample.loc[:, 'sampled'] = [True] * len(resample)

            final.append(resample)


        ### just dump it back into self.cusom?
        final = pd.concat(final)
        print("\t adding :", len(final), " synthetic stars")
        self.custom = pd.concat([final, self.custom])

        return



    def gen_interp_frame(self, input_frame = 'self'):
        ### Create interpolation frame, likely based on the <training> set.
        ### This will be used to make sure the networks are consistently interpolating from the training data,
        ### or else the observation will be flagged and likely erroneous.
        ### Need to be aware of the state of inputs, whether they are normalized or not

        span_window()
        print("... gen_interp_frame()")
        calibration = pd.DataFrame()

        if input_frame == "self":
            input_frame = self.custom

        for band in self.params['format_bands'] + self.colors:
            calibration.loc[:, band] = [np.percentile(input_frame[band], 1), np.percentile(input_frame[band], 99)]

        self.interp_frame = calibration

        return calibration

    def set_interp_frame(self, input_frame):
        self.interp_frame = input_frame
        return


    def check_interp(self, interp_frame=None):
        ### Basically just check what percentage of the dataset is within the assigned interp frame

        current = np.ones(len(self.custom))
        for column in self.colors:
            current = current * self.custom[column].between(interp_frame[column].iloc[0], interp_frame[column].iloc[1])

        print("percent in interp_frame:   ", float(current.sum())/float(len(current)) * 100.)
        return

    def check_input_match(self, second_frame, lim = 0.15):
        print("check_input_match")
        ### basically just want to quantitatively ensure that the train/target distributions match
        ### and recommend colors for rejection in the event that they do not.
        ### maybe a KS test?...
        colors = np.array(self.colors.copy())
        match_score = np.array([np.std(self.custom[color]) / np.std(second_frame.custom[color]) for color in self.colors])

        for cur_color, score in zip(colors[np.argsort(match_score)], match_score[np.argsort(match_score)]):
            print("\t", '{:11s}'.format(cur_color), " : ", '%6.3f' %score)


        exclude = colors[abs(match_score - np.ones(len(match_score))) > lim]

        if len(exclude) == 0:
            print("\t no erroneous color distributions found")
            self.error_colors = None
            return

        print("\t recommending the following colors for exclusion:  ")
        print("\t", exclude)
        self.error_colors = exclude

        return





    def scale_photometry(self):
        span_window()
        print("... scale_photometry()")

        #### Precondition: We need self.scale_frame to be set
        #### performs scaling from inputs defined in params.format_bands

        working = self.custom.copy(deep=True)  # I don't want to funk with the original frame
        for band in self.params['format_bands'] + self.colors:

            working.loc[:, band] = stat_functions.linear_scale(working[band],
                                                               self.scale_frame[band].iloc[0],
                                                               self.scale_frame[band].iloc[1])

        self.custom = working


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

        try:
            for band in input_array:
                popt, pcov = stat_functions.gaussian_sigma(self.custom[band])
                print("\t", '{:11s}'.format(band), " : ", '%6.3f' %popt[1], '%6.3f' %popt[2])
        except:
            print(" "*20 + "median -- range")
            for band in input_array:

                print("\t", '{:11s}'.format(band), " : ", '%6.3f' % np.median(self.custom[band]), '%6.3f' % (max(self.custom[band]) - min(self.custom[band])))


    def scale_variable(self, mean=None, std=None, variable=None, method="median"):
        #### FOR TRAINING SETS ONLY
        span_window()
        print("...scale_variable()")
        if (mean == None) and (std==None):
            if method == "gauss":
                try:
                    print("scaling", self.variable, "on gaussian")
                    popt, pcov = stat_functions.gaussian_sigma(self.custom[self.variable])
                    self.scale_frame[self.variable] = [popt[1], popt[2]]
                except:
                    print("scaling", self.variable, "on mean/std")
                    self.scale_frame[self.variable] = [np.mean(self.custom[self.variable]), np.std(self.custom[self.variable])]

            elif method == "median":
                p_min, p_max = np.percentile(self.custom[self.variable], 5), np.percentile(self.custom[self.variable], 95)
                self.scale_frame[self.variable] = [(p_min + p_max)/2.0, np.abs(p_max - p_min)]

            #self.scale_frame[self.variable] = [popt[1], popt[2]]
            self.custom.loc[:, self.variable] = stat_functions.linear_scale(self.custom[self.variable],
                                                    self.scale_frame[self.variable].iloc[0],
                                                    self.scale_frame[self.variable].iloc[1])

        else:
            print("I have not implemented this yet")

        print("\t", self.variable, " mean: ", self.scale_frame[self.variable].iloc[0])
        print("\t", self.variable, " std:  ", self.scale_frame[self.variable].iloc[1])
        return


    def merge_master(self, array_size=0, vars=None):
        ### Precondition: Network estimates are completed on self.custom,
        ### Postcondition: Merges the network estimates with self.master using SPHINX_ID
        #columns = ["NET_" + self.variable, "NET_"+ self.variable + "_ERR", "NET_ARRAY_" + self.variable + "_FLAG",'SPHINX_ID']
        #if which == "all"
        if array_size != 0:
            print("\t merging final parameters")
            columns = ["NET_" + self.variable, "NET_"+ self.variable + "_ERR", "NET_ARRAY_" + self.variable + "_FLAG",'SPHINX_ID'] + self.colors
            #columns = columns + ["NET_" + str(i) + "_" + self.variable for i in range(array_size)]
            #columns = columns + ["NET_" + str(i) + "_" + self.variable + "_FLAG" for i in range(array_size)]
            self.custom = pd.merge(self.master,self.custom[columns], on="SPHINX_ID")

        elif vars != None:
            columns = [["NET_" + variable, "NET_" + variable + "_ERR", "NET_ARRAY_" + variable + "_FLAG"] for variable in vars]
            flat_list = [item for sublist in columns for item in sublist]

            self.custom = pd.merge(self.master, self.custom[flat_list + ["SPHINX_ID"]], on="SPHINX_ID")

        else:
            print("\t sorry about the duplicate columns")
            self.custom = pd.merge(self.master,self.custom, on="SPHINX_ID")



    def save(self, filename=None):
        print("FILENAME:  ", filename)
        if filename == None:
            filename = self.params['output_filename']

        print("... Saving training set to ", filename)
        self.custom.to_csv(self.params['SPHINX_path'] + self.params["output_directory"] + filename, index=False)


################################################################################
### Here's a crucial dataset function for final estimates

def merge_datasets(COOL, HOT, TCRIT = [5500, 5750], ID_TAG = "SPH_IND"):
    print("... merging datasets")
    print("\t presizes:  LEFT:  ", len(COOL), "  RIGHT:    ", len(HOT))
    print("\t Teff boundary for join:  ", TCRIT)

    ## Tries to merge the FEH/AC estimate in the transition region
    LEFT = COOL[COOL['NET_TEFF'] < TCRIT[0]].copy()
    RIGHT = HOT[HOT['NET_TEFF'] > TCRIT[1]].copy()

    MID_LEFT = COOL[COOL['NET_TEFF'].between(*TCRIT, inclusive=True)].copy()
    MID_RIGHT = HOT[HOT['NET_TEFF'].between(*TCRIT, inclusive=True)].copy()

    ### want to join the mid, while maximizing stars with estimates

    ID = pd.merge(MID_LEFT, MID_RIGHT, on=ID_TAG)[ID_TAG]

    ### save the observations unique to each side

    MID_LEFT_save = MID_LEFT[~MID_LEFT[ID_TAG].isin(ID)].copy()
    MID_RIGHT_save = MID_RIGHT[~MID_RIGHT[ID_TAG].isin(ID)].copy()

    ### now prepare the common stars
    MID_LEFT_COMMON = MID_LEFT[MID_LEFT[ID_TAG].isin(ID)].copy()
    MID_RIGHT_COMMON = MID_RIGHT[MID_RIGHT[ID_TAG].isin(ID)].copy()

    print(len(MID_LEFT_COMMON), len(MID_RIGHT_COMMON))

    ### Rename columns

    MID_LEFT_COMMON = MID_LEFT_COMMON.rename(columns={"NET_FEH": "NET_FEH_LEFT", "NET_FEH_ERR": "NET_FEH_ERR_LEFT",
                                                      "NET_AC": "NET_AC_LEFT", "NET_AC_ERR": "NET_AC_ERR_LEFT",
                                                      "NET_ARRAY_FEH_FLAG": 'NET_ARRAY_FEH_FLAG_LEFT',
                                                      "NET_ARRAY_AC_FLAG" : 'NET_ARRAY_AC_FLAG_LEFT'},
                                                      errors="raise")

    MID_RIGHT_COMMON = MID_RIGHT_COMMON.rename(columns={"NET_FEH": "NET_FEH_RIGHT", "NET_FEH_ERR": "NET_FEH_ERR_RIGHT",
                                                        "NET_AC": "NET_AC_RIGHT", "NET_AC_ERR": "NET_AC_ERR_RIGHT",
                                                        "NET_ARRAY_FEH_FLAG": 'NET_ARRAY_FEH_FLAG_RIGHT',
                                                        "NET_ARRAY_AC_FLAG" : 'NET_ARRAY_AC_FLAG_RIGHT'},
                                                        errors="raise")

    ### Now merge the estimates
    COMBO = pd.merge(MID_LEFT_COMMON[['NET_FEH_LEFT', 'NET_FEH_ERR_LEFT', 'NET_AC_LEFT', 'NET_AC_ERR_LEFT',
                                      'NET_ARRAY_FEH_FLAG_LEFT', 'NET_ARRAY_AC_FLAG_LEFT', ID_TAG]],
                     MID_RIGHT_COMMON[['NET_FEH_RIGHT', 'NET_FEH_ERR_RIGHT', 'NET_AC_RIGHT', 'NET_AC_ERR_RIGHT',
                                      'NET_ARRAY_FEH_FLAG_RIGHT', 'NET_ARRAY_AC_FLAG_RIGHT', ID_TAG]],
                                      on=ID_TAG)
    print('COMBO:   ', len(COMBO))
    FEH_FLAG = []
    FEH_VALUE = []
    FEH_ERR_VALUE = []
    for i, row in COMBO.iterrows():
        if np.isfinite(row['NET_FEH_LEFT']) & np.isfinite(row['NET_FEH_RIGHT']) & \
           np.isfinite(row['NET_FEH_ERR_LEFT']) & np.isfinite(row['NET_FEH_ERR_RIGHT']) & \
           (row['NET_ARRAY_FEH_FLAG_LEFT'] > 1) & (row['NET_ARRAY_FEH_FLAG_RIGHT'] > 1):
        ### then do a nice weighted average

            #FEH_VALUE.append(np.average([row['NET_FEH_LEFT'], row['NET_FEH_RIGHT']],
            #                        weights=[1./row['NET_FEH_ERR_LEFT'], 1./row['NET_FEH_ERR_RIGHT']]))

            FEH_ERR_VALUE.append(max([row['NET_FEH_ERR_LEFT'], row['NET_FEH_ERR_RIGHT']]))
            FEH_FLAG.append(min([row['NET_ARRAY_FEH_FLAG_LEFT'], row['NET_ARRAY_FEH_FLAG_RIGHT']]))


        ### then do a nice weighted average
            try:
                FEH_VALUE.append(np.average([row['NET_FEH_LEFT'], row['NET_FEH_RIGHT']],
                                    weights=[1./row['NET_FEH_ERR_LEFT'], 1./row['NET_FEH_ERR_RIGHT']]))
            except:
                print("weird", row['NET_FEH_ERR_LEFT'], row['NET_ARRAY_FEH_FLAG_LEFT'], row['NET_FEH_ERR_RIGHT'], row['NET_ARRAY_FEH_FLAG_RIGHT'])
                FEH_VALUE.append(np.average([row['NET_FEH_LEFT'], row['NET_FEH_RIGHT']]))




        ###  then take the one that works
        elif np.isfinite(row['NET_FEH_LEFT']) & np.isfinite(row['NET_FEH_ERR_LEFT']) & (row['NET_ARRAY_FEH_FLAG_LEFT'] > 1):
            FEH_VALUE.append(row['NET_FEH_LEFT'])
            FEH_ERR_VALUE.append(row['NET_FEH_ERR_LEFT'])
            FEH_FLAG.append(row['NET_ARRAY_FEH_FLAG_LEFT'])



        elif np.isfinite(row['NET_FEH_RIGHT']) & np.isfinite(row['NET_FEH_ERR_RIGHT']) & (row['NET_ARRAY_FEH_FLAG_RIGHT'] > 1):
            FEH_VALUE.append(row['NET_FEH_RIGHT'])
            FEH_ERR_VALUE.append(row['NET_FEH_ERR_RIGHT'])
            FEH_FLAG.append(row['NET_ARRAY_FEH_FLAG_RIGHT'])


        else:
            print("Bad row: ", i, end='\r')
            sys.stdout.flush()
            FEH_VALUE.append(np.nan)
            FEH_ERR_VALUE.append(np.nan)
            FEH_FLAG.append(np.nan)


    ### Now let's just repeat with AC
    AC_FLAG = []
    AC_VALUE = []
    AC_ERR_VALUE = []
    for i, row in COMBO.iterrows():
        if np.isfinite(row['NET_AC_LEFT']) & np.isfinite(row['NET_AC_RIGHT']) & \
           np.isfinite(row['NET_AC_ERR_LEFT']) & np.isfinite(row['NET_AC_ERR_RIGHT']) & \
           (row['NET_ARRAY_AC_FLAG_LEFT'] > 1) & (row['NET_ARRAY_AC_FLAG_RIGHT'] > 1):
        ### then do a nice weighted average
            try:
                AC_VALUE.append(np.average([row['NET_AC_LEFT'], row['NET_AC_RIGHT']],
                                    weights=[1./row['NET_AC_ERR_LEFT'], 1./row['NET_AC_ERR_RIGHT']]))
            except:
                print("weird", row['NET_AC_ERR_LEFT'], row['NET_AC_ERR_RIGHT'])
                AC_VALUE.append(np.average([row['NET_AC_LEFT'], row['NET_AC_RIGHT']]))


            AC_ERR_VALUE.append(max([row['NET_AC_ERR_LEFT'], row['NET_AC_ERR_RIGHT']]))
            AC_FLAG.append(min([row['NET_ARRAY_AC_FLAG_LEFT'], row['NET_ARRAY_AC_FLAG_RIGHT']]))

        ###  then take the one that works
        elif np.isfinite(row['NET_AC_LEFT']) & np.isfinite(row['NET_AC_ERR_LEFT']) & (row['NET_ARRAY_AC_FLAG_LEFT'] > 1):
            AC_VALUE.append(row['NET_AC_LEFT'])
            AC_ERR_VALUE.append(row['NET_AC_ERR_LEFT'])
            AC_FLAG.append(row['NET_ARRAY_AC_FLAG_LEFT'])


        elif np.isfinite(row['NET_AC_RIGHT']) & np.isfinite(row['NET_AC_ERR_RIGHT']) & (row['NET_ARRAY_AC_FLAG_RIGHT'] > 1):
            AC_VALUE.append(row['NET_AC_RIGHT'])
            AC_ERR_VALUE.append(row['NET_AC_ERR_RIGHT'])
            AC_FLAG.append(row['NET_ARRAY_AC_FLAG_RIGHT'])


        else:
            print("Bad:  ", i, end='\r')
            sys.stdout.flush()
            AC_VALUE.append(np.nan)
            AC_ERR_VALUE.append(np.nan)
            AC_FLAG.append(np.nan)



    COMBO.loc[:, 'NET_FEH']            = FEH_VALUE
    COMBO.loc[:, 'NET_FEH_ERR']        = FEH_ERR_VALUE
    COMBO.loc[:, 'NET_ARRAY_FEH_FLAG'] = FEH_FLAG

    COMBO.loc[:, 'NET_AC']            = AC_VALUE
    COMBO.loc[:, 'NET_AC_ERR']        = AC_ERR_VALUE
    COMBO.loc[:, 'NET_ARRAY_AC_FLAG'] = AC_FLAG


    for label in ['NET_FEH_LEFT', 'NET_FEH_RIGHT', 'NET_FEH_ERR_LEFT', 'NET_FEH_ERR_RIGHT',
                  'NET_AC_LEFT', 'NET_AC_RIGHT', 'NET_AC_ERR_LEFT', 'NET_AC_ERR_RIGHT',
                  'NET_ARRAY_FEH_FLAG_LEFT', 'NET_ARRAY_FEH_FLAG_RIGHT', 'NET_ARRAY_AC_FLAG_LEFT', 'NET_ARRAY_AC_FLAG_RIGHT']:
        del COMBO[label]

    for frame in [LEFT, RIGHT, MID_LEFT_save, MID_RIGHT_save]:
        frame.loc[:, 'fix_feh'] = np.zeros(len(frame))
        frame.loc[:, 'fix_ac'] = np.zeros(len(frame))


    COMBO.loc[:, 'fix_feh'] = np.ones(len(COMBO))
    COMBO.loc[:, 'fix_ac'] = np.ones(len(COMBO))

    print('COMBO', len(COMBO))

    COMBO = pd.merge(COMBO, MID_LEFT_COMMON, on=ID_TAG)

    return pd.concat([LEFT, RIGHT, MID_LEFT_save, MID_RIGHT_save, COMBO], sort=True)
