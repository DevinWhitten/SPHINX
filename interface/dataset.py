### Author: Devin Whitten
### Date: May, 2019

### This is a streamlined version of train_fns.py
### Hoping to remove clutter

##################################################################

import pandas as pd
import numpy as np
import os, sys
sys.path.append("interface")
from io_functions import span_window
import itertools
from scipy.optimize import curve_fit, minimize
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.ion()


###################################################################


class Dataset():

    def __init__(self, path, variable, params,
                 mode="SEGUE", scale_frame=pd.DataFrame(), interp_frame=pd.DataFrame()):

        ### path should be set from the param file
        ### variable dictates the behavior of many member functions

        print("... Reading database:  ", path)

        try:
            self.master = pd.read_csv(path)
        except:
            print("... catalog was compressed.")
            self.master = pd.read_csv(path, compression="gzip")

        # generate ID for postprocessing remerge
        self.master.loc[:, "SPHINX_ID"] = np.arange(0, len(self.master), 1)

        ### this is the version of database that we will modify throughout
        self.custom = self.master.copy(deep=True)

        self.variable = variable
        self.mode = mode

        self.scale_frame = scale_frame
        self.interp_frame = interp_frame
        self.params = params

        self.colors = []

        self.set_error_bands()

    def set_error_bands(self):
        if self.mode == 'SEGUE':
            self.error_bands = self.params['segue_sigma']

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
        print("...format_names()")

        if self.mode == "SEGUE":
            #df.rename(columns=dict(zip(old_names, new_names)), inplace=True)
            self.custom.rename(columns=dict(zip(self.params['segue_bands'], self.params['format_bands'])), inplace=True)
            self.custom.rename(columns={"TEFF_ADOP": "TEFF", "TEFF_ADOP_ERR": "TEFF_ERR"}, inplace=True)
            self.custom.rename(columns={"FEH_BIW": "FEH", "FEH_BIW_ERR":"FEH_ERR"}, inplace=True)


        elif self.mode == "TARGET": ### For use of Dataset with the target list
            print("Replacing:  ", self.params['target_bands'])
            print("With:       ", self.params['format_bands'])
            self.custom.rename(columns=dict(zip(self.params['target_bands'], self.params['format_bands'])), inplace=True)

        else:
            raise Exception("mode {} not yet implemented in format_names")


    def SNR_threshold(self, SNR_limit = 30):
        ### Remove sources with SNR below the SNR_limit from self.custom
        ### This will probably greatly improve training
        print("... SNR_threshold:  ", SNR_limit)
        original_length = len(self.custom)
        self.custom = self.custom[self.custom['SNR'] > SNR_limit]
        print("\tStars removed:  ", original_length - len(self.custom))

    def EBV_threshold(self, EBV_limit):
        ## Remove stars above the EBV limit in param.params
        print("... EBV_threshold:  ", EBV_limit)
        original_length = len(self.custom)
        self.custom[self.custom['EBV_SFD'] < EBV_limit]
        print("\tStars removed:  ", original_length - len(self.custom))

    def faint_bright_limit(self):
        span_window()
        print(".... faint_bright_limit:  ", self.params['mag_bright_lim'], self.params['mag_faint_lim'],)
        #print("custom columns:     ", self.custom.columns)
        for band in self.params['format_bands']:
            #print(self.custom)
            #print(self.custom[self.custom[band].between(self.params['mag_bright_lim'], self.params['mag_faint_lim'],  inclusive=True)])
            print("\tminimum in:", band, min(self.custom[band]))
            self.custom = self.custom[np.isfinite(self.custom[band])]
            self.custom = self.custom[self.custom[band].between(self.params['mag_bright_lim'], self.params['mag_faint_lim'],  inclusive=True)]
            print("\tCurrent length after:", band, len(self.custom))

    def error_reject(self, training=False):
        span_window()
        print("...error_reject()")
        #### Reject observations above the input error threshold
        print("\tRejection with max err:  ", self.params['mag_err_max'])
        original = len(self.custom)
        for band in self.error_bands:
            self.custom = self.custom[self.custom[band] < self.params['mag_err_max']]

        if training:
            if self.variable == "TEFF":
                self.custom = self.custom[self.custom['TEFF_ERR'] < self.params['T_ERR_MAX']]

            elif self.variable == "FEH":
                self.custom = self.custom[self.custom['FEH_ERR'] < self.params['FEH_ERR_MAX']]

            elif self.variable == 'CFE':
                self.custom = self.custom[self.custom['CFE_ERR'] < self.params['CFE_ERR_MAX']]

            else:
                raise Exception("Could not find variable listed:  {}".format(self.variable))

        print("\tRejected:   ", original - len(self.custom))

    def set_variable_bounds(self, run=False):
        print("...set_variable_bounds()")

        if run == True:

            if self.variable == 'TEFF':
                self.custom = self.custom[self.custom["TEFF"].between(self.params['TMIN'], self.params['TMAX'], inclusive=True)]

            elif self.variable == 'FEH':
                self.custom = self.custom[self.custom["FEH"].between(self.params['FEH_MIN'], self.params['FEH_MAX'], inclusive=True)]

            elif self.variable == 'CFE':
                self.custom = self.custom[self.custom["CFE"].between(self.params['CFE_MIN'], self.params['CFE_MAX'], inclusive=True)]

            else:
                raise Exception("Could not find variable listed:  {}".format(self.variable))


    def build_colors(self):
        ### generate color combinations corresponding to each of the params['format_bands']
        print("...build_colors()")
        color_combinations = list(itertools.combinations(self.params['format_bands'], 2))
        self.colors = []
        for each in color_combinations:
            self.custom.loc[:, each[0] + "_" + each[1]] = self.custom[each[0]] - self.custom[each[1]]
            self.colors.append(each[0] + "_" + each[1])
        return





