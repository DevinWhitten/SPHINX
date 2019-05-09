#################################################################
# Author: Devin D Whitten
# Date:   May, 2018
# This is the main driver of the parameter determinations with SPHINX

# All specifications for running should be made in the input
# parameter file.
# Please direct questions/suggestions to dwhitten@nd.edu
#################################################################

import pandas as pd
import numpy as np
import param_SPLUS82 as param
import sys,os

sys.path.append("interface")
import dataset, net_functions, network_array
import io_functions

#################################################################
io_functions.intro()
### READ TRAINING SET

TEFF_train = dataset.Dataset(path = param.params['segue_path'],
                               variable='TEFF',
                               params = param.params,
                               mode='SEGUE')
TEFF_train.remove_duplicate_names()
TEFF_train.SNR_threshold(25)
TEFF_train.format_names()
TEFF_train.faint_bright_limit()
TEFF_train.error_reject(training=True)
TEFF_train.build_colors()
TEFF_train.set_variable_bounds(run=True)
