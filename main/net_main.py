#################################################################
# Author: Devin D Whitten
# Date:   Feb, 2020
# This is the main driver of the [Fe/H] determinations with SPHINX

# All specifications for running should be made in the input
# parameter file.
# Please direct questions/suggestions to dwhitten@nd.edu
# Whitten et al. 2018

#################################################################

import pandas as pd
import numpy as np

import sys,os

sys.path.append("../interface")
import dataset, train_fns, net_functions, network_array, io_functions, master_state

################################################################################
io_functions.span_window()
io_functions.intro()
################################################################################

params = eval(open("../params/param_SPLUS82.py", 'r').read())

print("Target set:  ", params['target_path'])

target = dataset.Dataset(path = params['target_path'],
                         variable=None,
                         params=params,
                         mode="TARGET")


################################################################################
### Load networks
################################################################################
io_functions.span_window()
print("... initializing master state")

MASTER = master_state.Master_State(params = params)

################################################################################

print("... formating target set")
target.format_names()
target.set_error_bands()
target.error_reject(training = False)
target.build_colors()

io_functions.span_window()
print("... running networks")

target = MASTER.predict(target)

target.save(params['output_filename'])


print(" -- COMPLETE -- ")
################################################################################
################################################################################
