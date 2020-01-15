#################################################################
# Author: Devin D Whitten
# Date:   October 3, 2018
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
import dataset, train_fns, net_functions, network_array, io_functions

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
print("... loading networks")
TEFF_NET = io_functions.load_network_state(params, "TEFF_NET")
FEH_NET  = io_functions.load_network_state(params, "FEH_NET" )
AC_NET   = io_functions.load_network_state(params, "AC_NET"  )
################################################################################

print("... formating target set")
target.format_names()
target.set_error_bands()
target.error_reject(training = False)
target.build_colors()

io_functions.span_window()
print("... running networks")

TEFF_NET.predict(target)
AC_NET.predict(target)
FEH_NET.predict(target)


target.merge_master()
target.save(params['output_filename'])





################################################################################


################################################################################
#target.gen_scale_frame("self", method="gauss")

#### Define training set

#### target should be scaled differently for TEFF and FEH...

#target.set_scale_frame('self')
#target.scale_photometry()
#target.get_input_stats(inputs="colors")



#io_functions.span_window()

################################################################################
##### Network section
################################################################################


################################################################################
