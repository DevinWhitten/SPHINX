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
#import param_SPLUS82 as param
import sys,os

sys.path.append("../interface")
import dataset, net_functions, network_array
import io_functions

#################################################################
io_functions.intro()
### READ TRAINING SET
params = eval(open("../params/param_FEH.py", 'r').read())

TEFF_train = dataset.Dataset(path = params['segue_path'],
                               variable='FEH',
                               params = params,
                               mode='TRAINING')


TEFF_train.format_names()
TEFF_train.SNR_threshold(25)
TEFF_train.faint_bright_limit()
TEFF_train.error_reject(training=True)
TEFF_train.build_colors()
TEFF_train.set_variable_bounds(run=True)
TEFF_train.get_input_stats(inputs='colors')

### generate the important stuff

TEFF_train.uniform_kde_sample()

TEFF_train.get_input_stats(inputs='colors')

### Initialize network

FEH_NET = network_array.Network_Array(TEFF_train, target_variable = "FEH",
                                        interp_frame = None,
                                        scale_frame  = None,
                                        params       = params,
                                        input_type   = "colors")

FEH_NET.set_input_type()
FEH_NET.generate_inputs(assert_band   = ['F395'],
                         assert_colors = None,
                         reject_colors = ['F395_F410', 'F410_F430'])
FEH_NET.initialize_networks()
FEH_NET.construct_scale_frame()
FEH_NET.normalize_dataset()
FEH_NET.construct_interp_frame()

FEH_NET.generate_train_valid()

FEH_NET.train(iterations=2, pool=True)
FEH_NET.eval_performance()
FEH_NET.write_network_performance()
FEH_NET.skim_networks(select=params['skim'])
FEH_NET.write_training_results()
FEH_NET.training_plots()
FEH_NET.save_state("FEH_NET")
