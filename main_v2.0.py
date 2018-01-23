import pandas as pd
import numpy as np
import param
import sys,os
##### This is the main driver of the JPLUS_temperature_network, implementing
##### train_fns.Training_Set()

sys.path.append("/Users/MasterD/Google Drive/JPLUS/Pipeline3.0/Temperature/interface")
import temperature_functions, train_fns, net_functions
from train_fns import span_window

################################################################################
span_window()
print("  Main_v2.0.py  ")

### read target file
print("Target set:  ", param.params['target_path'])

target = train_fns.Dataset(path=param.params['target_path'], mode="TARGET")
training = train_fns.Dataset(path=param.params['segue_path'])

span_window()

################################################################################
#### Process target catalog
target.format_names()
target.faint_bright_limit()
scale_frame  = target.gen_scale_frame("self") ### sets target's scale frame using its own inputs

################################################################################

################################################################################
#### Define training set
#scale_frame = temperature_functions.set_scale_frame(target, param.params['format_bands'])
training.process(scale_frame = target.scale_frame, normal_columns=param.params['format_bands'],
                verbose=False, show_plot=False)


print("Final Length:  ", training.get_length())

span_window()

training.get_input_stats()
training.save()

################################################################################
##### Network section
################################################################################

Network = net_functions.Network(training_set = training.custom,
                                scale_frame = scale_frame)
