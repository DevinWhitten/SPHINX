#################################################################
# Author: Devin D Whitten
# Date:   October 3, 2018
# This is the main driver of the Teff determinations with SPHINX

# All specifications for running should be made in the input
# parameter file.
# Please direct questions/suggestions to dwhitten@nd.edu

# Whitten et al. 2018

#################################################################


import pandas as pd
import numpy as np
import param_P0_teff as param
import sys,os

sys.path.append("interface")
import train_fns, net_functions, network_array, io_functions


#################################################################
print("  temp_main.py  ")
print(param.params['output_filename'])
print("Target set:  ", param.params['target_path'])
print("Target bands: ", param.params['target_bands'])

target = train_fns.Dataset(path=param.params['target_path'], variable='TEFF',
                           params=param.params, mode="TARGET")


print("---------- TARGET NAMES ----------")
for column in target.custom.columns:
    print(column)


### remove duplicates is probably not necessary right now
target.remove_duplicates()
target.format_names()
#target.faint_bright_limit()



io_functions.span_window()

target.format_colors()
#target.gen_scale_frame("self", method="gauss")
#target.scale_photometry()

print("Color Test")
print(target.custom['gSDSS_iSDSS'])


target.get_input_stats(inputs="colors")

io_functions.span_window()
print("Reading/Formatting Training Database")

training = train_fns.Dataset(path=param.params['segue_path'], variable="TEFF",
                             params=param.params, mode="SEGUE")
training.remove_duplicates()
training.process(scale_frame = "self", threshold=75, SNR_limit=25, normal_columns=None,
                 set_bounds = True, bin_number=20, bin_size=200,
                 verbose=True, show_plot=True)



target.set_scale_frame(training.scale_frame)
target.scale_photometry()


################################################################################
##### Network section
################################################################################
Network_Array = network_array.Network_Array(training, interp_frame=training.interp_frame,
                                            target_variable = "TEFF",
                                            scale_frame = training.scale_frame,
                                            param_file = param,
                                            input_type="colors",
                                            array_size=param.params['array_size'])



Network_Array.set_input_type()
Network_Array.generate_inputs(assert_band=["F410"], reject_band=['F430'])

# Let's beef this function up
Network_Array.train(iterations=2)
Network_Array.eval_performance()
Network_Array.write_network_performance()
Network_Array.skim_networks(select=25)
Network_Array.prediction(target, flag_thing = False)

Network_Array.write_training_results()
target.merge_master(array_size=param.params['array_size'])

Network_Array.write_training_results()
Network_Array.training_plots()
target.merge_master(array_size=param.params['array_size'])
target.save()
