import pandas as pd
import numpy as np
import param_teff as param
import sys,os

##### This script is a test of the M15 temperature procedures
sys.path.append("interface")
import train_fns, net_functions, network_array
from train_fns import span_window
#################################################################
print("  temp_main.py  ")

print("Target set:  ", param.params['target_path'])

target = train_fns.Dataset(path=param.params['target_path'], variable='TEFF', mode="TARGET")

target.remove_duplicates()
target.format_names()
target.faint_bright_limit()
target.format_colors()
target.gen_scale_frame("self", method="gauss")
target.scale_photometry()
target.get_input_stats(inputs="colors")

training = train_fns.Dataset(path=param.params['idr_segue_dr10_path'], variable="TEFF", mode="IDR_SEGUE")

training.process(scale_frame = target.scale_frame, threshold=75, SNR_limit=35, normal_columns=None,
                 set_bounds = True, bin_number=15, bin_size=200,
                 verbose=True, show_plot=True)



################################################################################
##### Network section
################################################################################
Network_Array = network_array.Network_Array(training, interp_frame=training.interp_frame, target_variable = "TEFF",
                                            scale_frame = training.scale_frame, input_type="colors",
                                            array_size=30)

Network_Array.set_input_type()
Network_Array.generate_inputs()
Network_Array.train()
Network_Array.eval_performance()
Network_Array.prediction(target)
#Network_Array.predict_all_networks(target)
Network_Array.write_training_results()

target.merge_master(array_size=30)
target.save()
