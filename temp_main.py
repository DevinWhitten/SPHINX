import pandas as pd
import numpy as np
import param_teff as param
import sys,os

sys.path.append("interface")
import train_fns, net_functions, network_array
from train_fns import span_window
#################################################################
print("  temp_main.py  ")
print(param.params['output_filename'])
print("Target set:  ", param.params['target_path'])
#print("Training set:", param.params['training_set'])
print("Target bands: ", param.params['target_bands'])

target = train_fns.Dataset(path=param.params['target_path'], variable='TEFF', mode="TARGET")


print("---------- TARGET NAMES ----------")
for column in target.custom.columns:
    print(column)


### remove duplicates is probably not necessary right now
#target.remove_duplicates()
target.format_names()

target.faint_bright_limit()
target.format_colors()
#target.gen_scale_frame("self", method="gauss")
#target.scale_photometry()
print("Color Test")
print(target.custom['gSDSS_iSDSS'])


target.get_input_stats(inputs="colors")

print("Reading/Formatting Training Database")

training = train_fns.Dataset(path=param.params['segue_path'], variable="TEFF", mode="SEGUE")
training.remove_duplicates()
training.process(scale_frame = "self", threshold=75, SNR_limit=25, normal_columns=None,
                 set_bounds = True, bin_number=20, bin_size=200,
                 verbose=True, show_plot=True)



target.set_scale_frame(training.scale_frame)
target.scale_photometry()



################################################################################
##### Network section
################################################################################
Network_Array = network_array.Network_Array(training, interp_frame=training.interp_frame, target_variable = "TEFF",
                                            scale_frame = training.scale_frame,
                                            param_file = param,
                                            input_type="colors",
                                            array_size=30)

Network_Array.set_input_type()
Network_Array.generate_inputs(assert_band=["F410"])

# Let's beef this function up
Network_Array.train()

Network_Array.eval_performance()

Network_Array.prediction(target)
Network_Array.predict_all_networks(target)
Network_Array.write_training_results()
Network_Array.training_plots()
target.merge_master(array_size=25)
target.save()
