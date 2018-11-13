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
import param_IDR as param
import sys,os

sys.path.append("interface")
import train_fns, net_functions, network_array, io_functions
################################################################################
io_functions.span_window()
io_functions.print_init()


################################################################################
### read target file
print("Target set:  ", param.params['target_path'])

target = train_fns.Dataset(path=param.params['target_path'], variable='FEH',
                           params=param.params, mode="TARGET")

#### Process target catalog
print("1:  ", len(target.custom))
target.remove_duplicates()
#target.remove_discrepant_variables(threshold=0.15)
#target.SNR_threshold(20)

print("2:  ", len(target.custom))
target.format_names()

#target.faint_bright_limit()
print("3:  ", len(target.custom))

#target.error_reject()
target.format_colors()

#target.gen_scale_frame("self")
#target.scale_photometry()
#target.get_input_stats(inputs='colors')

################################################################################
### read training file

training_FEH = train_fns.Dataset(path=param.params['segue_path'], variable="FEH",
                                 params=param.params, mode="SEGUE")

################################################################################
io_functions.span_window()

################################################################################
#target.gen_scale_frame("self", method="gauss")

#### Define training set

training_FEH.process(scale_frame = "self", threshold=0.20,
                     SNR_limit=30, normal_columns=None,
                     set_bounds = True, bin_number=25, bin_size=200,
                     verbose=True, show_plot=True)


#### target should be scaled differently for TEFF and FEH...

target.set_scale_frame(training_FEH.scale_frame)
target.scale_photometry()
target.get_input_stats(inputs="colors")

print("Final FEH Training Length:  ", training_FEH.get_length())

io_functions.span_window()

################################################################################
##### Network section
################################################################################


##### Part One: Individual Network

#singular_inputs = ['F395_F660', 'F515_F660', 'F410_F515', 'F395_F515', 'F515_F861', 'gSDSS_rSDSS']
#FEH_network = net_functions.Network(target_variable="FEH", hidden_layer=6, inputs=singular_inputs,
#                                    act_fct="tanh", training_set = training_FEH.custom,
#                                    scale_frame=training_FEH.scale_frame, interp_frame=training_FEH.interp_frame)
#FEH_network.train()
#target.custom.loc[:, "NET_FEH"] = train_fns.unscale(FEH_network.predict(input_frame = target.custom), *training_FEH.scale_frame['FEH'])
#target.custom.loc[:, "NET_FLAG"] = FEH_network.is_interpolating(target_frame= target.custom, interp_frame=training_FEH.interp_frame)

#target.save(filename="FEH_singular_testing.csv")


##### Part Two: Network Array
print("... Assemble FEH network array")
FEH_array = network_array.Network_Array(training_FEH, interp_frame=training_FEH.interp_frame,
                                        target_variable = "FEH", scale_frame = target.scale_frame,
                                        param_file = param,
                                        input_type="colors", input_number = 6, array_size=param.params['array_size'])


FEH_array.set_input_type()
FEH_array.generate_inputs(assert_band=['F395'], assert_colors=['gSDSS_rSDSS'],
                          reject_colors=['F395_F410'])  #assert_band=["F395"]

FEH_array.train(iterations=2)
FEH_array.eval_performance()
FEH_array.write_network_performance()
FEH_array.skim_networks(select=25)
FEH_array.prediction(target, flag_thing = False)
#FEH_array.predict_all_networks(target)
FEH_array.write_training_results()
#FEH_array.training_plots()

target.merge_master(array_size=param.params['array_size'])
target.save()

################################################################################
