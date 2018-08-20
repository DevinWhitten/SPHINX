import pandas as pd
import numpy as np
import param_IDR as param
import sys,os
##### This script is a test of the metallicity procedures

sys.path.append("interface")
import train_fns, net_functions, network_array
from train_fns import span_window

################################################################################
span_window()
print("  Main_v2.0.py  ")

################################################################################
### read target file
print("Target set:  ", param.params['target_path'])

target = train_fns.Dataset(path=param.params['target_path'], variable='FEH', params=param.params, mode="TARGET")

#### Process target catalog
target.remove_duplicates()
#target.remove_discrepant_variables(threshold=0.15)
#target.SNR_threshold(20)
target.format_names()
target.faint_bright_limit()
target.format_colors()
#target.set_bounds(run=True)


#target.gen_scale_frame("self")
#target.scale_photometry()


################################################################################
### read training files

training_FEH = train_fns.Dataset(path=param.params['idr_segue_path'], variable="FEH", params=param.params, mode="IDR_SEGUE")

################################################################################
span_window()

################################################################################
#target.gen_scale_frame("self", method="gauss")

#### Define training set
training_FEH.process(scale_frame = "self", threshold=0.20, SNR_limit=40, normal_columns=None,
                     set_bounds = True, bin_number=25, bin_size=200,
                     verbose=True, show_plot=True)


#### target should be scaled differently for TEFF and FEH...
target.set_scale_frame(training_FEH.scale_frame)
target.scale_photometry()

target.get_input_stats(inputs="colors")

print("Final FEH Training Length:  ", training_FEH.get_length())

span_window()

################################################################################
##### Network section
################################################################################
##### Part One: Individual Network
print("... Singular network")
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
                                        target_variable = "FEH", scale_frame = training_FEH.scale_frame,
                                        param_file = param,
                                        input_type="colors", array_size=10)

FEH_array.set_input_type()
FEH_array.generate_inputs(assert_band=["F395"])  #assert_band=["F395"]
FEH_array.train(iterations=2)
FEH_array.eval_performance()
FEH_array.write_network_performance()

FEH_array.prediction(target)
FEH_array.predict_all_networks(target)
FEH_array.write_training_results()
FEH_array.training_plots()

target.merge_master(array_size=10)
#print(target.custom.columns)
target.save()

#### Test prediction column add

#target.save(filename="FEH_array_testing.csv")
