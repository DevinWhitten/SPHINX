import pandas as pd
import numpy as np
import param
import sys,os
##### This script is a test of the metallicity procedures

sys.path.append("/Users/MasterD/Google Drive/JPLUS/Pipeline3.0/Temperature/interface")
import temperature_functions, train_fns, net_functions, network_array
from train_fns import span_window

################################################################################
span_window()
print("  Main_v2.0.py  ")

################################################################################
### read target file
print("Target set:  ", param.params['target_path'])

target = train_fns.Dataset(path=param.params['target_path'], variable='FEH', mode="TARGET")

#### Process target catalog
target.remove_duplicates()
target.remove_discrepant_variables(threshold=0.2)
target.SNR_threshold(20)
target.format_names()
target.faint_bright_limit()
target.format_colors()

#target.gen_scale_frame("self")
#target.scale_photometry()


################################################################################
### read training files

training_FEH = train_fns.Dataset(path=param.params['idr_segue_dr10_path'], variable="FEH", mode="IDR_SEGUE")

################################################################################
span_window()

################################################################################
target.gen_scale_frame("self")

#### Define training set
training_FEH.process( scale_frame = target.scale_frame, threshold=0.08, SNR_limit=40, normal_columns=None,
                      set_bounds = True, bin_number=20, bin_size=200,
                      verbose=True, show_plot=True)


#### target should be scaled differently for TEFF and FEH...

target.scale_photometry()

print("----------------------------------")
for element in target.colors:
    print(element)
    print("MEAN: ", "%.2f" % np.mean(target.custom[element]), "%.2f" % np.mean(training_FEH.custom[element]))
    print("STD:  ", "%.2f" %  np.std(target.custom[element]), "%.2f" % np.std(training_FEH.custom[element]))


print("Final FEH Training Length:  ", training_FEH.get_length())

span_window()
#training_FEH.get_input_stats()


################################################################################
##### Network section
################################################################################
##### Part One: Individual Network
print("... Singular network")
singular_inputs = ['F395', 'F515', 'F410_F515', 'F395_F410', 'F515_F861', 'gSDSS_rSDSS']
FEH_network = net_functions.Network(target_variable="FEH", hidden_layer=6, inputs=singular_inputs,
                                    act_fct="tanh", training_set = training_FEH.custom, scale_frame=training_FEH.scale_frame)
FEH_network.train()
#target.custom.loc[:, "NET_FEH"] = train_fns.unscale(FEH_network.predict(input_frame = target.custom), *training_FEH.scale_frame['FEH'])




##### Part Two: Network Array
print("... Assemble FEH network array")
FEH_array = network_array.Network_Array(training_FEH, interp_frame = None, target_variable = "FEH",
                                      scale_frame = training_FEH.scale_frame, input_type="both",
                                      array_size=50)

FEH_array.set_input_type()
FEH_array.generate(assert_band=["F395"])
FEH_array.train()
FEH_array.info()
FEH_array.eval_performance()
span_window()
FEH_array.prediction(target)
FEH_array.predict_all_networks(target)
FEH_array.write_training_results()

#### Test prediction column add

target.save(filename="FEH_singular_testing.csv")
