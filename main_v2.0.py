import pandas as pd
import numpy as np
import param
import sys,os
##### This is the main driver of the JPLUS_temperature_network, implementing
##### train_fns.Training_Set()

sys.path.append("/Users/MasterD/Google Drive/JPLUS/Pipeline3.0/Temperature/interface")
import temperature_functions, train_fns, net_functions, network_array
from train_fns import span_window

################################################################################
span_window()
print("  Main_v2.0.py  ")

### read target file
print("Target set:  ", param.params['target_path'])

target = train_fns.Dataset(path=param.params['target_path'], variable='TEFF', mode="TARGET")

#training_segue = train_fns.Dataset(path=param.params['segue_path'], variable="TEFF")
training_TEFF = train_fns.Dataset(path=param.params['idr_segue_path'], variable="TEFF", mode="IDR_SEGUE")
training_FEH = train_fns.Dataset(path=param.params['segue_path'], variable="FEH", mode="SEGUE")


span_window()

################################################################################
#### Process target catalog
target.remove_duplicates()
target.format_names()
target.faint_bright_limit()
target.format_colors()

################################################################################
### Set target frame
#scale_frame  = target.gen_scale_frame("self") ### sets target's scale frame using its own inputs
interp_frame = target.gen_interp_frame("self")


################################################################################
#target.scale_photometry()  ### HOW DOES TARGET HAVE A SCALE FRAME???



#### Define training set
TEFF_scale = training_TEFF.process(scale_frame = target.scale_frame, normal_columns=None,
                    verbose=False, show_plot=False)

FEH_scale  = training_FEH.process( scale_frame = target.scale_frame,
                    normal_columns=["F395"], set_bounds = True, bin_number=20, bin_size=200,
                    verbose=True, show_plot=True)

target.set_scale_frame(FEH_scale)
target.scale_photometry()



print("Final TEFF Training Length:  ", training_TEFF.get_length())
print("Final FEH Training Length:  ", training_FEH.get_length())

span_window()

training_TEFF.get_input_stats()
training_TEFF.save(filename = "teff_training.csv")

training_FEH.get_input_stats()
training_FEH.save(filename = "FEH_training.csv")



################################################################################
##### Network section
################################################################################
print("... Assemble TEFF network array")
TEFF_net = network_array.Network_Array(training_TEFF, interp_frame = interp_frame, target_variable = "TEFF",
                                       scale_frame = TEFF_scale, array_size=15)
TEFF_net.generate()
TEFF_net.train()
#TEFF_net.info()
TEFF_net.eval_performance()
span_window()
target.custom.loc[:, "NET_TEFF"], target.custom.loc[:, "NET_TEFF_ERR"] = TEFF_net.prediction(target)


print("... Assemble FEH network array")
FEH_net = network_array.Network_Array(training_FEH, interp_frame = interp_frame, target_variable = "FEH",
                                      scale_frame = FEH_scale, array_size=25)

FEH_net.generate(assert_band=["F395"])
FEH_net.train()
FEH_net.info()
FEH_net.eval_performance()
span_window()
target.custom.loc[:, "NET_FEH"], target.custom.loc[:, "NET_FEH_ERR"] = FEH_net.prediction(target)



for i, net in enumerate(TEFF_net.network_array):
    target.custom.loc[:, "TEFF_NET_"+str(i)] = train_fns.unscale(net.prediction, *TEFF_scale['TEFF'])

for i, net in enumerate(FEH_net.network_array):
    target.custom.loc[:, "FEH_NET_"+str(i)] = train_fns.unscale(net.prediction, *FEH_scale['FEH']) ### WTF




target.custom.to_csv(param.params['output_directory'] + "target_output.csv", index=False)
