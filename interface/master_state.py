### This is a very new addition, intended to address the
### new network operation where we segregate on temperature

import sys
import copy
sys.path.append("interface")
import train_fns, net_functions, io_functions, stat_functions, dataset




class Master_State():

    def __init__(self, params):
        ### here we're just going to load the
        ### networks specified in the parameter file

        self.params = params
        print("... loading network states")

        self.TEFF_NET = io_functions.load_network_state(params, params['TEFF_NET'])


        self.FEH_NET  = {'WARM' : io_functions.load_network_state(params, params['FEH_NET_WARM']),
                         'COOL' : io_functions.load_network_state(params, params['FEH_NET_COOL'])}

        self.AC_NET   = {'WARM' : io_functions.load_network_state(params, params['AC_NET_WARM']),
                         'COOL' : io_functions.load_network_state(params, params['AC_NET_COOL'])}


        return


    def predict(self, target):
        #### main network estimation procedure
        print("... master_state.predict()")
        #### first get the temperatures


        self.TEFF_NET.predict(target)


        ### unfortunately the most straightforward way is to
        ### duplicated the frame prior to merge. might be a memory problem
        ### with large data

        ### gen unique custom tag
        target.custom.loc[:, "SPH_IND"] = np.arange(0, len(target.custom))

        LEFT_FRAME  = copy.copy(target)
        RIGHT_FRAME = copy.copy(target)

        ### BASIC HARD TEMP CUT

        #### METALLICITY RUNS
        io_functions.span_window()
        print("     Running [Fe/H] Network Determinations")

        self.FEH_NET['WARM'].predict(RIGHT_FRAME)

        self.FEH_NET['COOL'].predict(LEFT_FRAME)


        io_functions.span_window()
        print("     Running A(C) Networks Determinations")

        self.AC_NET['WARM'].predict(RIGHT_FRAME)

        self.AC_NET['COOL'].predict(LEFT_FRAME)



        LEFT_FRAME.merge_master(vars = ['TEFF', 'FEH', 'AC'])
        RIGHT_FRAME.merge_master(vars = ['TEFF', 'FEH', 'AC'])


        io_functions.span_window("~")

        target.custom = dataset.merge_datasets(LEFT_FRAME.custom, RIGHT_FRAME.custom)

        print("... recycling old frames.  :)")
        del LEFT_FRAME
        del RIGHT_FRAME


        return target
