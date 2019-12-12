### Author: Devin Whitten
### Date: 1/5/18

### splitting the interfaces for the network class
### and the network array class

import numpy as np
import pandas as pd
import pickle as pkl
import random
import sklearn.neural_network as sknet
from matplotlib.backends.backend_pdf import PdfPages
import multiprocessing
import matplotlib.pyplot as plt
#import param_IDR as param
import sys, itertools
import pickle as pkl

sys.path.append("interface")
import train_fns, net_functions, io_functions, stat_functions


################################################################################
## Unfortunately Necessary functions

def isin(array, target_filter):
    ### Check array of colors and magnitudes and determine if filter is present
    for ele in array:
        if target_filter in ele.split("_"):
            return True

    return False


def isnotin(array, target_filter):
    ### Check array of colors and magnitudes and determine if filter is present
    for ele in array:
        if target_filter in ele.split("_"):
            return False

    return True

################################################################################
## Main Network_Array class

class Network_Array():
    ### The goal here is to generate an array of networks corresponding to the
    ### combinations of possible network inputs, enables averaging,


    def __init__(self, training_set, target_variable,
                 interp_frame, scale_frame,
                 params, input_type="colors"):


        self.params       = params
        self.training_set = training_set

        ### Forget these for now! They're now set by a member function.
        self.interp_frame = interp_frame
        self.scale_frame  = scale_frame

        self.target_var   = target_variable

        ########################################################################
        ## Things that should be set by the param file
        ########################################################################
        self.array_size    = self.params['array_size']
        self.input_number  = self.params['input_number']
        self.hidden_layers = self.params['hidden_layers']
        self.solver        = self.params['solver']
        self.input_type    = input_type



    def set_input_type(self):
        ### DEPRECIATED, use construct_input_combinations()

        print("Network input_type:  ", self.input_type)

        if self.input_type == "both":
            self.inputs = self.params['format_bands'] + self.training_set.colors

        elif self.input_type == "magnitudes":
            self.inputs = self.params['format_bands']

        elif self.input_type == "colors":
            self.inputs = self.training_set.colors

        else:
            print("ERROR: Bad network input_type")


        print(self.inputs)

        print("... Generating input combinations")
        self.combinations = np.array(list(itertools.combinations(self.inputs, self.input_number)))


        print(len(self.combinations), " of given input type")
        return

    def construct_input_combinations(self):
        ### just builds the combinations ndarray for all of the possible color combinations given the input number

        print("... Generating input combinations : ", self.input_type)

        if self.input_type == "both":
            self.inputs = self.params['format_bands'] + self.training_set.colors

        elif self.input_type == "magnitudes":
            self.inputs = self.params['format_bands']

        elif self.input_type == "colors":
            self.inputs = self.training_set.colors

        else:
            print("ERROR: Bad network input_type")

        self.combinations = np.array(list(itertools.combinations(self.inputs, self.input_number)))
        print("\t" + str(len(self.combinations)), " of given input type")
        return

    def generate_inputs(self, assert_band=None, reject_band=None,
                              assert_colors=None, reject_colors=None):

        ### Assemble the network array according to self.combinations
        print("... Generating", self.target_var,"network array")
        print("\t pre-assert band:  ", len(self.combinations))

        if type(assert_band) != type(None):
            for band in assert_band:
                print("... Asserting: ", band)
                self.combinations = self.combinations[np.array([isin(ele, band) for ele in self.combinations])]

        print("\t pre-assert band:  ", len(self.combinations))
        if type(reject_band) != type(None):

            for band in reject_band:
                print("... Rejecting the following band:  ", band)
                self.combinations = self.combinations[np.array([isnotin(ele, band) for ele in self.combinations])]


        print("\t pre-assert colors length:   ", len(self.combinations))
        #return self.combinations
        if type(assert_colors) != type(None):
            for color in assert_colors:
                print(".... Asserting:  ", color)
                self.combinations = self.combinations[np.array([color in combo for combo in self.combinations])]


        print("\tpre-color rejection combinations:  ", len(self.combinations))

        if type(reject_colors) != type(None):
            print("\tRejecting the following colors:  ", reject_colors)
            for color in reject_colors:
                assert (True in [color in combo for combo in self.combinations]), color + " not in network combinations!"
                self.combinations = self.combinations[np.array([color not in combo for combo in self.combinations])]

        ### shuffle combinations

        self.combinations = self.combinations[np.random.permutation(len(self.combinations))]
        print(len(self.combinations), " total input combinations")

        return

    def initialize_networks(self):
        ### just initializes the Network classes for each ANN subunit in the array
        io_functions.span_window()
        print("... initializing network array:  ", 'random!')
        print("\t solver:  ", 'random!')

        solvers = ['adam', 'lbfgs', 'sgd']
        act_funcs = ['identity', 'logistic', 'tanh', 'relu']

        #if type(self.hidden_layers) == int:


        self.network_array = [net_functions.Network(target_variable = self.target_var, inputs=current_permutation,
                                                    hidden_layer=train_fns.random_layer(self.hidden_layers),
                                                    act_fct = act_funcs[np.random.randint(0, len(act_funcs))],
                                                    solver =  solvers[np.random.randint(0, len(solvers))], ID = ID) for ID, current_permutation in enumerate(self.combinations[0:self.array_size])]


        print("\t total networks: ", len(self.network_array))
        return

    def construct_scale_frame(self, input_frame = None):
        ### These functions enable the input scaling process
        ### to occur entirely in the network_array class, instead of dataset.Dataset
        print('... construct_scale_frame()')
        if input_frame == None:
            print("\t setting input to scale frame")
            input_frame = self.training_set.custom.copy()

        scale_frame = pd.DataFrame()

        for band in self.inputs:
            #### set the stat values for every band
            median = np.median(input_frame[band])
            smad   = stat_functions.S_MAD(input_frame[band])
            std    = np.std(input_frame[band])

            if (smad/std > 1.25) or (smad/std < 0.75):
                print("\t warning: scale discrepancy in band:  ", band)

            scale_frame.loc[:, band] = [median, smad]


        #### Now time for that variable
        print("\t setting ", self.target_var, ' scale')
        scale_frame.loc[:, self.target_var] = [np.median(input_frame[self.target_var]),
                                             np.std(input_frame[self.target_var])]

        print("\t setting scale_frame")
        self.scale_frame = scale_frame

        return



    def construct_interp_frame(self, input_frame = None):
        ### Create interpolation frame, likely based on the <training> set.
        ### I will come back to this!!
        ### wouldn't it be really cool if this used a convex hull?...
        print('... construct_input_frame()')
        if input_frame == None:
            print("\t setting input to scale frame")
            input_frame = self.training_set.custom.copy()

        working = pd.DataFrame()

        for band in self.inputs:
            working.loc[:, band] = [np.percentile(input_frame[band + "_norm"], 1),
                                    np.percentile(input_frame[band + "_norm"], 99)]

        self.interp_frame = working

        return

    def normalize_dataset(self, input_set = None):
        ### scale_frame needs to have been set


        if input_set == None:
            print("... normalizing network training set")
            working = self.training_set.custom.copy()
            mode = "TRAIN"

        else:
            print("... normalizing input_set")
            working = input_set.copy()
            mode = "SCALE"

        ########################################################################

        for band in self.inputs:
            working.loc[:, band + '_norm'] = stat_functions.linear_scale(working[band],
                                                                self.scale_frame[band].iloc[0],
                                                                self.scale_frame[band].iloc[1])

        ########################################################################

        if mode == 'TRAIN':
            self.training_set.custom = working

        elif mode == "SCALE":
            return working

        return




    def generate_train_valid(self, train_fct=0.65):
        ### Just cuts the sample into training and validation sets
        ### Careful with the construction of scale frame

        ### This function changes the nature of self.training_set, just be careful..

        print("... generating train/valid sets")


        self.verification_set = self.training_set.custom.iloc[int(len(self.training_set.custom)*train_fct):].copy()
        self.training_set = self.training_set.custom.iloc[0:int(len(self.training_set.custom)*train_fct)].copy()

        if self.target_var == 'FEH':
            print('Verification < -2.5 :  ', len(self.verification_set[stat_functions.unscale(self.verification_set['FEH'], *self.scale_frame['FEH']) < -2.5]))
            print('Training     < -2.5 :  ', len(self.training_set[stat_functions.unscale(self.training_set['FEH'], *self.scale_frame['FEH']) < -2.5]))


        print('Training   Size:   ', len(self.training_set))
        print('Validation Size:   ', len(self.verification_set))


        return


    def train(self, iterations=3, pool=False):
        io_functions.span_window()
        print("... training network array")

        ### Trains array of networks, sets the verification and target set
        ### iterations: number of networks to train
        ### initializes the verification and training sets

        if pool:
            nodes = multiprocessing.cpu_count()
            print("\t training on:   ", nodes, 'nodes...')
            p=multiprocessing.Pool(nodes)



        for i in range(iterations):
            print("\t iterating training procedure:  ", i)


            if pool:
                pool_obj = train_fns.train_pool(self.training_set, self.network_array)
                self.trained_networks = p.map(pool_obj.train_network, [(net, ID) for net, ID in zip(self.network_array,
                                                                        np.arange(0, len(self.network_array), 1)
                                                                        )])

                self.network_array = self.trained_networks


            else:
                print("\t training sequentially")
                [net.train_on(self.training_set, ID) for ID, net in enumerate(self.network_array)]

            ###### Adding the outlier rejection here
            ### no score information yet
            output = np.matrix([net.predict(self.training_set) for net in self.network_array]).T


            self.training_set.loc[:, 'NET_' + self.target_var] = [np.average(output[i, :]) for i in range(output.shape[0])]
            #self.training_set.loc[:, self.target_var] = train_fns.unscale(self.training_set[self.target_var], *self.scale_frame[self.target_var])


            output = np.matrix([net.predict(self.verification_set) for net in self.network_array]).T

            self.verification_set.loc[:, 'NET_' + self.target_var] = [np.average(output[i, :]) for i in range(output.shape[0])] #(np.dot(output, self.scores)/self.scores.sum()).T
            #self.verification_set.loc[:, self.target_var] = train_fns.unscale(self.verification_set[self.target_var], *self.scale_frame[self.target_var])

            ### compute residual

            self.training_residual = self.training_set['NET_' + self.target_var] - self.training_set[self.target_var]
            self.verification_residual = self.verification_set['NET_' + self.target_var] - self.verification_set[self.target_var]

            ### trim the training set

            print("\t trimming training set by residual")
            original = len(self.training_residual)
            ### this is a problem, the outlier rejection is removing low metallicity stars in the training set.........
            boo_array = np.ones(len(self.training_set), dtype=bool)

            if self.target_var == 'FEH':
                boo_array = list(stat_functions.unscale(self.training_set['FEH'], *self.scale_frame['FEH']) < -2.5)
                print('cool')

            outlier = list(abs(self.training_residual) < np.percentile(abs(self.training_residual), 99))



            self.training_set = self.training_set.loc[np.logical_or(boo_array, outlier)]
            self.training_residual = self.training_residual.loc[np.logical_or(boo_array, outlier)]

            print("\t Stars removed:   ", original - len(self.training_set))

        return




    def train_pool(self, iterations=50, core_fraction=0.5):
        #### DEPRECIATED
        ### Let's try to multiprocess the network_array training
        ### I'll come back to this
        print("... train_pool")
        core_number = int(core_fraction*multiprocessing.cpu_count())
        print("\t training on ", core_number, ' cores')
        ### multiprocessing pool
        pool = multiprocessing.Pool(core_number)
        pool.map(self.train_on, enumerate(self.network_array))
        return

    def info(self):
        print()
        for i in range(len(self.network_array)):
            print("\t Net:  ", self.network_array[i].get_id(), self.network_array[i].get_inputs())


    def eval_performance(self):
        ### Runs the network array on the verification sets
        ### Sets the median absolute deviation

        ### This should take into account the low residual in the event that variable == FEH
        print("... eval_performance()")
        print("\t target variable: ", self.target_var)

        [net.compute_residual(verify_set = self.verification_set, scale_frame = self.scale_frame) for net in self.network_array]


        [net.set_mad(train_fns.MAD(net.residual)) for net in self.network_array]

        if self.target_var == 'FEH':
            print("\t FEH:  setting low_mad in network score")
            [net.set_low_mad(train_fns.MAD(net.low_residual)) for net in self.network_array]
            total_mad = np.array([net.get_mad() + net.get_low_mad() + 3.*np.median(net.get_low_residual()) for net in self.network_array])

        elif self.target_var == "TEFF":
            print("\t TEFF:  setting mad to network score")
            total_mad = np.array([net.get_mad() for net in self.network_array])

        elif self.target_var == 'CFE':
            total_mad = np.array([net.get_mad() for net in self.network_array])

        elif self.target_var == 'AC':
            total_mad = np.array([net.get_mad() for net in self.network_array])

        else:
            print("\t eval_performance() can't handle:  ", self.target_var)


        print("\t Setting network mad")

        self.scores = np.divide(1., np.power(total_mad,3))

        return

        ######## NEW EDIT, SELECT SPECIFIED NUMBER OF HIGHEST PERFORMING NETWORKS

    def skim_networks(self, select):
        ### select specified the number of highest performing networks to keep
        ### Precondition: eval_performance() has set the scores

        print("... skim_networks()")
        self.network_array = [self.network_array[i] for i in np.argsort(self.scores)[::-1]]
        self.scores = self.scores[np.argsort(self.scores)[::-1]]

        #### should be sorted, now skim
        self.network_array = self.network_array[:select]
        self.scores = self.scores[:select]

        return

    def write_network_performance(self, filename=None):
        ### Write out the performance of the randomly drawn input combinations
        ### Must have run eval_performance()
        ### handles low_residual for FEH nets
        print("... write_network_performance()")
        if filename == None: filename = "network_combination_residuals.pkl"

        if self.target_var == "FEH":
            print("\t writing network input residuals to cache/", filename)
            network_residual = {"combination": [net.inputs for net in self.network_array],
                                "residual": [net.get_mad() for net in self.network_array] ,
                                "low_residual": [net.get_low_mad() for net in self.network_array],
                                "score": self.scores}

        else:
            network_residual = {"combination": [net.inputs for net in self.network_array],
                                "residual":  [net.get_mad() for net in self.network_array],
                                "score": self.scores}


        print("\t writing")
        #pd.DataFrame(network_residual).to_csv("cache/" + filename, index=False)
        file_out = open(self.params['SPHINX_path'] + "cache/"+filename, "wb")
        pkl.dump(network_residual, file_out)
        file_out.close()

        return

    def predict(self, target_set, flag_invalid=True):

        ### use the 1/MADs determined in eval_performance to perform a weighted average estimate for the
        ### target input
        ### might as well unscale here as well.
        ### Checks inputs against the interp_frame, sets flag variable in the target set

        ### Here now we also must run scale_photometry
        target_set = normalize_dataset(target_set)

        print("... running array prediction:  ")

        output = np.vstack([train_fns.unscale(net.predict(target_set.custom), *self.scale_frame[self.target_var]) for net in self.network_array]).T

        print("\t flagging network extrapolations")
        flag = np.vstack([net.is_interpolating(target_frame = target_set.custom, interp_frame = self.interp_frame) for net in self.network_array]).T

        self.output = output
        self.flag = flag
        nan_flag = np.copy(self.flag)
        nan_flag[nan_flag == 0] = np.nan

        ##### This is the line
        self.target_err = np.array([train_fns.MAD_finite(np.array(row))/0.6745 for row in output*nan_flag])
        #self.target_err = np.array([train_fns.weighted_error(row, self.scores) for row in output*nan_flag])


        ### FLAG MASKS extrapolation estimates, however scores will be different for each row!!
        flagged_score_array = np.dot(flag, self.scores)
        flagged_score_array[flagged_score_array == 0] = np.nan

        if flag_invalid:
            print("\t masking output matrix with network flags")
            self.target_est = np.divide(np.dot(output * flag, self.scores), flagged_score_array)
        else:
            print("\t not masking output matrix with network flags")
            self.target_est = np.divide(np.dot(output, self.scores), self.scores.sum())

        print("\t appending columns to target_set")

        ######### DONT NEED TO UNSCALE!!!!!!!!!
        target_set.custom.loc[:, "NET_" + self.target_var] = self.target_est # train_fns.unscale(self.target_est, *self.scale_frame[self.target_var])

        ######### DONT NEED TO UNSCALE!!!!!!!!!
        target_set.custom.loc[:, "NET_" + self.target_var + "_ERR"] = self.target_err

        target_set.custom.loc[:, "NET_ARRAY_"+self.target_var + "_FLAG"] = [row.sum() for row in self.flag]
        print("\t complete")

        return self.target_est, self.target_err



    def predict_all_networks(self, target_set):
        ### Run each network in the array on the target_set
        ### append values with ID to target frame
        print("... Running all networks on target set")
        for net in self.network_array:
            target_set.custom.loc[:, "NET_" + str(net.get_id()) +"_"+ self.target_var] = train_fns.unscale(net.predict(target_set.custom), *self.scale_frame[self.target_var])
            target_set.custom.loc[:, "NET_" + str(net.get_id()) +"_"+ self.target_var + "_FLAG"] = net.is_interpolating(target_frame= target_set.custom, interp_frame=self.interp_frame)

    def write_training_results(self):
        ### Just run prediction on the verification and testing sets
        ### THIS IS WHERE THE TRAINING and VERIFICATION variables ARE UNSCALED
        print("... write_training_results()")
        output = np.matrix([train_fns.unscale(net.predict(self.training_set), *self.scale_frame[self.target_var]) for net in self.network_array]).T
        print('\t ', output)
        self.training_set.loc[:, 'NET_' + self.target_var] = (np.dot(output, self.scores)/self.scores.sum()).T
        self.training_set.loc[:, self.target_var] = train_fns.unscale(self.training_set[self.target_var], *self.scale_frame[self.target_var])

        output = np.matrix([train_fns.unscale(net.predict(self.verification_set), *self.scale_frame[self.target_var]) for net in self.network_array]).T
        self.verification_set.loc[:, 'NET_' + self.target_var] = (np.dot(output, self.scores)/self.scores.sum()).T
        self.verification_set.loc[:, self.target_var] = train_fns.unscale(self.verification_set[self.target_var], *self.scale_frame[self.target_var])


        print("\t writing training/verification outputs")

        self.training_set.to_csv(self.params['SPHINX_path'] + self.params['output_directory'] + self.target_var + "_array_training_results.csv", index=False)
        self.verification_set.to_csv(self.params['SPHINX_path'] + self.params['output_directory'] + self.target_var + "_array_verification_results.csv", index=False)

        print("\t done.")

        return


    def training_plots(self):
        print("... Generating training plots")
        ### show the training results in one-to-one residual plots
        span = np.linspace(min(self.verification_set[self.target_var]), max(self.verification_set[self.target_var]), 30)
        pp = PdfPages(self.params['SPHINX_path'] + self.params["output_directory"] + self.target_var + "_FINALplot.pdf")

        fig, ax = plt.subplots(2,1)

        ax[0].scatter(self.verification_set[self.target_var], self.verification_set['NET_' + self.target_var],
                      s=1, label="Verification", alpha=0.65)
        ax[0].scatter(self.training_set[self.target_var], self.training_set['NET_' + self.target_var],
                      s=1, color="black",label="Training", alpha=0.65)


        ax[1].scatter(self.verification_set[self.target_var],
                      self.verification_set['NET_' + self.target_var] - self.verification_set[self.target_var],
                      s=1, alpha=0.65)
        ax[1].scatter(self.training_set[self.target_var],
                      self.training_set['NET_' + self.target_var] - self.training_set[self.target_var],
                      s=1, color="black", alpha=0.65)

        ax[0].plot(span, span, linestyle="--", color="black", alpha=0.75)
        ax[1].plot(span, np.zeros(30), linestyle="--", color="black", alpha=0.75)

        [label.axvline(self.params[self.target_var + "_MIN"]) for label in ax]
        [label.axvline(self.params[self.target_var + "_MAX"]) for label in ax]

        ax[0].legend()
        pp.savefig()
        pp.close()


    def process(self, assert_band, assert_colors, reject_colors, target_set):
        self.set_input_type()
        self.generate_inputs(assert_band = assert_band, assert_colors = assert_colors, reject_colors=reject_colors)
        self.train()
        self.eval_performance()
        self.prediction(target_set)





    def save_state(self, file_name):
        ### For now I will try to simply save the network in a pickle file
        print("... saving network")
        pkl.dump(self, open(self.params['SPHINX_path'] + "net_pkl/" + file_name + ".pkl", 'wb'), protocol=4)


        return

    def load_state(self, file_name):
        print("... loading:  ", file_name)
        self = pkl.load(open(self.params['SPHINX_path'] + "net_pkl/" + file_name, 'rb'))

        return
