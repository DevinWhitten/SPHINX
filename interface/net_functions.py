### Author: Devin Whitten
### Date: 11/11/17

### This script serves as the interface for all functions associated with the
### temperature network.
import numpy as np
import pandas as pd
import pickle
import sklearn.neural_network as sknet
import param
import sys
import itertools

sys.path.append("interface")
import train_fns

def MAD(input_vector):
    return np.median(np.abs(input_vector - np.median(input_vector)))

def identify_outliers(frame, scale):
    ### Returns outliers in input frame by comparing n-SSPP Temperatures
    ### with network predictions
    RESIDUAL = frame['NET_TEFF'] - frame['TEFF']
    outliers = frame[np.abs(RESIDUAL) > scale*MAD(RESIDUAL)]
    print(len(outliers), " identified in input set")
    return outliers

def iterative_fit(iterations=5):
    ### iterates network fitting following successive outlier rejection
    return

def performance(network, Train, Valid, Native, Inputs):
    network.fit(X = Train[Inputs].values, y=Train['FEH'].values)

    Train_NET = network.predict(Train[Inputs].values)
    Valid_NET = network.predict(Valid[Inputs].values)
    Native_NET= network.predict(Native[Inputs].values)

    return MAD(Train_NET - Train['FEH']), MAD(Valid_NET - Valid['FEH']), MAD(Native_NET- Native['FEH'])


####### MAIN Network class definition
### Needs to be able to save/load weights
### optimization technique would be important too.

class Network():
    def __init__(self, target_variable, hidden_layer=6, inputs=param.params['format_bands'],
                 act_fct ="tanh", training_set=None, scale_frame=None, interp_frame =None, ID=None):

        self.network = network = sknet.MLPRegressor(hidden_layer_sizes = hidden_layer, activation = act_fct,
                            solver="adam", tol=1e-9, max_iter=int(2e8), learning_rate="adaptive", early_stopping=False,  random_state=200)

        self.hidden_layer = hidden_layer

        self.act_fct = "tanh"

        self.training_set = training_set

        self.inputs = inputs

        self.scale_frame = scale_frame

        self.interp_frame = interp_frame

        self.target_var = target_variable #param.params['format_var']

        self.ID = ID

    def train(self, train_fct=0.85):
        ### Precondition: network must have been instantiated
        ### train:
        ### test:
        ### inputs: string array of input column names
        ### est: name of column to use for estimate

        self.verification_set = self.training_set.iloc[int(len(self.training_set)*train_fct):].copy()
        self.training_set = self.training_set.iloc[0:int(len(self.training_set)*train_fct)].copy()

        print("... training network")
        self.network.fit(self.training_set[self.inputs].values,
                         self.training_set[self.target_var].values)
        print("... complete")

        self.verification_set.loc[:,"NET_" + self.target_var] = self.network.predict(self.verification_set[self.inputs].values)
        self.training_set.loc[:,"NET_" + self.target_var] = self.network.predict(self.training_set[self.inputs].values)

        return self.verification_set

    def train_on(self, training_input, ID=None):
        ### This is the function for self.training_set == None,
        ### intended for Network_Array.train_array()
        print("Training on: " + str(ID), sep=" ", end = "\r", flush=True)
        #print(training_input.columns)

        self.network.fit(training_input[list(self.inputs)].values,
                         training_input[self.target_var].values)

    def set_target(self, input_frame):
        ### Mutator to set the input science (target) frame for the network object
        self.target_frame = input_frame
        print("... target frame set")


    def is_interpolating(self, target_frame, interp_frame):
        ### check if the inputs in the target frame are withing interp_frame Bounds
        ### set flag
        FLAG = []
        for i, row in target_frame.iterrows():
            FLAG.append(False not in [interp_frame[column].iloc[0] < row[column] < interp_frame[column].iloc[1] for column in self.inputs])

        self.FLAG = np.array(FLAG, dtype=int)
        return self.FLAG




    def predict(self, input_frame, interp_frame =None):
        ## Basic prediction procedure, with added function to reject
        #############################################################


        ### This should be the main function for network prediction
        if input_frame is None: ### == not working for DataFrame comparison
            ## DEPRECATED
            self.target_frame.loc[:, "NET_" + self.target_var] = self.network.predict(self.target_frame[self.inputs].values)

            return self.target_frame

        else: ### May I want to run a net input_frame on the same network
            #print("... running network", self.ID, "on target frame")
            print("... running network: " + str(self.ID), sep=" ", end = "\r", flush=True)
            self.prediction = self.network.predict(input_frame[list(self.inputs)].values)

            ### 1 for successful estimate, 0 otherwise
            #self.prediction_flag = input_frame[list(self.inputs)]

            return self.prediction





    def compute_residual(self, verify_set, scale_frame):
        ### Intending to use with Network_Array.eval_performance()
        ### sets the residual array from the input verification set

        output = train_fns.unscale(self.network.predict(verify_set[list(self.inputs)].values), *scale_frame[self.target_var])
        ### still scaled
        true_value = train_fns.unscale(verify_set[self.target_var], *scale_frame[self.target_var])

        self.residual = output - true_value
        self.low_residual = output[true_value < -2.5] - true_value[true_value < -2.5]

    def set_mad(self, mad):
        ### sets the median absolute deviation of network performance
        self.mad = mad

    def set_low_mad(self, low_mad):
        self.low_mad = low_mad

    def get_mad(self):
        return self.mad

    def get_low_mad(self):
        return self.low_mad

    def unscale_target_variable(self):
        print("unscale_target_variable")
        ## Precondition: Training must have been run to produce "NET" results
        self.verification_set.loc[:, "NET_"+self.target_var] = train_fns.unscale(self.verification_set.loc[:, "NET_"+self.target_var],
                                                                                            self.scale_frame[self.target_var].iloc[0],
                                                                                            self.scale_frame[self.target_var].iloc[1])

        self.verification_set.loc[:, self.target_var] = train_fns.unscale(self.verification_set.loc[:, self.target_var],
                                                                                            self.scale_frame[self.target_var].iloc[0],
                                                                                            self.scale_frame[self.target_var].iloc[1])

        self.training_set.loc[:, "NET_"+self.target_var] = train_fns.unscale(self.training_set.loc[:, "NET_"+self.target_var],
                                                                                            self.scale_frame[self.target_var].iloc[0],
                                                                                            self.scale_frame[self.target_var].iloc[1])

        self.training_set.loc[:, self.target_var] = train_fns.unscale(self.training_set.loc[:, self.target_var],
                                                                                            self.scale_frame[self.target_var].iloc[0],
                                                                                            self.scale_frame[self.target_var].iloc[1])

        self.target_frame.loc[:, "NET_" + self.target_var] = train_fns.unscale(self.target_frame.loc[:, "NET_" + self.target_var],
                                                                               self.scale_frame[self.target_var].iloc[0],
                                                                               self.scale_frame[self.target_var].iloc[1])

    def assess_performance(self):
        ### Compute relavent statistics of the network performance
        ### I'll probably modify this function a lot
        print("assess_performance()")
        self.training_residual = self.training_set["NET_"+self.target_var] - self.training_set[self.target_var]
        self.verification_residual = self.verification_set["NET_"+self.target_var] - self.verification_set[self.target_var]

        print("Training Residual:     ", train_fns.gaussian_sigma(self.training_residual)[0])
        print("Verification Residual: ", train_fns.gaussian_sigma(self.verification_residual)[0])


    def write_estimates(self):
        ## This is the tentative function to write all network predictions of the training, validation, and target sets
        print("... writing estimates")
        self.training_set.to_csv(param.params['output_directory']+self.target_var + "_training_result.csv", index=False)
        self.verification_set.to_csv(param.params['output_directory']+self.target_var + "_verification_result.csv", index=False)
        self.target_frame.to_csv(param.params['output_directory']+self.target_var + "_target_result.csv", index=False)



    def save(self):
        print("Finish")
        intercept_out = open(self.params['output_directory'] + self.target_var + "_net_intercepts.pkl", "wb")
        coefs_out =     open(self.params['output_directory'] + self.target_var + "_net_coefs.pkl", "wb")

        pickle.dump(self.network.intercepts_, intercept_out)
        pickle.dump(self.network.coefs_,      coefs_out)

        intercept_out.close()
        coefs_out.close()


    def load(self):
        ### There needs to be a check to ensure the same archeticture is being loaded.
        ### I'll do that eventually
        intercept_in = open(self.params['output_directory'] + "net_intercepts.pkl", "r")
        coefs_in     = open(self.params['output_directory'] + "net_coefs.pkl", "r")

        self.Network.intercepts_ = pickle.load(intercept_in)
        self.Network.coefs_      = pickle.load(coefs_in)

    ###### Mutators

    def set_scale(self, input_frame):
        self.scale_frame = input_frame
        return

    def set_training(self, input_frame):
        self.scale_frame = input_frame
        return

    def run(self):
        self.train()
        self.predict()
        self.unscale_target_variable()
        self.assess_performance()
        self.write_estimates()

    ###### Accessors
    def get_id(self):
        return self.ID

    def get_inputs(self):
        return self.inputs

################################################################################
