### Author: Devin Whitten
### Date: 11/11/17

### This script serves as the interface for all functions associated with the
### temperature network.
import numpy as np
import pandas as pd
import pickle
import sklearn.neural_network as sknet
import param


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
    def __init__(self, hidden_layer=6, act_fct ="tanh", training_set=None, scale_frame=None):

        self.network = network = sknet.MLPRegressor(hidden_layer_sizes = hidden_layer, activation = act_fct,
                            tol=1e-8, max_iter=int(2e8), random_state=200)

        self.hidden_layer = hidden_layer

        self.act_fct = "tanh"

        self.training_set = training_set

        self.inputs = param.params['format_bands']

        self.scale_frame = scale_frame

    def train(self, train_fct=0.75):
        ### Precondition: network must have been instantiated
        ### train:
        ### test:
        ### inputs: string array of input column names
        ### est: name of column to use for estimate

        self.verification_set = self.training_set.iloc[int(len(self.training_set)*train_fct):]
        self.training_set = self.verification_set.iloc[0:int(len(self.training_set)*train_fct)]

        print("... training network")
        self.network.fit(self.training_set[self.inputs].values,
                         self.training_set[param.params['format_var']].values)
        print("... complete")

        self.verification_set.loc[:,"NET_" + param.params['format_var']] = self.network.predict(self.training_set[self.inputs].values)
        self.training_set.loc[:,"NET_" + param.params['format_var']] = self.network.predict(self.training_set[self.inputs].values)

        return self.verification_set

    def save(self):
        print("Finish")
        intercept_out = open(self.params['output_directory'] + "net_intercepts.pkl", "wb")
        coefs_out =     open(self.params['output_directory'] + "net_coefs.pkl", "wb")

        pickle.dump(self.network.intercepts_, intercept_out)
        pickle.dump(self.network.coefs_,      coefs_out)

        intercept_out.close()
        coefs_out.close()


    def load(self):
        ### There needs to be a check to ensure the same archeticture is being loaded.
        ### I'll do that eventually
        print("Feature not implemented")
        self.Network

    ###### Mutators

    def set_scale(self, input_frame):
        self.scale_frame = input_frame
        return

    def set_training(self, input_frame):
        self.scale_frame = input_frame
        return
