### Author: Devin Whitten
### Date: 1/5/18

### splitting the interfaces for the network class
### and the network array class

import numpy as np
import pandas as pd
import pickle, random
import sklearn.neural_network as sknet
import multiprocessing
import param
import sys, itertools

sys.path.append("/Users/MasterD/Google Drive/JPLUS/Pipeline3.0/Temperature/interface")
import train_fns, net_functions


class Network_Array():
    ### The goal here is to generate an array of networks corresponding to the
    ### combinations of possible network inputs, enables averaging,

    def __init__(self, training_set, interp_frame, target_variable, scale_frame, input_number=6, array_size=50):
        try:
            self.inputs = param.params['format_bands'] + training_set.colors
        except:
            print("no colors in training set... fine")
            self.inputs = param.params['format_bands']

        self.training_set = training_set.custom
        self.interp_frame = interp_frame
        self.target_var = target_variable
        self.array_size = array_size
        self.scale_frame = scale_frame


        self.combinations = np.array(list(itertools.combinations(self.inputs, input_number)))
        print("Network array built for the following inputs:  ")
        print(self.inputs)



    def generate(self, assert_band="ALL"):
        ### Assemble the network array according to self.combinations
        print("... Generating", self.target_var,"network array")
        if assert_band != "ALL":
            for element in assert_band:
                print("... Asserting: ", element)
                self.combinations = self.combinations[[element in ele for ele in self.combinations]]

        ### shuffle combinations
        self.combinations = self.combinations[np.random.permutation(len(self.combinations))]
        print(len(self.combinations), " total input combinations")
        self.network_array = [net_functions.Network(target_variable = self.target_var, inputs=current_permutation, ID = ID) for ID, current_permutation in enumerate(self.combinations[0:self.array_size])]


    def train(self, train_fct=0.75):
        print("train_array...")
        ### Trains array of networks, sets the verification and target set
        ### iterations: number of networks to train

        self.verification_set = self.training_set.iloc[int(len(self.training_set)*train_fct):].copy()
        self.training_set = self.training_set.iloc[0:int(len(self.training_set)*train_fct)].copy()



        [net.train_on(self.training_set, ID) for ID, net in enumerate(self.network_array)]

    def train_test(self, net):
        net[1].train_on(self.training_set, net[0])
        return

    def train_pool(self, train_fct=0.75, iterations=50, core_fraction=0.5):
        ### Let's try to multiprocess the network_array training
        ### I'll come back to this
        print("train_pool")
        core_number = int(core_fraction*multiprocessing.cpu_count())
        ### multiprocessing pool
        pool = multiprocessing.Pool(core_number)
        pool.map(self.train_on, enumerate(self.network_array))
        return

    def info(self):
        print()
        for i in range(len(self.network_array)):
            print("Net:  ", self.network_array[i].get_id(), self.network_array[i].get_inputs())


    def eval_performance(self):
        ### Runs the network array on the verification sets
        ### Sets the median absolute deviation
        [net.verify(verify_set = self.verification_set) for net in self.network_array]
        #thing = self.network_array[0].predict(input_frame = self.verification_set)
        #print(self.network_array[2].residual* self.scale_frame[self.target_var].iloc[1])

        [net.set_mad(train_fns.MAD(net.residual * self.scale_frame[self.target_var].iloc[1])) for net in self.network_array]
        self.scores = np.divide(1., np.array([net.get_mad() for net in self.network_array]))
        return

    def prediction(self, target_set):
        print()
        ### use the 1/MADs determined in eval_performance to perform a weighted average estimate for the
        ### target input
        ## might as well unscale here as well.
        output = np.matrix([train_fns.unscale(net.predict(target_set.custom), *self.scale_frame[self.target_var]) for net in self.network_array]).T

        self.output = output
        #return np.dot(output, np.divide(1.,self.MADs))/np.divide(1., TEFF_net.MADs).sum()
        self.target_err = np.array([train_fns.MAD(np.array(row)) for row in output])
        #return output
        self.target_est = (np.dot(output, self.scores)/self.scores.sum()).T
        return self.target_est, self.target_err



    def save_state(self):
        ### Need to work on this.
        return
