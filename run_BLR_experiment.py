#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 20:25:24 2019

@author: jeremiasknoblauch

Description: This runs our experiments for GVI Bayesian Linear Regs
"""


import os,sys
import numpy as np
from simulator import BLRSimulator
from BBGVI import BBGVI
from Loss import LogNormalLossBLR
from Divergence import MFN_MFN_RAD, MFN_MFN_AD, MFN_MFN_FD, MFN_MFN_ED
from Divergence import MFN_MFN_KLD, MFN_MFN_reverse_KLD, MFN_MFN_JeffreysD



"""This function (i) sets up the data, (ii) sets up the model and (iii) runs
the entire model/inference before returning the relevant object"""
def BLR_experiment(d, n, sigma2, coefs, X_mean, X_var, seed_data, 
                       settings = {
                                   'test print': True,
                                   'print outcomes': True,
                                   'divergence type': "KLD", 
                                   'divergence hyperparameter': 0.5, 
                                   'divergence weight': 1.0
                                   }, 
                        opt_settings = {
                              'learning rate': 0.01, 
                              'epochs': 2000, 
                              'num_samples': 100,
                              'optimization seed': 0,
                              'data subsample size': None
                              }):
        
        """STEP 1: Set up/generate the data"""
        BLR_gen = BLRSimulator(d,coefs,X_mean,X_var,sigma2=sigma2)
        (X,Y) = BLR_gen.generate(n, seed = seed_data)
        
        #Print the truth to console
        if settings['test print']:
            print("Coefs are", coefs)
            print("sigma2 is", sigma2)
            print("X mean is", X_mean)
            print("X var is", X_var)
            print("X has column means", np.mean(X,axis=0))
            print("X has column variance", np.var(X,axis=0))
            print("expected value of Y is ", np.dot(coefs, X_mean))
            print("actual meean of Y is ", np.mean(Y))
            print("LS estimator is", np.linalg.solve(a=np.transpose(X).dot(X),b=np.transpose(X).dot(Y)))
            print("X shape", X.shape, "Y shape", Y.shape)          
            
        """STEP 2: Set up the model"""
        
        """STEP 2.1: Set up the prior"""
        #+1 because we fit the observation noise, too
        prior_mu = np.zeros(d+1).flatten()
        prior_sigma = 10.0 * np.ones(d+1).flatten()
            
        """STEP 2.2: Retrieve hyperparameters & weight for D"""
        hyperparameter = settings['divergence hyperparameter']
        weight = settings['divergence weight']
        
        """STEP 2.3: Set up the prior regularizers"""
        if settings['divergence type'] == "RAD":
            D = MFN_MFN_RAD(prior_mu, prior_sigma, alpha = hyperparameter,
                            weight = weight)
        elif settings['divergence type'] == "AD":
            D = MFN_MFN_AD(prior_mu, prior_sigma, alpha = hyperparameter,
                            weight = weight) 
        elif settings['divergence type'] == "ED":
            D = MFN_MFN_ED(prior_mu, prior_sigma, weight = weight)
        elif settings['divergence type'] == "FD":
            D = MFN_MFN_FD(prior_mu, prior_sigma, weight = weight)
        elif settings['divergence type'] == "JeffreysD":
            D = MFN_MFN_JeffreysD(prior_mu, prior_sigma, 
                    relative_weight = hyperparameter, weight = weight)
        elif settings['divergence type'] == "reverseKLD":
            D = MFN_MFN_reverse_KLD(prior_mu, prior_sigma, weight = weight)
        elif settings['divergence type'] == "KLD":
            D = MFN_MFN_KLD(prior_mu, prior_sigma, weight = weight)
        
        """STEP 2.4: Set up the loss"""
        BLRLoss = LogNormalLossBLR(d)
        
        """STEP 3: Put together GVI object"""
        
        """STEP 3.2: Set up Black Box GVI object"""
        BBGVI_obj = BBGVI(D=D, L=BLRLoss)
        
        """STEP 3.3: Optimization/Inference"""
        if opt_settings['data subsample size'] is None:
            data_samples = n
        else:
            data_samples = opt_settings['data subsample size']
        
        np.random.seed(opt_settings['optimization seed'])
        BBGVI_obj.fit_q(Y=Y,X=X, 
                   epochs = opt_settings['epochs'], 
                   learning_rate = opt_settings['learning rate'], 
                   K=opt_settings['num_samples'], 
                   M = data_samples)
        
        """STEP 3.4: Print outcomes and return result"""
        if settings['print outcomes']:
            #check that output is sensible
            BBGVI_obj.report_parameters()
        return BBGVI_obj
    
    
"""Take in the output of the BMM_experiment function (a BBGVI object) and save 
the results to the hard drive under the relevant path"""    
def save_GVI_BLR_results(BBGVI_obj, path_to_save, file_name, file_id):
    
    """     Input:      BBGVI object. 
            Extract:    the variational parameters.
            Save:       these parameters
    """
    
    #Build path if it does not exist
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    
    #Extract
    names, values = BBGVI_obj.L.report_parameters(BBGVI_obj.q_parser, 
                                                  BBGVI_obj.q_params)
    
    #Re-arrange ('means', 'variances', 'mean', 'variance') 
    result_types = []
    for name in names:
        result_types = result_types + [name.split(' ')[-1]]


    #Save the three parts of the inference into three different files
    for (result_type, res) in zip(result_types, values):
        
        #check if res is a scalar -- need numpy array to use np.savetxt
        if isinstance(res, float):
            res = np.array([res])
        
        file_string = (path_to_save + "/" + file_name + result_type + 
                       "_" + file_id + ".txt")
        np.savetxt(file_string, res, fmt='%10.8f')
        
#        
#        with open(file_string + "_test_ll.txt", 'a') as f:
#                f.write(repr(neg_test_ll) + '\n')
#        with open(file_string + "_test_error.txt", 'a') as f:
#                f.write(repr(test_error) + '\n')
#        with open(file_string + "_test_time.txt", 'a') as f:
#                f.write(repr(running_time) + '\n')
        
    print("Saved", file_string)




if __name__ == '__main__':
    
    # read in from console    
    n = int(sys.argv[1])
    Dtype = str(sys.argv[2])
    Dhyperparameter = float(sys.argv[3])
    simulation_number = int(sys.argv[4])  
       
    # path to save stuff
    base_path = ""
    path_to_save = base_path + "/BLR/" "n=" + str(n) + "/D=" + Dtype + "_param=" + str(Dhyperparameter)
    file_name = "" # once in the correct folder, we just enumerate files
    file_id = str(simulation_number)
    
    
    # set up the data generating mechanism
    d = 20
    np.random.seed(10)
    coefs = np.random.normal(3, 10, d)
    X_mean = np.random.normal(0, 2, d)
    X_var = (2.0 + np.random.normal(0, 3, d) ** 2)
    sigma2 = 25
    seed_data = 0
    
    
    # reshuffle seed for the optimization
    np.random.seed(simulation_number)
    
    # set up the settings
    settings = {
               'test print': False,
               'print outcomes': False,
               'divergence type': Dtype, 
               'divergence hyperparameter': Dhyperparameter, 
               'divergence weight': 1.0
               }
    
    opt_settings = {
                  'learning rate': 0.01, 
                  'epochs': 2000, 
                  'num_samples': 100,
                  'optimization seed': 0,
                  'data subsample size': 100
                  }
    
    # run GVI
    res = BLR_experiment(d, n, sigma2, coefs, X_mean, X_var, seed_data, 
                       settings, opt_settings
                        )
    
    # save results
    save_GVI_BLR_results(res, path_to_save, file_name, file_id)
        
 

