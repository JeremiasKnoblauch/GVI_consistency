#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 22 20:50:01 2019

@author: jeremiasknoblauch

Description: This runs BMM experiments on server
"""


import os,sys
import numpy as np
from simulator import BMMSimulator
from BBGVI import BBGVI
from Loss import BMMLogLossFixedSigma
from Divergence import MFN_MFN_RAD, MFN_MFN_AD, MFN_MFN_FD, MFN_MFN_ED
from Divergence import MFN_MFN_KLD, MFN_MFN_reverse_KLD, MFN_MFN_JeffreysD


"""TEST: Can I set up my BBGVI algorithm?"""
TEST_BMM = True



"""This function (i) sets up the data, (ii) sets up the model and (iii) runs
the entire model/inference before returning the relevant object"""
def BMM_experiment(d, num_clusters, n, mixture_probs, mu, sigma2, seed_data,
                   settings = {
                              'test plot': True,
                              'print outcomes': True,
                              'outliers': False, 
                              'outlier frequency': 0.05,
                              'outlier size': 10.0, 
                              'outlier variance': 3,
                              'robust likelihood': False,
                              'robust likelihood gamma': 1.01,
                              'prior misspecification': False, 
                              'prior misspecification pi variance': None,
                              'prior misspecification pi mean': None,
                              'robust divergence': False, 
                              'robust divergence type': "RAD", 
                              'robust divergence hyperparameter': 0.5, 
                              'robust divergence weight': 1.0
                              }, 
                    opt_settings = {
                              'learning rate': 0.01, 
                              'epochs': 2000, 
                              'num_samples': 100,
                              'optimization seed': 0, 
                              'data subsample size': None
                              }):
    
    """STEP 1: Generate the data"""
    
    """STEP 1.1: Check that inputs are correct"""
    if mixture_probs.shape != (num_clusters,):
        print("ERROR! shape of mixture_probs = ", mixture_probs.shape, 
               "and does not match num_clusters = ", num_clusters)
    if mu.shape != (num_clusters, d):
        print("ERROR! shape of mu = ", mu.shape, 
               "and does not match num_clusters x d = ", (num_clusters, d))
    if sigma2.shape != (num_clusters, d):
        print("ERROR! shape of mu = ", sigma2.shape, 
               "and does not match num_clusters x d = ", (num_clusters, d))
     
    """STEP 1.2: Generate the data"""
    BMM_gen = BMMSimulator(d, mixture_probs, mu, sigma2)
    Y, clusters = BMM_gen.generate(n, seed = seed_data)
    Y.reshape(n,d)
    
    """STEP 1.3: Add noise if needed"""
    if settings['outliers']:
        
        #retrieve values
        frequency = settings['outlier frequency']
        size = settings['outlier size']
        variance = settings['outlier variance']
    
        #corrupt Y-values
        corruption = size + variance * np.random.normal(0,1,(n,d))
        rand_ind = np.random.choice(n, int(n * frequency), replace=False)
        Y[rand_ind,:] = Y[rand_ind,:]  + corruption[rand_ind,:]
    
    """STEP 1.4: Plot to make sure the data is as we want it"""
    if settings['test plot']:
        LABEL_COLOR_MAP = {0 : 'r', 1 : 'k',  2 : 'b'}
        import matplotlib.pyplot as plt
        label_color = [LABEL_COLOR_MAP[l] for l in clusters]
        
        if d>2:
            
            #Initialize plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            
            #Plot the (three) clusters
            x,y,z = Y[:,0], Y[:,1], Y[:,2]
            ax.scatter(x,y,z,marker = 'o',c=label_color)
        
        elif d>1:
            plt.scatter(Y[:,0], Y[:,1], c=label_color)
            
        elif d==1:
            plt.scatter(Y[:,0], y=np.zeros(n), c=label_color)
        plt.show()
    
    
    """STEP 2: Set up the model specs"""
    
    """STEP 2.1: Set up the priors"""
    if settings['prior misspecification']:
        
        #Prior mean
        if settings['prior misspecification pi mean'] is not None:
            prior_mu = settings['prior misspecification pi mean']
        else:
            prior_mu = -10.0 * np.ones(shape = (d,num_clusters)).flatten()
          
        #Prior variance    
        if settings['prior misspecification pi variance'] is not None:
            prior_sigma = settings['prior misspecification pi variance']
        else:
            prior_sigma = 0.1 * np.ones(shape = (d,num_clusters)).flatten()
            
    else:
        prior_mu = np.zeros(shape = (d,num_clusters)).flatten()
        prior_sigma = 10.0 * np.ones(shape = (d,num_clusters)).flatten()
        
    """STEP 2.2: Retrieve hyperparameters & weight for D"""
    hyperparameter = settings['robust divergence hyperparameter']
    weight = settings['robust divergence weight']
    
    """STEP 2.3: Set up the prior regularizers"""
    if settings['robust divergence']:
        if settings['robust divergence type'] == "RAD":
            D = MFN_MFN_RAD(prior_mu, prior_sigma, alpha = hyperparameter,
                            weight = weight)
        elif settings['robust divergence type'] == "AD":
            D = MFN_MFN_AD(prior_mu, prior_sigma, alpha = hyperparameter,
                            weight = weight) 
        elif settings['robust divergence type'] == "ED":
            D = MFN_MFN_ED(prior_mu, prior_sigma, weight = weight)
        elif settings['robust divergence type'] == "FD":
            D = MFN_MFN_FD(prior_mu, prior_sigma, weight = weight)
        elif settings['robust divergence type'] == "JeffreysD":
            D = MFN_MFN_JeffreysD(prior_mu, prior_sigma, 
                    relative_weight = hyperparameter, weight = weight)
        elif settings['robust divergence type'] == "reverseKLD":
            D = MFN_MFN_reverse_KLD(prior_mu, prior_sigma, weight = weight)
    else:
        D = MFN_MFN_KLD(prior_mu, prior_sigma, weight = weight)
            
    
    """STEP 3: Set up the loss function"""
    
    """STEP 3.1: Make loss robust if required"""
    if settings['robust likelihood']:
        gamma = settings['robust likelihood gamma']
    else:
        gamma = None    
    BMMLossFixedSig = BMMLogLossFixedSigma(d,num_clusters,n,gamma)
    
    """STEP 3.2: Set up Black Box GVI object"""
    BBGVI_obj = BBGVI(D=D, L=BMMLossFixedSig)
    
    """STEP 3.3: Optimization/Inference"""
    if opt_settings['data subsample size'] is None:
        data_samples = n
    else:
        data_samples = opt_settings['data subsample size']
    
    """STEP 3.4: Optimization/Inference"""
    np.random.seed(opt_settings['optimization seed'])
    BBGVI_obj.fit_q(Y=Y,X=None, 
               epochs = opt_settings['epochs'], 
               learning_rate = opt_settings['learning rate'], 
               K=opt_settings['num_samples'], 
               M = data_samples)
    
    """STEP 3.5: Print outcomes and return result"""
    if settings['print outcomes']:
        #check that output is sensible
        BBGVI_obj.report_parameters()
    return BBGVI_obj
  

"""Take in the output of the BMM_experiment function (a BBGVI object) and save 
the results to the hard drive under the relevant path"""    
def save_GVI_BMM_results(BBGVI_obj, path_to_save, file_name, file_id):
    
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
    
    #Re-arrange ('cluster', 'mean', 'variance')
    result_types = []
    for name in names:
        result_types = result_types + [name.split(' ')[-1]]


    #Save the three parts of the inference into three different files
    for (result_type, res) in zip(result_types, values):
        file_string = (path_to_save + "/" + file_name + result_type + 
                       "_" + file_id + ".txt")
        np.savetxt(file_string, res, fmt='%10.8f')
        
    print("Saved", file_string)
    

def str_to_bool(s):
    if s == 'True':
         return True
    elif s == 'False':
         return False
    

if __name__ == '__main__':
    
    # read in from console    
#    n = int(sys.argv[1])
#    d = int(sys.argv[2])
#    Dtype = str(sys.argv[3])
#    Dhyperparameter = float(sys.argv[4])
#    prior_misspecification = str_to_bool(sys.argv[5])
#    simulation_number = int(sys.argv[6])

    # hard code stuff
    n = 50
    Dtypes = ["AD", "AD", "ED", "FD", "JeffreysD", "reverseKLD"]
    Dhypers = [0.5, 2.0, 0.0, 0.0, 0.0, 0.0]
    d_list = [10, 25, 50, 100, 250]
    model_misspecification = False
    losstype="log"
    losshyperparameter=0.0
    prior_misspecification = True
    
    
    for Dtype, Dhyperparameter in zip(Dtypes, Dhypers):
        for d in d_list:
            for simulation_number in range(0,100):
               
                # path to save stuff
                base_path = ("/Users/jeremiasknoblauch/Documents/OxWaSP/GVI/JASA" + 
                      "/experiments_VARIABLE_DATASET_NEW/BMM1")
                misspec_dir = ("/prior_mis=" + str(prior_misspecification) + 
                               "/model_mis=" + str(model_misspecification) )
                spec_string = (misspec_dir + "/n=" + str(n) + "/d=" + str(d) + 
                               "/D=" + Dtype + "_param=" + str(Dhyperparameter))
                path_to_save = base_path + spec_string
                file_name = ""
                file_id = str(simulation_number) 
                
            #    base=("/Users/jeremiasknoblauch/Documents/OxWaSP/GVI/JASA" + 
            #          "/experiments_VARIABLE_DATASET/BMM1" + 
            #          "/prior_mis=" + str(prior_misspecification) + 
            #          "/model_mis=False" + 
            #          "/n=50" + 
            #          "/d=" + str(d) + 
            #          "/D=" + D + "_param=" + str(Dhyper) + "/")
            #    
            #    path_to_save = base
            #    file_name = ""
            #    file_id = str(simulation_number)
                
                # set up the artificial data for the mixture model
                num_clusters = 2
                mixture_probs = np.array([0.5, 0.5])
                mu_data = np.array([np.array([-2.0]*d),np.array([2.0]*d)])
                sigma2_data = np.array([np.array([1.0]*d),np.array([1.0]*d)])
                seed_data = simulation_number
                
                # reshuffle seed 
                np.random.seed(int(file_id))
                
                # If we use misspecified priors (these are also the default settings)
                if prior_misspecification:
                    prior_mu_mis = -10.0 * np.ones(shape = (d,num_clusters)).flatten()
                    prior_sigma2_mis = 0.1 * np.ones(shape = (d,num_clusters)).flatten()
                else:
                    prior_mu_mis = np.zeros(shape = (d,num_clusters)).flatten()
                    prior_sigma2_mis = 10.0 * np.ones(shape = (d,num_clusters)).flatten()
                    
                # if we use misspecified model
                if model_misspecification:
                    outlier_frequency = 0.05
                    outlier_size = 10.0
                    outlier_variance = 3.0
                else:
                    # this is actually redundant
                    outlier_frequency = 0.0
                    outlier_size = 0.0
                    outlier_variance = 0.00001
                
                #Problem set-up
                settings = {'test plot': False,
                            'print outcomes': False,
                            'outliers': model_misspecification, 
                            'outlier frequency': outlier_frequency,
                            'outlier size': outlier_size, 
                            'outlier variance': outlier_variance,
                            'robust likelihood': (losstype == 'gamma'),
                            'robust likelihood gamma': losshyperparameter,
                            'prior misspecification': prior_misspecification, 
                            'prior misspecification pi variance': prior_sigma2_mis,
                            'prior misspecification pi mean': prior_mu_mis,
                            'robust divergence': (Dtype != 'KLD'), 
                            'robust divergence type': Dtype, 
                            'robust divergence hyperparameter': Dhyperparameter, 
                            'robust divergence weight': 1.0}
                
                #Optimization set-up
                opt_settings = {'learning rate': 0.01, 
                                'epochs': 2000, 
                                'num_samples': 100,
                                'optimization seed': simulation_number,
                                'data subsample size': 32
                                }
                
                #Run everything
                res = BMM_experiment(d, num_clusters, n, mixture_probs, 
                                     mu_data, sigma2_data, seed_data,
                               settings = settings, opt_settings = opt_settings)
                
                #Save results
                save_GVI_BMM_results(res, path_to_save, file_name, file_id)
                
                
                
               
