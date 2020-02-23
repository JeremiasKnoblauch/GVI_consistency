#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 09:10:36 2020

@author: jeremiasknoblauch
"""

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad

import sys

import math



"""AUXILIARY OBJECT. Purpose is to provide wrapper for all q_params"""
class ParamParser(object):
    """A helper class to index into a parameter vector."""
    def __init__(self):
        self.idxs_and_shapes = {}
        self.num_params = 0

    def add_shape(self, name, shape):
        start = self.num_params
        self.num_params += np.prod(shape)
        self.idxs_and_shapes[ name ] = (slice(start, self.num_params), shape)

    def get(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return np.reshape(vect[ idxs ], shape)

    def get_indexes(self, vect, name):
        idxs, shape = self.idxs_and_shapes[name]
        return idxs



class SGD():
    """This class computes the frequentist estimate of a gamma-divergence based
    scoring rule, given data for a model & a model.
    
        L       Loss function to be optimized
        X       Input data (optional)
        Y       Output data 
    
    """
    
    def __init__(self, L, Y, X=None):
        self.L = L        
        self.X = X
        self.Y = Y
        self.n = Y.shape[0]
        
        self.parser, self.params = self.make_params() 
    
    
    def make_params(self):
        """depending on the parameters (i.e., the loss), I need to 
        create a container for the params, too!"""
        parser = ParamParser()

        """Give the parser the right entries & names (for BLR)"""
        parser, params = self.L.make_parser(parser)
        return (parser, params)
        
          
    def create_objective(self):
                
        """create the objective function that we want to differentiate"""
        def objective(params, parser, Y_, X_, indices):           
            """Y_, X_ will be sub-samples of Y & X"""
            
            return (np.mean(self.n * self.L.avg_loss(params, parser, 
                                                     Y_, X_, indices)))
        
        return objective
        
        
    def fit(self, batch_size, epochs=500, learning_rate = 0.0001):
        
        """STEP 1: Set up what the optimization routine will be"""
        
        """Just to streamline with GVI code, re-name variables"""
        self.M = min(batch_size, self.n)
        Y = self.Y
        X = self.X
        
        """Create objective & take gradient"""
        objective = self.create_objective()
        objective_gradient = grad(objective)
        params = self.params
               
    
        """STEP 2: Sample from X, Y and perform ADAM steps"""

        """STEP 2.1: These are just the ADAM optimizer default settings"""  
        m1 = 0
        m2 = 0
        beta1 = 0.9
        beta2 = 0.999
        epsilon = 1e-8
        t = 0
    
        """STEP 2.2: Loop over #epochs and take step for each subsample"""
        for epoch in range(epochs):
            
            """STEP 2.2.1: For each epoch, shuffle the data"""
            permutation = np.random.choice(range(Y.shape[ 0 ]), Y.shape[ 0 ], 
                                           replace = False)

            """HERE: Should add a print statement here to monitor algorithm!"""
            if epoch % 100 == 0:
                print("epoch #", epoch, "/", epochs)
                #print("sigma2", np.exp(-q_params[3]))
            
            """STEP 2.2.2: Process M data points together and take one step"""
            for i in range(0, int(self.n/self.M)):
                
                """Get the next M observations (or less if we would run out
                of observations otherwise)"""
                end = min(self.n, (i+1)*self.M)
                indices = permutation[(i*self.M):end]
                
                
                """ADAM step for this batch"""
                t+=1
                if X is not None:
                    grad_params = objective_gradient(params,self.parser, 
                                        Y[ indices ], X[ indices,: ], indices)
                else:
                    grad_params = objective_gradient(params,self.parser, 
                                        Y[ indices ], X_=None, indices=indices)
                
                
                m1 = beta1 * m1 + (1 - beta1) * grad_params
                m2 = beta2 * m2 + (1 - beta2) * grad_params**2
                m1_hat = m1 / (1 - beta1**t)
                m2_hat = m2 / (1 - beta2**t)
                params -= learning_rate * m1_hat / (np.sqrt(m2_hat) + epsilon)
            
        self.params = params
        
        
    def report_parameters(self):
        #get two lists of param names & values
        names, values = self.L.report_parameters(self.parser, self.params)        
        for i in range(0, len(names)):
            print(names[i], values[i])      
            


if True:
    """test if stuff works"""
    
    """get a simulator for BLRs"""
    from BLRSimulator import BLRSimulator
    
    """set up the simulation params"""
    coefs = np.array([1.0,-2.0, 0.5, 4.0, -3.5])
    d= len(coefs)
    sigma2 = 15.0
    n = 1000
    
    """create simulation object + simulated variables"""
    sim =  BLRSimulator(d, coefs, sigma2=sigma2)
    X, Y = BLRSimulator.generate(n, seed = 0)
    
    """create SGD object + run SGD"""
    from Loss_frequentist import LogLossLinearRegression
    StandardLoss = LogLossLinearRegression(d)
    
    optimization_object = SGD(StandardLoss, Y, X)
    
    
    
    
    
    
    
    
    
    
        
        