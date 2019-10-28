#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:17:11 2019

@author: jeremiasknoblauch

Description: Class that does BB-GVI inference on BLR
    
"""

#from __future__ import absolute_import
#from __future__ import print_function
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


class BBGVI():
    """This builds an object that performs Black Box GVI inference. 
    Internal states of the object:
        D       A divergence object (will work with a fixed prior)
        L       A loss object (determines the parameter wrapping)
        Y       A data vector
        X       A data matrix (optional)
        K       The number of samples drawn from q (default = 100)
        M       The number of samples drawn from (X,Y) (default = max(1000, len(Y)))
        n       = len(Y)
        q_params  The parameters of the MFN that we fit (= q)
        q_parser  The locations/names of the parameters 
    """
    
    def __init__(self, D, L):
        """create the object"""
        
        #DEBUG: Still need to account for case where X = None!
        self.D, self.L = D, L
        
        self.q_parser, self.q_params, self.converter = self.make_q_params() 
          
    def draw_samples(self,q_params):
        return self.L.draw_samples(self.q_parser, q_params, self.K)
          
    def create_GVI_objective(self):
                
        """create the objective function that we want to differentiate"""
        def GVI_objective(q_params, q_parser, converter, Y_, X_, indices):           
            """Y_, X_ will be sub-samples of Y & X"""
            
            """Make sure to re-weight the M data samples by n/M"""
            q_sample = self.draw_samples(q_params)
            
            if False: #Great for debugging
                print("Loss:", np.mean(self.n * self.L.avg_loss(q_sample, Y_, X_, indices)))
                print("Div:", self.D.prior_regularizer(q_params, q_parser, converter))
            
            return (np.mean(self.n * self.L.avg_loss(q_sample, Y_, X_, indices))
                    + self.D.prior_regularizer(q_params, q_parser, converter))
        
        return GVI_objective
        
        
        
    def make_q_params(self):
        """depending on the parameters (i.e., the loss), I need to 
        create a container for the q_params, too!"""
        parser = ParamParser()

        """Give the parser the right entries & names (for BLR)"""
        parser, params, converter = self.L.make_parser(parser)
        return (parser, params, converter)
    
    
    def fit_q(self, Y, X=None, K=100, M = 1000, 
              epochs = 500, learning_rate = 0.0001,):
        """This function puts everything together and performs BB-GVI"""
        
        """STEP 0: Make sure our M is AT MOST as large as data"""
        self.K = K
        self.n = Y.shape[0]
        self.M = min(M, self.n)
        
        """STEP 1: Get objective & take gradient"""
        GVI_obj = self.create_GVI_objective()
        GVI_obj_grad = grad(GVI_obj)
        q_params = self.q_params
        
        
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
                    grad_q_params = GVI_obj_grad(q_params,self.q_parser, self.converter,
                                             Y[ indices ], X[ indices,: ], indices)
                else:
                    grad_q_params = GVI_obj_grad(q_params,self.q_parser, self.converter,
                                             Y[ indices ], X_=None, indices=indices)
                
                
                m1 = beta1 * m1 + (1 - beta1) * grad_q_params
                m2 = beta2 * m2 + (1 - beta2) * grad_q_params**2
                m1_hat = m1 / (1 - beta1**t)
                m2_hat = m2 / (1 - beta2**t)
                q_params -= learning_rate * m1_hat / (np.sqrt(m2_hat) + epsilon)
            
        self.q_params = q_params
                

    def report_parameters(self):
        #get two lists of param names & values
        names, values = self.L.report_parameters(self.q_parser, self.q_params)        
        for i in range(0, len(names)):
            print(names[i], values[i])
        
        
        
        
        
        
        
        