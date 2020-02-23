#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 16:17:11 2019

@author: jeremiasknoblauch

Description: Class that does BB-GVI inference on BLR
    
"""

from __future__ import absolute_import
from __future__ import print_function
import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad

import sys

import math


"""AUXILIARY OBJECT. Purpose is to provide wrapper for all params"""
class WeightsParser(object):
    """A helper class to index into a parameter vector."""
    def __init__(self):
        self.idxs_and_shapes = {}
        self.num_weights = 0

    def add_shape(self, name, shape):
        start = self.num_weights
        self.num_weights += np.prod(shape)
        self.idxs_and_shapes[ name ] = (slice(start, self.num_weights), shape)

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
        params  The parameters of the MFN that we fit (= q)
    """
    
    def __init__(self, D, L, Y, K=100, M = 1000, X=None):
        """create the object"""
        self.D, self.L, self.Y, self.X = D, L, Y, X
        self.K = K
        self.params = self.make_params() #this is a parser object
        self.n = len(self.Y)
        self.M = max(M, self.n)
        
        
    def draw_samples(self):
        return ( npr.randn(self.K, len(self.params[ 'm' ])) * 
                np.sqrt(self.params[ 'v' ]) + self.params[ 'm' ])
        
    
    def create_GVI_objective(self):
                
        """create the objective function that we want to differentiate"""
        def GVI_objective(params, Y_, X_):           
            """Y_, X_ will be sub-samples of Y & X"""
            
            """Make sure to re-weight the M data samples by n/M"""
            return ((self.n/self.M) * self.L.avg_loss(params, Y_, X_)
                    + self.D.prior_regularizer(params))
        
        return GVI_objective
        
        
        
    def make_params(self):
        """depending on the parameters (i.e., the loss), I need to 
        create a container for the params, too!"""
        parser = WeightsParser()
#        parser.add_shape('mean', (N, 1))
#        parser.add_shape('log_variance', (N, 1))
#        parser.add_shape('log_v_noise', (1, 1))
#    
#        w = 0.1 * np.random.randn(parser.num_weights)
#        w[ parser.get_indexes(w, 'log_variance') ] = w[ parser.get_indexes(w, 'log_variance') ] - 10.0
#        w[ parser.get_indexes(w, 'log_v_noise') ] = np.log(1.0)
        
        """Give the parser the right entries & names (for BLR)"""
        parser = self.L.make_parser(self.Y, self.X, parser)
        return parser
    
    
    def fit_q(self, D, L, Y, K=100, M = 1000, X=None,
              epochs = 500, learning_rate = 1e-2,):
        """This function puts everything together and performs BB-GVI"""
        
        """STEP 1: Get objective & take gradient"""
        GVI_obj = self.create_GVI_objective()
        GVI_obj_grad = grad(GVI_obj)
        
        
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
            
            """STEP 2.2.2: Process M data points together and take one step"""
            for i in range(0, int(self.n/self.M + 1)):
                
                """Get the next M observations (or less if we would run out
                of observations otherwise)"""
                end = min(self.n-1, (i+1)*self.M)
                indices = permutation[(i*self.M):end]
                
                """ADAM step for this batch"""
                t+=1
                grad_params = GVI_obj_grad(self.params,Y[ indices ], X[ indices ])
                
                m1 = beta1 * m1 + (1 - beta1) * grad_params
                m2 = beta2 * m2 + (1 - beta2) * grad_params**2
                m1_hat = m1 / (1 - beta1**t)
                m2_hat = m2 / (1 - beta2**t)
                self.params -= learning_rate * m1_hat / (np.sqrt(m2_hat) + epsilon)
                
        
        
        
        
        
        
        
        
        
        