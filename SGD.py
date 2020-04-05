#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 09:10:36 2020

@author: jeremiasknoblauch
"""

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.special import logsumexp
from autograd import grad

import numpy

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
        def objective(params, parser, Y_, X_):           
            """Y_, X_ will be sub-samples of Y & X"""
            
            return (np.mean(self.L.avg_loss(params, parser, 
                                            Y_, X_)))
        
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
                    if False:
                        print("Y", Y[indices])
                        print("X*coefs", np.matmul(X[indices,:], np.array([1.0,-2.0, 0.5, 4.0, -3.5])))
                        print("X*params", np.matmul(X[indices,:], params[:-1]))
                    
                    grad_params = objective_gradient(params,self.parser, 
                                        Y[ indices ], X[ indices,: ])
                else:
                    grad_params = objective_gradient(params,self.parser, 
                                        Y[ indices ], X_=None)
                
#                print(grad_params)
#                print("before:", params)
                m1 = beta1 * m1 + (1 - beta1) * grad_params
                m2 = beta2 * m2 + (1 - beta2) * grad_params**2
                m1_hat = m1 / (1 - beta1**t)
                m2_hat = m2 / (1 - beta2**t)
                params -= learning_rate * m1_hat / (np.sqrt(m2_hat) + epsilon)
#                print("after", params)

            
        self.params = params
        
        
    def report_parameters(self):
        #get two lists of param names & values
        names, values = self.L.report_parameters(self.parser, self.params)        
        for i in range(0, len(names)):
            print(names[i], values[i])      
            


if True:
    """test if stuff works"""
    
    n = 10000
    
    u, v = numpy.random.multivariate_normal([0, 0],
                                            [[1, 0.8], [0.8, 1]],
                                            n).T
    
    z = numpy.random.normal(2, 1, n)
    X = np.array([np.ones(n), 1 * z + v]).T
    d = X.shape[1]
    y_coefs = [0, 0.5]
    y = X.dot(y_coefs) + u

    
    """create SGD object + run SGD"""
    from Loss_frequentist import LogLossLinearRegression
    StandardLoss = LogLossLinearRegression(d)
    
    
    """THIS DOES NOT WORK! WHY?!?!"""
    optimization_object = SGD(StandardLoss, y, X)
    
    optimization_object.fit(batch_size=1,epochs=30,learning_rate=0.1)
    
    # this is pretty bad. alpha = -.79, beta = .8
    optimization_object.report_parameters()
    
    
    """create SGD object + run SGD for robust version"""
    from Loss_frequentist import GammaLossLinearRegression
    
    GammaLoss = GammaLossLinearRegression(d, gamma = 1.1)
    
    
    """THIS DOES NOT WORK! WHY?!?!"""
    optimization_object = SGD(GammaLoss, y, X)
    
    optimization_object.fit(batch_size=1,epochs=50,learning_rate=0.01)
    
    optimization_object.report_parameters()
    
    
    # That would be the inconsistent thing to do. 
    # instead, let's do the correct IV thing
    
    # First, we would need to do y on z
    y_on_z = SGD(StandardLoss, y, z)
    # this doesn't work becasue z is 1-dimensional
    y_on_z.fit(batch_size = 1, epochs = 50, learning_rate = 0.01)
    y_on_z.report_parameters()
    
    # and then, in second step, do x on z
    x_on_z = SGD(StandardLoss, X, z)
    x_on_z.fit(batch_size = 1, epochs = 50, learning_rate = 0.01)
    x_on_z.report_parameters()
    
    # and then we need to report b_y_on_z / b_x_on_z
    
    
    
    
    
        
        