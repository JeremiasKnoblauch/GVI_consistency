#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:33:17 2019

@author: jeremiasknoblauch

Description: Class that generates (Bayesian) Linear Regression data


"""

import numpy as np

class BLRSimulator():
    """This build an object that can simulate from Bayesian Linear
    Regression (BLR). Internal states of the object:
        coefs           the coefficients b in Y = Xb + e
        X_var           the variance with which X is generated
        X_mean          the means of X
        d               the dimension of coefs/X.
    """
    
    """Give the dimensions and what the X/error looks like"""
    def __init__(self, d, coefs, X_mean = None, X_var = None, sigma2 = None):
        self.d, self.coefs, self.sigma2 = d,coefs,sigma2
        self.X_mean, self.X_var = X_mean, X_var
        #ADD: Error check -- is d same dim as coefs?
        #ADD: If X_mean, X_var, sigma are None, set them to 0, 1, 0 (vecs)
        
        # Jeremias' shitty testing hack
        varbool = True
        meanbool = True
        
        if X_mean is None:
            self.X_mean = np.zeros(d)
        else:
            meanbool = (X_mean.shape[0] == d)
        if X_var is None:
            self.X_var = np.ones(d)
        else:
            varbool = (X_var.shape[0] == d)
        if sigma2 is None:
            self.sigma2 = 1.0
        
        if (not varbool) or (not meanbool):
            print("ERROR! dimension of X_mean or X_var not equal to d!")
            
            
        
    """Produce a sample of size n"""
    def generate(self,n, seed = None):
        
        """Set seed"""
        if seed is not None:
            np.random.seed(seed)
        
        """column c of X: X_mean[c-1]-mean normals with X_var[c-1] variance"""
        X = np.random.normal(loc=self.X_mean, scale = np.sqrt(self.X_var), 
                            size = (n, self.d))
        
        """Get errors"""
        e = np.random.normal(loc=0,scale=np.sqrt(self.sigma2), size = n)
        
        """Use self.coefs and errors to produce Y"""
        Y = X.dot(self.coefs) + e
        
        """return (X,Y)"""
        return (X, Y)
    

#short test if everything works as we want
TEST = True
if TEST:
    d = 3
    coefs = np.array([1,2,3])
    X_mean = np.array([0,1,-1])
    X_var = np.array([0.5, 0.5, 0.5])
    sigma = 2.0
    
    BLR_gen = BLRSimulator(d,coefs,X_mean,X_var,sigma)
    (X,Y) = BLR_gen.generate(100000)
    print(X.shape)
            
    print("Coefs are", coefs)
    print("X mean is", X_mean)
    print("X var is", X_var)
    print("X has column means", np.mean(X,axis=0))
    print("X has column variance", np.var(X,axis=0))
    print("expected value of Y is ", np.dot(coefs, X_mean))
    print("actual mean of Y is ", np.mean(Y))
        
        
        