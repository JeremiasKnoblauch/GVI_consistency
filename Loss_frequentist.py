#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 09:44:52 2020

@author: jeremiasknoblauch
"""

"""Description: Loss functions, but for frequentist setting (simpler)"""

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd.scipy.misc import logsumexp
from autograd import grad

import math
import sys


class Loss():
    """Compute loss between data & parameters, create q_parameter objects for
    the BBGVI class etc. Internal states of the object are none because
    it is EMPTY/ABSTRACT.            
    """

    def __init__(self):
        self.losstype = 0
    
    def make_parser(self, parser):
        return 0
        
    def avg_loss(self, params, parser, Y_, X_):
        return 0
    
    def report_parameters(self, params, parser):
        return 0
    
    
class LogLossLinearRegression(Loss):
    """Compute log likelihood loss between data & parameters for Linear Regr.
    Underlying model:
        
        Y ~ X*b + e
    
    Internal states of the object are:
        d           dimension of X (regressor matrix)
    """
    
    def __init__(self, d):
        self.d = d
        
    def make_parser(self, parser):
        """Because we have to supply the entire vector of parameters at 
        once, it makes sense to create a parser object that tells us which
        element of the vector corresponds to which parameter"""
        
        """Create the parser object that tells you which  elements of 'param'
        store which parameters"""        
        parser.add_shape('regression coefficients', (self.d, 1))
        parser.add_shape('log_sigma2', (1,1))
        
        """Initialize the parameters via the parser object"""
        params = 1.0 * np.random.randn(parser.num_params)
        
        params[ parser.get_indexes(params, 'log_sigma2') ] = (
            params[ parser.get_indexes(params, 'log_sigma2') ] + 10.0)
        
        return (parser, params)
        
    def avg_loss(self, params, parser, Y_, X_):
        """produce log likelihood loss of a linear regression"""
        
        """We store sigma2 in log form for computational reasons. here, we 
        convert that back to compute the actual loss itself."""
        sigma2 = np.exp(-parser.get(params, 'log_sigma2')) 
        coefs = parser.get(params, 'regression coefficients')
        
        """X_ is M x d, coef_sample is K x d, so Y_hat will be M x K"""
        Y_hat = np.matmul(X_, coefs) #coefs is  d x 1, X is  M x d
#        
#        print(X_)
#        print(coefs)
#        print(Y_hat)
        
        """Next, get average neg log lklhood"""
        neg_log_lkl = -( 
            - 0.5 * np.log(2 * math.pi * sigma2) 
            - 0.5 * (Y_ - Y_hat)**2 /sigma2
            )
      
        return np.mean(neg_log_lkl) #Division by K, not M!
        
    
    def report_parameters(self, parser, params):
        
        sigma2 = np.exp(-parser.get(params, 'log_sigma2')) 
        coefs = parser.get(params, 'regression coefficients')
        
        param_names = ['sigma2', 'regression coefficients']
        param_values = [sigma2, coefs]
        
        return (param_names, param_values)


class GammaLossLinearRegression(LogLossLinearRegression):
    """Compute log likelihood loss between data & parameters for Linear Regr.
    Underlying model:
        
        Y ~ X*b + e
    
    Internal states of the object are:
        d           dimension of X (regressor matrix)
    """
    
    def __init__(self, d, gamma):
        self.d = d
        self.gamma = gamma
        
        
    def avg_loss(self, params, parser, Y_, X_):
        """produce log likelihood loss of a linear regression"""
        
        """We store sigma2 in log form for computational reasons. here, we 
        convert that back to compute the actual loss itself."""
        sigma2 = np.exp(-parser.get(params, 'log_sigma2')) 
        coefs = parser.get(params, 'regression coefficients')
        
        """X_ is M x d, coef_sample is K x d, so Y_hat will be M x K"""
        Y_hat = np.matmul(X_, coefs)
        
        """Next, get average neg log lklhood"""
        log_lkl = ( 
            - 0.5 * np.log(2 * math.pi * sigma2) 
            - 0.5 * (Y_ - Y_hat)**2 /sigma2
            )
        
        log_Integral = (np.log(sigma2) * (-0.5*1.0*self.d * (self.gamma-1.0)) + 
                        np.log(2*math.pi) * (-0.5*(self.gamma -1.0)) + 
                        np.log(self.gamma) * (-0.5))
        
        L = - (np.log((self.gamma/self.gamma - 1.0)) + 
                (self.gamma-1.0)*log_lkl + 
                (-self.gamma/(self.gamma - 1.0)) * log_Integral)
      
        return np.mean(L) #Division by K, not M!


