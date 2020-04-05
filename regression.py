#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:33:17 2019

@author: laravomfell

Description: class that is just a linear regression but holds dimensionality info


"""

import numpy as np

class lara_reg():
    """This builds an objects with a bit of info on the regression
    """
    
    """Give the dimensions and what the X/error looks like"""
    def __init__(self, d, coefs, X, y, sigma2 = None):
        self.d, self.coefs, self.sigma2 = d,coefs,sigma2
        self.X, self.y = X, y
        self.n = y.shape[0]    
    

#short test if everything works as we want
TEST = True
if TEST:
    d = 2
    coefs = np.array([0,2])
    sigma2 = 1.0
    n = 1000
    X = np.array([np.ones(n), np.random.normal(1, 0, n)]).T
    e = np.random.normal(loc=0,scale=np.sqrt(sigma2), size = n)
    y = X.dot(coefs) + e
   
    
    test = lara_reg(d, coefs, X, y, sigma2)    
    print("Dimensions", test.d)