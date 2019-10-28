#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:33:17 2019

@author: jeremiasknoblauch

Description: Class that generates (Bayesian) Linear Regression data


"""

import numpy as np
import scipy.stats as stats

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
    
    
class BGLMSimulator():
    """This builds an object that can simulate from a Bayesian Generalized
    Linear Regression (BGLM) as in Wang & Blei '18. Internal states:    
            d               dimensionality of X (and coefs)
            X_var           the d-vector of variances with which X is generated
            X_mean          the d-vector of means for X
            intercept       beta_0, i.e. the intercept of the GLM
            coefs           the coefs for the GLM (i.e. beta_0 + X*coefs) 
            sigma2          the variance of the random effects U_i
    """
    
    def __init__(self, d, intercept, coefs, sigma2, X_var, X_mean):
        self.d, self.intercept, self.sigma2 = d, intercept, sigma2
        self.coefs = coefs
        self.X_var, self.X_mean = X_var, X_mean
        if (coefs.shape[0] != d):
            print("ERROR! coefs != d!")
        if ((X_var.shape[0] != d) or (X_var.shape[0] != X_mean.shape[0])):
            print("ERROR! X_var or X_mean do not have dimension d!")
    
    def generate(self,n,seed=None):
        
        """Set seed"""
        if seed is not None:
            np.random.seed(seed)
        
        """Draw X"""
        X = np.random.normal(loc=self.X_mean, scale = np.sqrt(self.X_var), 
                            size = (n, self.d))
        
        """Get random effects:"""
        U = np.random.normal(loc=0.0 ,scale=np.sqrt(self.sigma2), size = n)
        
        """Get Y"""
        lambda_Y = np.exp(self.intercept + X.dot(self.coefs) + U)
        Y = stats.poisson.rvs(mu=lambda_Y, size=n)
        
        return (X, Y, U)
        
    
    
class BMMSimulator():
    """This builds an object that can simulate from a Bayesian Mixture
    model (BMM) as in Wang & Blei '18. Internal states:  
            d              the dimension of X, the data described by the BMM
            mixture_probs  the mixture component probabilities
            mu             the K-vector of mixture means
            sigma2         the K-vector of mixture variances
    """
    
    def __init__(self, d, mixture_probs, mu, sigma2):
        self.d, self.mixture_probs, self.mu, self.sigma2 = d,mixture_probs,mu,sigma2
        if (sigma2.shape[0] != mixture_probs.shape[0] or 
            mu.shape[0] != mixture_probs.shape[0]):
            print("ERROR! sigma2 or mu does not have the same number of" + 
                  " entries as 'mixture_probs'!")
        if d > 1:
            if sigma2.shape[1] != d or mu.shape[1] != d:
                print("ERROR! sigma2 or mu does not have d columns per entry!")
        
    def generate(self,n,seed=None):
        
        """Set seed"""
        if seed is not None:
            np.random.seed(seed)
        
        """Draw X (without mu or sigma2) as normals"""
        X = np.random.normal(loc=np.zeros(self.d), scale = 1.0, size = (n,self.d))
        
        """Draw cluster membership for X"""
        cluster_membership = np.random.choice(len(self.mixture_probs), 
            size = n, replace=True, p = self.mixture_probs)
        
        """Rescale X (i.e., use mu and sigma2)"""
        #row-wise multiplication with the relevant sigma2 elements & 
        #row-wise addition of the relevant mu elements
        for i in range(0, len(self.mixture_probs)):
            cluster_i = (cluster_membership == i)
            X[cluster_i,:] = (
                X[cluster_i,:]*np.sqrt(self.sigma2[i]) + self.mu[i])
        
        return (X, cluster_membership)
        
        
    

#short test if everything works as we want
TEST_BLR = False
TEST_BMM = False
TEST_BGLM = False

if TEST_BLR:
    d = 3
    coefs = np.array([1,2,3])
    X_mean = np.array([0,1,-1])
    X_var = np.array([0.5, 0.5, 0.5])
    sigma = 2.0
    
    BLR_gen = BLRSimulator(d,coefs,X_mean,X_var,sigma)
    (X,Y) = BLR_gen.generate(100000)
            
    print("X mean is", X_mean)
    print("X has empirical column means", np.mean(X,axis=0))
    print("X var is", X_var)
    print("X has empirical column variance", np.var(X,axis=0))
    print("expected value of Y is ", np.dot(coefs, X_mean))
    print("empirical mean of Y is ", np.mean(Y))

        
if TEST_BMM:
    d,mixture_probs = 2, np.array([0.75, 0.25])
    mu,sigma2 =  np.array([[0.0, 0.0],[10.0, 10.0]]), np.array([[0.5, 0.5],[0.05,0.05]])
    BMM_gen = BMMSimulator(d, mixture_probs, mu, sigma2)
    X, clusters = BMM_gen.generate(1000)
 
    #check that this makes sense by plotting
    LABEL_COLOR_MAP = {0 : 'r',
                   1 : 'k'}
    import matplotlib.pyplot as plt
    label_color = [LABEL_COLOR_MAP[l] for l in clusters]

    plt.scatter(X[:,0], X[:,1], c=label_color)
    plt.show()



if TEST_BGLM:
    n=1000
    d = 3
    coefs = np.array([1,2,1])
    X_mean = np.array([0,1,-1])
    X_var = np.array([0.5, 0.5, 0.5])
    sigma2 = 1.0   
    intercept = -2.0 
      
    BGLM_gen = BGLMSimulator(d, intercept, coefs, sigma2, X_var, X_mean)
    (X, Y, U) = BGLM_gen.generate(n)

    plt.plot(Y)
    plt.show()









       
        