#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 14:33:17 2019

@author: jeremiasknoblauch

Description: This file runs some example code to compare GVI 
against standard VI and exact Bayesian inference.


"""
import numpy as np
from simulator import BLRSimulator, BMMSimulator
from BBGVI import BBGVI
from Loss import LogNormalLossBLR, LogLaplaceLossBLR, AbsoluteLoss, LogNormalLossBLRFixedSig, BMMLogLoss
from Loss import BMMLogLossFixedSigma
from Divergence import MFN_MFN_KLD, MFN_MFN_RAD, MFN_MFN_AD, MFN_MFN_FD, MFN_MFN_ED
from Divergence import MFN_MFN_reverse_KLD, MFN_MFN_JeffreysD


"""TEST: Can I set up my BBGVI algorithm?"""
TEST_BLR = True
TEST_ABSOLUTE_LOSS = False
TEST_BMM = False


#Test 1: Simple Likelihoood model (BLR):
if TEST_BLR:
     
    d = 3
    coefs = np.array([1,2,3])
    X_mean = np.array([-10,7.65,-1])
    X_var = np.array([1, 0.25, 2])
    sigma2 = 25
    
    np.random.seed(0)
    BLR_gen = BLRSimulator(d,coefs,X_mean,X_var,sigma2=sigma2)
    (X,Y) = BLR_gen.generate(1000, seed = 0)
            
    print("Coefs are", coefs)
    print("sigma2 is", sigma2)
    print("X mean is", X_mean)
    print("X var is", X_var)
    print("X has column means", np.mean(X,axis=0))
    print("X has column variance", np.var(X,axis=0))
    print("expected value of Y is ", np.dot(coefs, X_mean))
    print("actual meean of Y is ", np.mean(Y))
    print("LS estimator is", np.linalg.solve(a=np.transpose(X).dot(X),b=np.transpose(X).dot(Y)))
              
    KL = True
    AD = False
    RAD = False
    FD = False     
    
    #Initialize my loss
    fixed_sigma = False
    if not fixed_sigma:
        BLRLoss = LogNormalLossBLR(d)
        d_ = d+1
    elif fixed_sigma:
        BLRLoss = LogNormalLossBLRFixedSig(d)
        d_ = d
        
    #Set up the prior + regularizer
    if KL:
        Div = MFN_MFN_KLD(0*np.ones(d_), 10*np.ones(d_)) 
    if RAD:
        Div2 = MFN_MFN_RAD(0*np.ones(d_), 10*np.ones(d_), alpha = 10.5 )
    if FD:
        Div3 = MFN_MFN_FD(0*np.ones(d_), 10 *np.ones(d_), weight = 0.001)
    if AD:
        Div4 = MFN_MFN_AD(0*np.ones(d_), 10*np.ones(d_), alpha = 0.5 ) #sq. Hellinger
    
    #Set up BBGVI (With KLD reg)
    if KL:
        BBVI = BBGVI(D=Div, L=BLRLoss)
    
    #Set up two GVI methods (with renyi-alpha div & fisher div)
    if RAD:
        BBGVI1 = BBGVI(D=Div2, L=BLRLoss)
    if FD:
        BBGVI2 = BBGVI(D=Div3, L=BLRLoss)
    if AD:
        BBGVI3 = BBGVI(D=Div4, L=BLRLoss)
            
    
    """TEST 3: Can I run it?"""
    np.random.seed(0)
    if KL:
        BBVI.fit_q(Y=Y,X=X, epochs = 500, learning_rate = 0.1, K=100, M = 100)
    
    np.random.seed(0)
    if RAD:
        BBGVI1.fit_q(Y=Y,X=X, epochs = 500, learning_rate = 0.1, K=100, M = 100)
        
    np.random.seed(0)
    if FD:
        BBGVI2.fit_q(Y=Y,X=X, epochs = 500, learning_rate = 0.1, K=100, M = 100)
        
    np.random.seed(0)
    if AD:
        BBGVI3.fit_q(Y=Y,X=X, epochs = 500, learning_rate = 0.1, K=100, M = 100)
    
    """TEST 4: Does it produce something sensible?"""
    if KL:
        BBVI.report_parameters()
    if RAD:
        BBGVI1.report_parameters()
    if FD:
        BBGVI2.report_parameters()
    if AD:
        BBGVI3.report_parameters()

if TEST_BMM:
    #first generate my data
    d = 10
    K = 2
    n = 50
    mixture_probs=np.array([0.33, 0.33, 0.34])
    #mu,sigma2 =  np.array([[-10.0, -1.0],[10.0, 1.0]]), np.array([[0.05, 0.05],[0.01,0.01]])
    #mu = np.array([[12.0, 12.0], [-12.0, -12.0], [1.0,1.0]])
    #sigma2 = np.array([[1.0, 1.0], [1.0,1.0], [1.0,1.0]])
    mu,sigma2 =  np.array([[-2.0]*d,[2.0]*d]), np.array([[1.0]*d,[1.0]*d])
    mu,sigma2 =  np.array([np.array([-2.0]*d),np.array([2.0]*d)]), np.array([np.array([1.0]*d),np.array([1.0]*d)])
    mixture_probs = np.array([0.5, 0.5])
    BMM_gen = BMMSimulator(d, mixture_probs, mu, sigma2)
    Y, clusters = BMM_gen.generate(n, seed = 0)
    Y.reshape(n,d)
    
    corrupt_Y = True
    robust_likelihood = False
    prior_misspecification = False
    robust_divergence = True
    
    #INTERESTING SETTING: alpha = 0.5 (standard renyi-alpha) + n = 25 + 
    #                       strong prior at one cluster (correct for one cluster, but not for second)
    #                     really interesting for var=0.01 (exaggerated setting of course)
    
    if corrupt_Y:
        #Do we want to add noise (& perform robust vs non-robust comparison)?
        corruption_prob = 0.1
        corruption = 10 + 3 * np.random.normal(0,1,(n,d))
        rand_ind = np.random.choice(n, int(n * corruption_prob), replace=False)
        
        #corrupt the Y-data and set gamma
        Y[rand_ind,:] = Y[rand_ind,:]  + corruption[rand_ind,:]
    
    if robust_likelihood:
        gamma = 1.01
    else:
        gamma = None
    
    
    #number of parameters
    alpha = 0.5
    if prior_misspecification:
        var = 20.0
        mu  = np.array([-10]*d*K)
        if robust_divergence:
            #NOTE: We have the standard parameterization here!
            Div = MFN_MFN_ED(mu, var*np.ones(d * K), weight = 1.0) 
            Div = MFN_MFN_reverse_KLD(mu, var * np.ones(d*K))
            Div = MFN_MFN_RAD(mu, var*np.ones(d * K), alpha = alpha) 
            #Div = MFN_MFN_FD(mu, var*np.ones(d * K), weight = 1.0) 
        else:
            Div = MFN_MFN_KLD(mu, var*np.ones(d * K)) 
    else:
        var = 100.0
        mu  = np.array([0.0]*d*K)
        if robust_divergence:
            #NOTE: We have the standard parameterization here!            
            Div = MFN_MFN_ED(mu, var*np.ones(d * K), weight = 1.0) 
            Div = MFN_MFN_reverse_KLD(mu, var * np.ones(d*K))
            Div = MFN_MFN_JeffreysD(mu, var * np.ones(d*K))
            Div = MFN_MFN_RAD(mu, var*np.ones(d * K), alpha = alpha) 
        else:
            Div = MFN_MFN_KLD(mu, var*np.ones(d * K)) 
 
    #check that this makes sense by plotting
    LABEL_COLOR_MAP = {0 : 'r',
                   1 : 'k', 
                   2 : 'b'}
    import matplotlib.pyplot as plt
    label_color = [LABEL_COLOR_MAP[l] for l in clusters]

    if d < 3:
        plt.scatter(Y[:,0], Y[:,1], c=label_color)
        
    else:
        
        #Initialize plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
        #Plot the (three) clusters
        x,y,z = Y[:,0], Y[:,1], Y[:,2]
        ax.scatter(x,y,z,marker = 'o',c=label_color)
    plt.show()

    #Set up GVI problem
    BMMLoss = BMMLogLoss(d, K, n)
    BMMLossFixedSig = BMMLogLossFixedSigma(d,K,n,gamma)
    
       
    BBVI = BBGVI(D=Div, L=BMMLossFixedSig)
    
    np.random.seed(1)
    if True:
        BBVI.fit_q(Y=Y,X=None, epochs = 2000, learning_rate = 0.1, K=100, M = n)
    
    #check that output is sensible
    BBVI.report_parameters()


    
#Test 2: Likelihood-free model
if TEST_ABSOLUTE_LOSS:
    
    d = 1
    coefs = np.array([1.0])
    X_mean = np.array([-8])
    X_var = np.array([0.000001]) #i.e., X=-8, meaning Y = -8 + error, i.e. we look at median
    sigma2 = (10**2)
    
    np.random.seed(0)
    BLR_gen = BLRSimulator(d,coefs,X_mean,X_var,sigma2=sigma2)
    (X,Y) = BLR_gen.generate(300)
            
    print("Coefs are", coefs)
    print("sigma2 is", sigma2)
    print("X mean is", X_mean)
    print("X var is", X_var)
    print("X has column means", np.mean(X,axis=0))
    print("X has column variance", np.var(X,axis=0))
    print("expected value of Y is ", np.dot(coefs, X_mean))
    print("actual meean of Y is ", np.mean(Y))
    print("LS estimator is", np.linalg.solve(a=np.transpose(X).dot(X),b=np.transpose(X).dot(Y)))
        
    #Initialize my loss
    loss = AbsoluteLoss(d)
    
    #Set up the prior + regularizer
    Div = MFN_MFN_KLD_fixedSigma(0*np.ones(d), (100**2)*np.ones(d)) 
    Div2 = MFN_MFN_RAD_fixedSigma(0*np.ones(d), (100**2)*np.ones(d), alpha = 10.5 )
    Div3 = MFN_MFN_FD_fixedSigma(0*np.ones(d), (100**2)*np.ones(d), weight = 0.0001)
    Div4 = MFN_MFN_AD_fixedSigma(0*np.ones(d), (100**2)*np.ones(d), alpha = 0.5 ) #Hellinger
    
    #Set up BBGVI (With KLD reg)
    BBVI = BBGVI(D=Div, L=loss)
    
    #Set up two GVI methods (with renyi-alpha div & fisher div)
    BBGVI1 = BBGVI(D=Div2, L=loss)
    BBGVI2 = BBGVI(D=Div3, L=loss)
    BBGVI3 = BBGVI(D=Div4, L=loss)
            
    
    """TEST 3: Can I run it?"""
    np.random.seed(0)
    if True:
        BBVI.fit_q(Y=Y,X=X, epochs = 500, learning_rate = 0.1, K=100, M = 100)
    
    np.random.seed(0)
    if True:
        BBGVI1.fit_q(Y=Y,X=X, epochs = 500, learning_rate = 0.1, K=100, M = 100)
        
    np.random.seed(0)
    if True:
        BBGVI2.fit_q(Y=Y,X=X, epochs = 500, learning_rate = 0.1, K=100, M = 100)
    
    np.random.seed(0)
    if True:
        BBGVI3.fit_q(Y=Y,X=X, epochs = 500, learning_rate = 0.1, K=100, M = 100)
    
    """TEST 4: Does it produce something sensible?"""
    BBVI.report_parameters()
    BBGVI1.report_parameters()
    BBGVI2.report_parameters()
    BBGVI3.report_parameters()
    
    
    
    
    
