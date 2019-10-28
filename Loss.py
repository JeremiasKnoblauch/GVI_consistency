#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 18:49:30 2019

@author: jeremiasknoblauch

Description: Loss objects (generic superclass and a few special cases)

"""

#from __future__ import absolute_import
#from __future__ import print_function
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
    
    def get_num_global_params(self):
        return 0 #return number of variables we need posterior for
    
    def draw_samples(self, q_parser, q_params, K):
        return 0
        
    def avg_loss(self, params, parser, Y_, X_):
        return 0
    
    def report_parameters(self, params, parser):
        return 0
    
        
    
class LogNormalLossBLR(Loss):
    """Compute log likelihood loss between data & parameters (NOT q_params,
    but model params), create q_parameter objects for the BBGVI class etc. 
    Internal states of the object are:
        d           dimension of X (regressor matrix)
    """
    
    def __init__(self, d):
        self.d = d
    
    def make_parser(self, parser):
        
        """Create the parser object"""       
        parser.add_shape('mean', (self.d+1, 1))
        parser.add_shape('log_variance', (self.d+1,1))
        
        """Initialize the parser object"""
        params = 0.1 * np.random.randn(parser.num_params)        
        params[ parser.get_indexes(params, 'log_variance') ] = (
            params[ parser.get_indexes(params, 'log_variance') ] + 10.0)
        
        log_conversion = np.zeros((self.d+1, 1), dtype=bool)
        log_conversion[-1:,0] = True
        
        #Make sure to set the initial noise low!
        params[ parser.get_indexes(params, 'mean') ][-1] = ( -np.log(0.1) )

        return (parser, params, log_conversion)
    
    
    def draw_samples(self, q_parser, q_params, K):
        
        """Use parser & params to produce K samples of the parameters for which
        q produces a posterior distribution. Do this row-wise."""

        v_c = np.exp(-q_parser.get(q_params, 'log_variance'))[ :-1, 0 ]
        m_c = q_parser.get(q_params, 'mean')[ :-1, 0 ] 

        coef_sample = npr.randn(K, self.d) * np.sqrt(v_c) + m_c
        
        v_sigma = np.exp(-q_parser.get(q_params, 'log_variance'))[ -1, 0 ]
        m_sigma = np.exp((-q_parser.get(q_params, 'mean')))[ -1, 0 ] 
        
        sigma2_sample = npr.randn(K, 1) * np.sqrt(v_sigma) + m_sigma
        
        
        return (coef_sample, sigma2_sample)
        
        
    def avg_loss(self, q_sample, Y_, X_, indices):
        """Retrieve the parameters (using the parser) and compute the 
        average negative log likelihood across the q_sample as well as 
        the X/Y sub-sample"""
        
        """Note 1: The q_sample is a sample from the coefficients b in 
                
                Y = X * b + e
        
        And so we compute for each b-sample the term X * b = Y_hat"""
        coef_sample = q_sample[0]
        K = coef_sample.shape[0]
        M = len(Y_)
        sigma2_sample = q_sample[1].reshape(K)
        
        """X_ is M x d, coef_sample is K x d, so Y_hat will be M x K"""
        Y_hat = np.matmul(X_, np.transpose(coef_sample))
        
        """Next, get average neg log lklhood"""
        neg_log_lkl = -( 
            - 0.5 * np.log(2 * math.pi * (sigma2_sample)) 
            - 0.5 * (np.tile(Y_.reshape(M,1), (1, K)) - Y_hat)**2 /sigma2_sample
            )
        
        return np.mean(neg_log_lkl, 0) #Division by K, not M!
    

    def report_parameters(self, q_parser, q_params):
        
        v_c = np.exp(-q_parser.get(q_params, 'log_variance'))[ :-1, 0 ]
        m_c = q_parser.get(q_params, 'mean')[ :-1, 0 ] 
        
        v_sigma = np.exp(-q_parser.get(q_params, 'log_variance'))[ -1, 0 ]
        m_sigma = np.exp((-q_parser.get(q_params, 'mean')))[ -1, 0 ] 
        
        
        param_names = ['coefficient means', 'coefficient variances', 
                       'sigma2 mean', 'sigma2 variance']
        param_values = [m_c, v_c, m_sigma, v_sigma]
        
        return (param_names, param_values)


class LogNormalLossBLRFixedSig(Loss):
    """Compute log likelihood loss between data & parameters (NOT q_params,
    but model params), create q_parameter objects for the BBGVI class etc. 
    Internal states of the object are:
        d           dimension of X (regressor matrix)
    """
    
    def __init__(self, d):
        self.d = d
    
    def make_parser(self, parser):
        
        """Create the parser object"""
        
        parser.add_shape('mean', (self.d, 1))
        parser.add_shape('log_variance', (self.d,1))
        parser.add_shape('log_sigma2', (1,1))
        
        log_conversion = None
        
        """Initialize the parser object"""
        params = 0.1 * np.random.randn(parser.num_params)
        
        params[ parser.get_indexes(params, 'log_variance') ] = (
            params[ parser.get_indexes(params, 'log_variance') ] + 10.0)
        params[ parser.get_indexes(params, 'log_sigma2') ] = (
            params[ parser.get_indexes(params, 'log_sigma2') ]  -np.log(0.1) )
        
        return (parser, params, log_conversion)
    
    
    def draw_samples(self, q_parser, q_params, K):
        
        """Use parser & params to produce K samples of the parameters for which
        q produces a posterior distribution. Do this row-wise."""
        
        v_c = np.exp(-q_parser.get(q_params, 'log_variance'))[ :, 0 ]
        m_c = q_parser.get(q_params, 'mean')[ :, 0 ] 
        coef_sample = npr.randn(K, self.d) * np.sqrt(v_c) + m_c
        
        v_sigma = np.exp(-q_parser.get(q_params, 'log_sigma2')) 
        
        return (coef_sample, v_sigma)
        
        
    def avg_loss(self, q_sample, Y_, X_, indices):
        """Retrieve the parameters (using the parser) and compute the 
        average negative log likelihood across the q_sample as well as 
        the X/Y sub-sample"""
        
        """Note 1: The q_sample is a sample from the coefficients b in 
                
                Y = X * b + e
        
        And so we compute for each b-sample the term X * b = Y_hat"""
        coef_sample = q_sample[0]
        v_sigma = q_sample[1]
        K = coef_sample.shape[0]
        M = len(Y_)
        
        """X_ is M x d, coef_sample is K x d, so Y_hat will be M x K"""
        Y_hat = np.matmul(X_, np.transpose(coef_sample))
        
        """Next, get average neg log lklhood"""
        neg_log_lkl = -( 
            - 0.5 * np.log(2 * math.pi * (v_sigma)) 
            - 0.5 * (np.tile(Y_.reshape(M,1), (1, K)) - Y_hat)**2 /v_sigma
            )
      
        return np.mean(neg_log_lkl, 0) #Division by K, not M!
    
    def report_parameters(self, q_parser, q_params):
        
        v_c = np.exp(-q_parser.get(q_params, 'log_variance'))[ :, 0 ]
        m_c = q_parser.get(q_params, 'mean')[ :, 0 ] 
        
        v_sigma = np.exp(-q_parser.get(q_params, 'log_sigma2'))
        
        
        param_names = ['coefficient means', 'coefficient variances', 
                       'sigma2 point estimate']
        param_values = [m_c, v_c, v_sigma]
        
        return (param_names, param_values)



class LogLaplaceLossBLR(LogNormalLossBLR):
    """Compute log likelihood loss between data & parameters (NOT q_params,
    but model params), create q_parameter objects for the BBGVI class etc. 
    Internal states of the object are:
        d           dimension of X (regressor matrix)
    """
               
    def avg_loss(self, q_sample, Y_, X_, indices):
        """Retrieve the parameters (using the parser) and compute the 
        average negative log likelihood across the q_sample as well as 
        the X/Y sub-sample"""
        
        """Note 1: The q_sample is a sample from the coefficients b in 
                
                Y = X * b + e
        
        And so we compute for each b-sample the term X * b = Y_hat"""
        coef_sample = q_sample[0]
        K = coef_sample.shape[0]
        M = len(Y_)
        sigma2_sample = q_sample[1].reshape(K)
        
        """X_ is M x d, coef_sample is K x d, so Y_hat will be M x K"""
        Y_hat = np.matmul(X_, np.transpose(coef_sample))
        
        """Next, get average neg log lklhood"""
        neg_log_lkl = -( 
            - 0.5 * np.log(2 * math.pi * (sigma2_sample)) 
            - 0.5 * np.abs(np.tile(Y_.reshape(M,1), (1, K)) - Y_hat) /sigma2_sample
            )
        
        return np.mean(neg_log_lkl, 0) #Division by K, not M!


class AbsoluteLoss(Loss):
    """Compute log likelihood loss between data & parameters (NOT q_params,
    but model params), create q_parameter objects for the BBGVI class etc. 
    Internal states of the object are:
        d           dimension of X (regressor matrix)
    """
               
    def __init__(self, d):
        self.d = d
    
    def make_parser(self, parser):
        
        """Create the parser object"""       
        parser.add_shape('mean', (self.d, 1))
        parser.add_shape('log_variance', (self.d,1))
        
        """Initialize the parser object"""
        params = 0.1 * np.random.randn(parser.num_params)        
        params[ parser.get_indexes(params, 'log_variance') ] = (
            params[ parser.get_indexes(params, 'log_variance') ] - np.log(1.0))

        return (parser, params, None)
    
    
    def draw_samples(self, q_parser, q_params, K):
        
        """Use parser & params to produce K samples of the parameters for which
        q produces a posterior distribution. Do this row-wise."""

        v_c = np.exp(-q_parser.get(q_params, 'log_variance'))[ :, 0 ]
        m_c = q_parser.get(q_params, 'mean')[ :, 0 ] 

        coef_sample = npr.randn(K, self.d) * np.sqrt(v_c) + m_c
                
        return coef_sample
        
        
    def avg_loss(self, q_sample, Y_, X_, indices):
        """Retrieve the parameters (using the parser) and compute the 
        average negative log likelihood across the q_sample as well as 
        the X/Y sub-sample"""
        
        """Note 1: The q_sample is a sample from the coefficients b in 
                
                Y = X * b + e
        
        And so we compute for each b-sample the term X * b = Y_hat"""
        coef_sample = q_sample
        K = coef_sample.shape[0]
        M = len(Y_)
        
        """X_ is M x d, coef_sample is K x d, so Y_hat will be M x K"""
        Y_hat = np.matmul(X_, np.transpose(coef_sample))
        
        """Next, get average neg log lklhood"""
        loss = np.abs(np.tile(Y_.reshape(M,1), (1, K)) - Y_hat)
                    
        return np.mean(loss, 0)
    
    
    def report_parameters(self, q_parser, q_params):
        
        v_c = np.exp(-q_parser.get(q_params, 'log_variance'))[ :, 0 ]
        m_c = q_parser.get(q_params, 'mean')[ :, 0 ] 
                
        param_names = ['absolute value means', 'absolute value variances']
        param_values = [m_c, v_c]
        
        return (param_names, param_values)



class BMMLogLossFixedSigma(Loss):
    """Compute log likelihood loss between data & parameters (NOT q_params,
    but model params), create q_parameter objects for the BBGVI class etc. 
    Internal states of the object are:
            d           dimension of X (regressor matrix)
            K           number of clusters
            
    """
    
    def __init__(self,d, K, n, gamma = None):
        self.d = d
        self.K = K
        self.n = n
        self.gamma = gamma
        
    def make_parser(self, parser):
        """This parser needs: 
             - mu K-vector
             - sigma2 K-vector (in log form)
             - clusters n-vector (these are the [latent] cluster memberships)
         NOTE: clusters are NOT penalized with a div, obviously! They comple-
               tely occur inside the loss function and are treated accordingly    
        """
         
        """Create the parser object"""
        
        #mean and log variance for mu + sigma2
        parser.add_shape('mean', (self.d, self.K)) #for mu, sigma2
        parser.add_shape('log_variance', (self.d, self.K)) #for mu, sigma2

        #individual-specific latent terms. Categorial RV
        parser.add_shape('cluster_prob', (self.n, self.K))
        
        """Initialize the parser object"""
#        
#        """This means that the entries from self.K onwards in 'mean' are 
#        stored in log form & need to be transformed back"""
        """I.e., none of the mean parameters are stored as logs because
        we do not perform variational inference for the variances"""
        log_conversion = None 
        
        """Just produce some very small random numbers for the mean + var 
        of the clusters"""
        global_params = 0.1 * np.random.randn(self.K*2*self.d) #for global vars
        
        """For the discrete latent variable, just assign probability 1/K to
        each category for each observation & maybe slightly perturb it"""
        cluster_membership = np.ones((self.n,self.K))*(1.0/self.K) 
            
        """Set the log variances to be small (conversion: exp(-log_var))"""
        global_params[ parser.get_indexes(global_params, 'log_variance') ] = (
            global_params[ parser.get_indexes(global_params, 'log_variance') ] + 10.0)

        cluster_membership[parser.get_indexes(cluster_membership, 'cluster_prob') ] = (
            cluster_membership[ parser.get_indexes(cluster_membership, 'cluster_prob') ])
         
        """package all global variational parameters together -- only global
        parameters are passed directly to the divergence object & so we don't
        have to worry about passing along any of the other params. 
        BUT: We do need to pass along the mean log conversion object"""
        
        #PUT cluster_membership and global_params together
        all_params = np.concatenate((global_params.flatten(), cluster_membership.flatten()))
        
        return (parser, all_params, log_conversion)


    def draw_samples(self, q_parser, q_params, K):

        num_samples = K
        
        """global variational params"""
        mu_cluster_m = q_parser.get(q_params, 'mean')[:, :] 
        mu_cluster_v = np.exp(-q_parser.get(q_params, 'log_variance'))[ :, :]

        """local variational params for each x_i. Reparameterization used here
            is the softmax/kategorical RV"""
        c = q_parser.get(q_params, 'cluster_prob')
        cluster_probs_xi = np.exp(c) / np.sum(np.exp(c), axis=1).reshape(self.n,1)
        
        """Draw from cluster locations & variances"""
        #dim: d x K x S
        cluster_locations = (npr.randn(self.d, self.K, num_samples) * 
            np.sqrt(mu_cluster_v)[:,:,np.newaxis] + 
            mu_cluster_m[:,:,np.newaxis])
        
        """Don't draw from cluster assignments for the x_i -- the cluster_probs
        themselves are ALREADY defining a distribution!"""
        
        return (cluster_locations, cluster_probs_xi)


    
    def avg_loss(self, q_sample, Y_, X_=None, indices = None):
        """The average loss is a sum over the cluster probabilities (for each 
        x_i) and the samples from cluster centers & variances. objective is
        the following:
        
            E_{q(mu,sigma2)}[ 
                \sum_{i=1}^n\sum_{j=1}^K log(p(c_j) * p(x_i|c_{j,i}, mu_j, sigma_j)) 
            ] 
            + D(q||pi)
        
        OUTLINE OF COMPUTATIONS:
           
            We have the following hierarchy:
                
                \mu_{1:Kd} \sim prior_{\mu}
                \sigma_{1:Kd}^2 \sim prior_{\sigma^2}
                
                c_i \sim Categorical(1/K)
                x_i|c_i=c, \mu_{1:Kd}, \sigma_{1:Kd}^2 \sim 
                        N(x_i|\mu(c), \sigma^2(c))
                        
                where we have that for c_i = c
                
                \mu(c) = \mu_{(c-1)*d:c*d}
                \sigma(c) = \sigma^2_{(c-1)*d:c*d}
            
            Noticing that the prior terms will be dealt with inside the 
            divergence, we can focus on the likelihood computation. 
            Notice that an individual likelihood term is given by
            
                p(x_i| \mu_{1:K}, \sigma^2_{1:K}) = 
                    \sum_{j=1}^K prob(c_i = j) * N(x_i|\mu(j), \sigma^2(j))
                
            and the log likelihood for all data points is
            
              p(x_{1:n}| \mu_{1:K}, \sigma^2_{1:K})
                \sum_{j=1}^n \log\left(
                    \sum_{i=1}^K prob(c_i = j) * N(x_i|\mu(j), \sigma^2(j))  
                \right)
              
        
        NOTE: We do all of this ONLY for the indices of x that are in the 
              current batch. These are in the np.array indices
        """

        
        cluster_locations, cluster_probs_xi = q_sample
        
        #DEBUG: Does this make sense?
        cluster_probs_xi = cluster_probs_xi[indices]
        
        num_samples = cluster_locations.shape[-1] 
        n = Y_.shape[0]
        d = Y_.shape[1]
        LOG_FORMAT = True
    
        
        """STEP 1.1: Extract the Normal likelihood parts into n x S"""
            
        #dim: n x d x K x S  
        """NOTE: contains all likelihood terms on each dimension, for each 
                 cluter and for each sample"""
        negative_log_likelihood_raw = (
            (np.tile(Y_[:,:,np.newaxis,np.newaxis], (1, 1, self.K, num_samples)) - 
                     cluster_locations[np.newaxis,:,:,:])**2 /
                         1.0
                        #cluster_variances[np.newaxis:,:,:]
                        )
             
        #dim: d x K x S     
        """NOTE: contains all likelihood terms on each dimension"""
        log_dets = np.log(2 * math.pi * 
                          1.0
                          #(cluster_variances[:,:,:])
                          ) * np.ones((negative_log_likelihood_raw.shape[1], self.K,num_samples))
        
        #dim: n x d x K x S
        negative_log_likelihood_raw = negative_log_likelihood_raw + log_dets[np.newaxis, :,:,:]
        
        #dim: n x K x S
        """NOTE: likelihood terms aggregated across dimensions. This 
                 corresponds to independence across d."""
        negative_log_likelihood_raw = np.sum(negative_log_likelihood_raw, axis=1)
        
        
        """STEP 1.2: Multiply with the relevant individual-specific 
                     cluster-probabilities"""
        
        if LOG_FORMAT:
            #dim: n x K x S
            log_likelihoods_clusters = (-negative_log_likelihood_raw[:,:,:] + 
                                 np.log(cluster_probs_xi[:,:])[:,:,np.newaxis])
        elif not LOG_FORMAT:
            #dim: n x K x S
            likelihoods_clusters = (np.exp(-negative_log_likelihood_raw[:,:,:]) *  
                             (cluster_probs_xi[:,:])[:,:,np.newaxis])

            
        
        """STEP 2: Take the raw likelihoods we have and finally get the 
                   average log likelihood.
                   
                   We need two steps: 
                       
                       1. get to 
                   
                       \sum_{j=1}^K prob(c_i = j) * N(x_i|\mu(j), \sigma^2(j))
                
                   For each sample s.
                   
                       2. get the actual sample average over S and N.
                   
                   """
                   
        """STEP 2.1: row-logsumexp to get to n x S"""
        
        
        
        if LOG_FORMAT: 
            #dim: n x S
            logsumexp_observation_log_likelihoods = logsumexp(
                    log_likelihoods_clusters, axis=1)
            log_likelihoods = np.mean(logsumexp_observation_log_likelihoods)
            
            #Use robust losses where we approximate the integral with the observations
            ROBUST = (self.gamma is not None)
            if ROBUST:
                gamma = self.gamma
                log_integral = logsumexp(
                    (gamma) * logsumexp_observation_log_likelihoods - np.log(n),
                    axis=0)
                log_gamma_score = (
                    np.log(gamma / (gamma - 1.0)) + 
                    logsumexp_observation_log_likelihoods * (gamma - 1.0) + 
                    ((gamma - 1.0)/gamma)*log_integral)
                log_likelihoods = np.mean(np.exp(log_gamma_score))
                
            
        elif not LOG_FORMAT:
            #dim: n x S
            observation_log_likelihoods = np.log(
                    np.sum(likelihoods_clusters, axis=1))
            #dim: scalar
            log_likelihoods = np.mean(observation_log_likelihoods)
      
        
        """STEP 2.2: return the average over the sample- & observation axis"""
        
        return -log_likelihoods
    
    
    def report_parameters(self, q_parser, q_params):
        
        """global variational params"""
        mu_cluster_m = q_parser.get(q_params, 'mean')[:, :] 
        mu_cluster_v = np.exp(-q_parser.get(q_params, 'log_variance'))[ :, :]

        """local variational params for each x_i"""
        c = q_parser.get(q_params, 'cluster_prob')
        cluster_probs_xi = np.exp(c) / np.sum(np.exp(c), axis=1).reshape(self.n,1)
        
        #print("cluster_probs_xi", cluster_probs_xi)
                
        param_names = ['cluster membership probabilities', 
                       'cluster position mean', 'cluster position variance',
                       ]
        param_values = [cluster_probs_xi, 
                        mu_cluster_m, mu_cluster_v
                        ]
        
        return (param_names, param_values)





#DEBUG: DOES NOT WORK YET
class BMMLogLoss(Loss):
    """Compute log likelihood loss between data & parameters (NOT q_params,
    but model params), create q_parameter objects for the BBGVI class etc. 
    Internal states of the object are:
            d           dimension of X (regressor matrix)
            K           number of clusters
            
    """
    
    def __init__(self,d, K, n):
        self.d = d
        self.K = K
        self.n = n
        
    def make_parser(self, parser):
        """This parser needs: 
             - mu K-vector
             - sigma2 K-vector (in log form)
             - clusters n-vector (these are the [latent] cluster memberships)
         NOTE: clusters are NOT penalized with a div, obviously! They comple-
               tely occur inside the loss function and are treated accordingly    
        """
         
        """Create the parser object"""
        #add information to parser: Indices which model on the log scale
        #parser.add_shape('mean_log_conversion', (self.d,self.K*2))
        
        #mean and log variance for mu + sigma2
        parser.add_shape('mean', (self.d, self.K*2)) #for mu, sigma2
        parser.add_shape('log_variance', (self.d, self.K*2)) #for mu, sigma2

        #individual-specific latent terms. Categorial, i.e. we have K-1  
        #free parameters per observation & optimize over those
        parser.add_shape('cluster_prob', (self.n, self.K-1))
        
        """Initialize the parser object"""
        
        """This means that the entries from self.K onwards in 'mean' are 
        stored in log form & need to be transformed back"""
        log_conversion = np.zeros(( self.d, 2*self.K), dtype=bool)
        log_conversion[:, self.K:] = True
        
        """Just produce some very small random numbers for the mean + var 
        of the clusters"""
        global_params = 0.1 * np.random.randn(self.K*2*self.d*2) #for global vars
        
        """For the discrete latent variable, just assign probability 1/K to
        each category for each observation & maybe slightly perturb it"""
        cluster_membership = np.ones((self.n,self.K-1))*(1.0/self.K) #(1.0/self.K)

        global_params[ parser.get_indexes(global_params, 'log_variance') ] = (
            global_params[ parser.get_indexes(global_params, 'log_variance') ] + 3.0)
        cluster_membership[parser.get_indexes(cluster_membership, 'cluster_prob') ] = (
            cluster_membership[ parser.get_indexes(cluster_membership, 'cluster_prob') ])
         
        """package all global variational parameters together -- only global
        parameters are passed directly to the divergence object & so we don't
        have to worry about passing along any of the other params. 
        BUT: We do need to pass along the mean log conversion object"""
        
        #PUT cluster_membership and global_params 
        all_params = np.concatenate((global_params.flatten(), cluster_membership.flatten()))
        
        return (parser, all_params, log_conversion)


    def draw_samples(self, q_parser, q_params, K):

        num_samples = K
        
        """global variational params"""
        mu_cluster_m = q_parser.get(q_params, 'mean')[:, :self.K ] 
        mu_cluster_v = np.exp(-q_parser.get(q_params, 'log_variance'))[ :, :self.K ]
        
        sigma2_cluster_m = np.exp(-q_parser.get(q_params, 'mean'))[ :, self.K: ]
        sigma2_cluster_v = np.exp(-q_parser.get(q_params, 'log_variance'))[ :, self.K: ]

        """local variational params for each x_i"""
        cluster_probs_xi = q_parser.get(q_params, 'cluster_prob')
        
        """Draw from cluster locations & variances"""
        cluster_locations = (npr.randn(self.d, self.K, num_samples) * 
            np.sqrt(mu_cluster_v)[:,:,np.newaxis] + 
            mu_cluster_m[:,:,np.newaxis])
        cluster_variances = (npr.randn(self.d, self.K, num_samples) * 
            np.sqrt(sigma2_cluster_v)[:,:,np.newaxis] + 
            sigma2_cluster_m[:,:,np.newaxis])
        
        """Don't draw from cluster assignments for the x_i -- the cluster_probs
        themselves are ALREADY defining a distribution!"""
        
        return (cluster_locations, cluster_variances, cluster_probs_xi)


    
    def avg_loss(self, q_sample, Y_, X_=None, indices = None):
        """The average loss is a sum over the cluster probabilities (for each 
        x_i) and the samples from cluster centers & variances. objective is
        the following:
        
            E_{q(mu,sigma2)}[ 
                \sum_{i=1}^n\sum_{j=1}^K log(p(c_j) * p(x_i|c_{j,i}, mu_j, sigma_j)) 
            ] 
            + D(q||pi)
        
        OUTLINE OF COMPUTATIONS:
           
            We have the following hierarchy:
                
                \mu_{1:Kd} \sim prior_{\mu}
                \sigma_{1:Kd}^2 \sim prior_{\sigma^2}
                
                c_i \sim Categorical(1/K)
                x_i|c_i=c, \mu_{1:Kd}, \sigma_{1:Kd}^2 \sim 
                        N(x_i|\mu(c), \sigma^2(c))
                        
                where we have that for c_i = c
                
                \mu(c) = \mu_{(c-1)*d:c*d}
                \sigma(c) = \sigma^2_{(c-1)*d:c*d}
            
            Noticing that the prior terms will be dealt with inside the 
            divergence, we can focus on the likelihood computation. 
            Notice that an individual likelihood term is given by
            
                p(x_i| \mu_{1:K}, \sigma^2_{1:K}) = 
                    \sum_{j=1}^K prob(c_i = j) * N(x_i|\mu(j), \sigma^2(j))
                
            and the log likelihood for all data points is
            
              p(x_{1:n}| \mu_{1:K}, \sigma^2_{1:K})
                \sum_{j=1}^n \log\left(
                    \sum_{i=1}^K prob(c_i = j) * N(x_i|\mu(j), \sigma^2(j))  
                \right)
              
        """

        
        cluster_locations, cluster_variances, cluster_probs_xi = q_sample
        num_samples = cluster_locations.shape[-1] 
        n = Y_.shape[0]
    
        
        """STEP 1.1: Extract the Normal likelihood parts into n x S"""
            
        #dim: n x d x K x S  
        """NOTE: contains all likelihood terms on each dimension, for each 
                 cluter and for each sample"""
        negative_log_likelihood_raw = (
            (np.tile(Y_[:,:,np.newaxis,np.newaxis], (1, 1, self.K, num_samples)) - 
                     cluster_locations[np.newaxis,:,:,:])**2 /
                         cluster_variances[np.newaxis:,:,:]
                        )
             
        #dim: d x K x S     
        """NOTE: contains all likelihood terms on each dimension"""
        log_dets = np.log(2 * math.pi * 
                          (cluster_variances[:,:,:])
                          ) * np.ones((negative_log_likelihood_raw.shape[1], self.K,num_samples))
        
        #dim: n x d x K x S
        negative_log_likelihood_raw = negative_log_likelihood_raw + log_dets[np.newaxis, :,:,:]
        
        #dim: n x K x S
        """NOTE: likelihood terms aggregated across dimensions. This 
                 corresponds to independence across d."""
        negative_log_likelihood_raw = np.sum(negative_log_likelihood_raw, axis=1)

        
        """STEP 1.2: Multiply with the relevant individual-specific 
                     cluster-probabilities"""
                     
        #dim: n x (K-1) x S
        log_likelihoods_Km1_clusters = (-negative_log_likelihood_raw[:,:-1,:] + 
                         np.log(cluster_probs_xi[:,:])[:,:,np.newaxis])
        #dim: n x 1 x S
        log_likelihoods_K_cluster = (
                -negative_log_likelihood_raw[:,-1,:] + np.log(1.0 - 
                        np.sum(cluster_probs_xi[:,:],axis=1))[:,np.newaxis]
            ).reshape(n, 1, num_samples)
        
        
        
        """STEP 2: Take the raw likelihoods we have and finally get the 
                   average log likelihood.
                   
                   We need two steps: 
                       
                       1. get to 
                   
                       \sum_{j=1}^K prob(c_i = j) * N(x_i|\mu(j), \sigma^2(j))
                
                   For each sample s.
                   
                       2. get the actual sample average over S and N.
                   
                   """
                   
        """STEP 2.1: row-logsumexp to get to n x S"""
        
        #dim: n x 1 x S
        logsumexp_Km1 = logsumexp(log_likelihoods_Km1_clusters, axis=1)
        
        #dim: n x 1 x S
        max_vals = np.maximum(logsumexp_Km1, log_likelihoods_K_cluster)
        log_likelihoods = np.log(np.exp(logsumexp_Km1-max_vals) + 
                        np.exp(log_likelihoods_K_cluster-max_vals))
        log_likelihoods += max_vals
        
        """STEP 2.2: return the average over the sample- & observation axis"""
        
        return -np.mean(log_likelihoods)
    
    
    def report_parameters(self, q_parser, q_params):
        
        """global variational params"""
        mu_cluster_m = q_parser.get(q_params, 'mean')[:, :self.K ] 
        mu_cluster_v = np.exp(-q_parser.get(q_params, 'log_variance'))[ :, :self.K ]
        
        sigma2_cluster_m = np.exp(-q_parser.get(q_params, 'mean'))[ :, self.K: ]
        sigma2_cluster_v = np.exp(-q_parser.get(q_params, 'log_variance'))[ :, self.K: ]

        """local variational params for each x_i"""
        cluster_probs_xi = q_parser.get(q_params, 'cluster_prob')
        #print("cluster_probs_xi", cluster_probs_xi)
                
        param_names = ['cluster membership probabilities', 
                       'cluster position mean', 'cluster position variance',
                       'cluster variance mean', 'cluster variance variance',
                       ]
        param_values = [cluster_probs_xi, 
                        mu_cluster_m, mu_cluster_v,
                        sigma2_cluster_m, sigma2_cluster_v
                        ]
        
        return (param_names, param_values)



#Convince yourself that autograd can transcend classes...
if False:
    #from __future__ import absolute_import
    #from __future__ import print_function
    import autograd.numpy as np
    import autograd.numpy.random as npr
    from autograd.scipy.misc import logsumexp
    from autograd import grad
    
    def fun(x):
        return x**2
    
    class wrapper():
        
        def __init__(self,f):
            self.f = f
            
        def eval_f(self,x):
            return self.f(x)
    
    
    f_grad = grad(fun)
    
    fobj = wrapper(fun)
    
    def fun_via_obj(x):
        return fobj.eval_f(x)
    
    f_grad2 = grad(fun_via_obj)
    
    print(f_grad(2.45), f_grad(6.0))
    print(f_grad2(2.45), f_grad2(6.0))






        
        
        
    