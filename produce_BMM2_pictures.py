#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 20:45:46 2019

@author: jeremiasknoblauch

Description: produce BMM pics
"""


import numpy as np
import matplotlib.pyplot as plt
import os



def collect_i(i, n, losstype, losshyper, model_misspecification):
    """This function collects all files with spec d, D, Dhyper, 
    prior_misspecification and computes the averages & std dev for the 
    coefficient's means & variances"""
    
    base=("/Users/jeremiasknoblauch/Documents/OxWaSP/GVI/JASA" + 
          "/experiments_FIXED_DATASET/BMM2" + 
          "/prior_mis=False" + 
          "/model_mis=" + str(model_misspecification) + 
          "/n=" + str(n) + 
          "/d=50" +
          "/losstype=" + losstype + "_param=" + str(losshyper) + "/")
    
    # read in files regarding coefs/means (means + variances)
    means_i = np.loadtxt(base + "mean_" + str(i) + ".txt")
    vars_i = np.loadtxt(base +  "variance_" + str(i) + ".txt")
    probs_i = np.loadtxt(base + "probabilities_" + str(i) + ".txt")
    
    # return everything
    return (means_i, vars_i, probs_i) 



def collect(Dtype, Dhyper, n_list, gamma_list, outlier_list):
    """This function collects all files with spec n, d, D, Dhyper and 
    computes the averages & std dev for the coefficient's means & variances"""
    
    # create object that can store everything.
    n_size = len(n_list)
    loss_size = len(gamma_list)
    outl_size = len(outlier_list)
    
    # we want to store variances & means (which are of dim d = 50 for 2 clusters)
    res_means = np.zeros((outl_size, n_size, loss_size, 50, 2))
    res_vars = np.zeros((outl_size, n_size, loss_size, 50, 2))
       
    for n, n_count in zip(n_list, range(0, len(n_list))):
        
        for losshyper, loss_count in zip(gamma_list, range(0, len(gamma_list))):
            
            # unless gamma = 1.0, losstype is "gamma"
            if losshyper == 1.0:
                losstype = "log"
            else:
                losstype = "gamma"
            
            for outlier, outlier_count in zip(outlier_list, range(0, len(outlier_list))):
                
                model_misspecification = outlier
    
    
                # set directory containing files
                base=("/Users/jeremiasknoblauch/Documents/OxWaSP/GVI/JASA" + 
                      "/experiments_FIXED_DATASET/BMM2" + 
                      #"/D=KLD_param=0.0" + 
                      "/D=" + Dtype + "_param=" + str(Dhyper) + #reverseKLD_param=0.0" + 
                      "/prior_mis=False" + 
                      "/model_mis=" + str(model_misspecification) + 
                      "/n=" + str(n) + 
                      "/d=50" +
                      "/losstype=" + losstype + "_param=" + str(losshyper) + "/")
                
                # get the files and their ids
                all_files = os.listdir(path = base)
                sim_id_str = all_files[0].split("_")[1].split(".")[0]
                if sim_id_str == "Store":
                    sim_id_str = all_files[-1].split("_")[1].split(".")[0]
                
                # read in variance & mean from this directory
                means_i = np.loadtxt(base + "mean_" + sim_id_str + ".txt")
                vars_i = np.loadtxt(base +  "variance_" + sim_id_str + ".txt")
                
                if np.mean(means_i[:,0]) > np.mean(means_i[:,1]):
                    # if second column smaller, no change in ordering needed
                    res_means[outlier_count,n_count, loss_count, :, :] = means_i
                    res_vars[outlier_count,n_count, loss_count,  :, :] = vars_i        
                else:
                    res_means[outlier_count,n_count, loss_count, :, 0] = means_i[:,1]
                    res_means[outlier_count,n_count, loss_count, :, 1] = means_i[:,0]
                    res_vars[outlier_count,n_count, loss_count, :, 0] = vars_i[:,1]
                    res_vars[outlier_count,n_count, loss_count, :, 1] = vars_i[:,0]   
                
    
    # return everything
    return (res_means, res_vars)
        
        
        
def plot_all_settings_for_one_coef(Dtype, Dhyper, n_list, gamma_list, 
                                   outlier_list, colors, 
                                   fig_size = (10, 8),
                                   ylim = None):
                                   #,coef_index = 1, i=1):
    """For a list of settings, compare the coef. posteriors
    
    Plot structure: On a given panel, plot contains
        - the average mean square error of the MAP (=coef mean) for each D
        - the average variance around the coef mean for each D. 
        
    NOTE: This makes sense because we KNOW that the two true clusters are
            at +2 and -2 in all dimensions! So we can calculate the mean
            over all dimensions (averaging over dims) and the average the 
            variance (over dims)
    NOTE: We assume that entries in D_list and Dhyper_list are matched up.    
    """
    
    # STEP 1: Set up the plot into which we will draw results.
    #         structure: First row = misspec. Second row = no misspec.
    
    # create object that can store everything.
    n_size = len(n_list)
    loss_size = len(gamma_list)
    outl_size = len(outlier_list)
    
    # plot size
    fig, ax_array = plt.subplots(outl_size, n_size,
                                 figsize = fig_size, sharey=True)
    d=50
    
    # STEP 2: Get the labels on all x-axes
    subplotlabels = []
    for Dhyper_ in gamma_list:
        if Dhyper_ == 1.0:
            subplotlabels += [r'$\log$']
        else:
            subplotlabels += [r'$\gamma = $' + str(Dhyper_)] 
    
    # STEP 3: Collect outcomes
    # dim: outl_size x n_size x loss_size x 50 x 2
    res_means, res_vars = collect(Dtype, Dhyper, n_list, gamma_list, outlier_list)
    
    
    # STEP/LOOP 3: retrieve all settings & plot them
    for misspec_count, misspec in zip(range(0, outl_size), outlier_list):
        # one row of the plot dedicated to each of the entries in outlier_list
        
        for n, plot_count in zip(n_list, range(0,n_size)):
            # in each row, we a subplot for each n_list entry
            
            
            # summmarize the information by averaging over d
            # and subtracting the cluster means that are the truth
            
            # dim: loss_size x 50 x 2
            all_biases = res_means[misspec_count, plot_count, :, :, :].copy()
            all_vars = res_vars[misspec_count, plot_count, :, :, :].copy()
            all_biases[:,:,0] -= 2.0
            all_biases[:,:,1] += 2.0
            D_specific_avg_bias = np.mean(np.abs(all_biases), axis=(1,2))
            D_specific_avg_var = np.mean(np.sqrt(all_vars), axis=(1,2))
            
           
            
            # having collected all results with prior misspec for this choice of d,
            # proced to plot them all
            # set whisker plot stuff
            ax = ax_array[misspec_count, plot_count]
            xpos = np.linspace(1,loss_size,loss_size,dtype=int)
            
            print("xpos", xpos.shape, "y", D_specific_avg_bias.shape, "y err", 
                   D_specific_avg_var.shape)
            
            ax.errorbar(x=xpos, y=D_specific_avg_bias, yerr = D_specific_avg_var, fmt = 'none', ecolor = colors)
            ax.scatter(xpos[:-1],D_specific_avg_bias[:-1],s=40, c=colors[:-1]) #marker
            ax.scatter(xpos[-1],D_specific_avg_bias[-1],s=40, c=colors[-1], marker = "D")
            ax.set_xlim(xpos[0]-1, xpos[-1]+1)
            if ylim is not None:
                ax.set_ylim(ylim[0], ylim[1])
            if misspec:
                ax.set_title('n=' + str(n))
                ax.set_xticks([])
            else:
                ax.set_xticklabels(subplotlabels, rotation=80, ha='right', fontdict={'size':14})
                ax.set_xticks(xpos)
            ax.axhline(0, color="grey", linestyle="--")
            
            # if this is the first plot, add a description to the axis on LHS
            if plot_count == 0:
                if misspec:
                    lab = "model misspecified" #r'$\beta$' + str(coef_index + 1) 
                else:
                    lab = "model correct"
                ax.set_ylabel(lab, size = 11.5)

    # STEP 4: Return the figure
    return fig, ax_array

    

if True:
    
    # challenge: IGNORE the index/number at the end of file (carries no info)
    # solution: os.listdir()
    
    
    selected_settings = [0,1,2,3]
    
    n_list = [25, 50, 100, 250, 500, 1000, 2500]#, 5000]
    outlier_list = [True, False]
    gamma_list = [1.0, 1.01, 1.001]
    gamma_list = [1.01, 1.001, 1.0]
    #n_list = [100, 500, 5000]
    
    
    all_settings = []
    for n in n_list:
        for outliers in gamma_list:
            for gamma in gamma_list:
                all_settings += [[n,outliers,gamma]]
    
    my_settings = all_settings 
    
    
    my_setting = my_settings[0]
    
    n, model_misspecification, gamma = my_setting[0], my_setting[1], my_setting[2]
    losstype="log"
    
    
    lightblue = '#56B4E9'
    black = '#000000'
    darkblue = '#0072B2'
    green = '#009E73'
    orange = "#D55E00"
            
    colors = [darkblue, darkblue, orange] #, green]
#    colors = [lightblue]*3 + [black] * 3 + [green]*3
    
    fig_size = (10,5) #old fig size
    fig_size = (10,8)
    
    D_list = ["KLD", "reverseKLD", "RAD",  "ED", "FD", "JeffreysD"]
    Dhyper_list = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
    
    
    JASA_plot = False
    if fig_size == (10,5):
        y_pos = 0.99
        titlesize = 17
    elif fig_size == (10,8):
        y_pos = 0.96
        titlesize = 17
    ylim=None #[-0.1, 6.0]
    
    for D, Dhyper in zip(D_list, Dhyper_list):
    
    
        print("D = ", D)
        print("Dhyper = ", Dhyper)
        
        # get that pic
        fig, ax_array = plot_all_settings_for_one_coef(D, Dhyper, n_list, 
                                    gamma_list, outlier_list, colors, 
                                    fig_size, ylim)
        
        # get plot label
        if Dhyper == 0.0:
            if D == "reverseKLD":
                D_ = "Reverse KLD"
            elif D == "JeffreysD":
                D_ = "JD" #["Jeffrey's D"]
            elif D == "FD":
                D_ = D #["Fisher's D"]
            elif D == "ED":
                D_ = D #["Exponential D"]
            elif D == "KLD":
                D_ = D
        else:
            if JASA_plot == True:
                if D == "RAD":
                    D_ = r'$RD^{(\alpha)}$'
                elif D == "AD":
                    D_ = r'$AD^{(\alpha)}$'
            else:
                if D == "RAD":
                    D_ = r'$D_{AR}^{(\alpha)}$'
                elif D == "AD":
                    D_ = r'$D_A^{(\alpha)}$'
        
        
        if D == 'RAD' or D == 'AD':
            fig.suptitle("GVI posterior for Bayesian mixture model, " 
                         + r'$D=$' + D_ + ", " + r'$\alpha=$' + str(Dhyper), 
                         size = titlesize, y=y_pos)
        elif D == 'KLD':
            fig.suptitle("VI posterior for Bayesian mixture model, " 
                         + r'$D=$' + D_, size = titlesize, y=y_pos)
        else:
            fig.suptitle("GVI posterior for Bayesian mixture model, " 
                         + r'$D=$' + D_, size=titlesize, y=y_pos)
            
        if JASA_plot:
            lab = (r'$q^{\ast}_{GVI}(\mu_1^{true} { } - { } \mu_1|\kappa)$')
        else:
            lab = (r'$q^{\ast}_{GVI}(\theta_1^{true} { } - { } \theta_{1})$')
            
            
        if D in ['JeffreysD', 'RAD', 'KLD']:
            if fig_size == (10,5):
                x_pos = 0.02
            elif fig_size == (10,8):
                x_pos = 0.02
#        if D in ['KLD']:
#            x_pos = 0.01
#        elif D in ['ED']:
#            x_pos = 0.02
        elif D in ['FD',  'reverseKLD', 'ED']:
            if fig_size == (10,5):
                x_pos = 0.03
            elif fig_size == (10,8):
                x_pos = 0.03
        if fig_size == (10,5):
            y_position = 0.58
        elif fig_size == (10,8):
            y_position = 0.55
            #if fig size is (10,5):
        fig.text(x_pos, y_position, lab, va='center', rotation='vertical', 
             fontdict={'size': titlesize})


        fig.subplots_adjust(wspace = 0.10, hspace = 0.05, bottom = 0.22, top = 0.88)
        
        savestr = "/Users/jeremiasknoblauch/Documents/OxWaSP/GVI/JASA/pics/"
        if JASA_plot:
            
            fig.savefig(savestr + "JASA_model_misspec_D=" + D #+ "_param=" + str(Dhyper) 
                        + ".pdf", dpi = 400, format = 'pdf')
        if JASA_plot:
            fig.savefig(savestr + "model_misspec_D=" + D #+ "_param=" + str(Dhyper)
                         + ".pdf", dpi = 400, format = 'pdf')    
            
        
    



    # plot a number of convergence results for different 

    
    
    
