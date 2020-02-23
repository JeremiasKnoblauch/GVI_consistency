#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 20:45:46 2019

@author: jeremiasknoblauch

Description: produce BMM pics
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:17:23 2019

@author: jeremiasknoblauch

Description: Extract the information from BLR experiments & produce plots
"""


import numpy as np
import matplotlib.pyplot as plt



def collect_i(i, d, D, Dhyper, prior_misspecification):
    """This function collects all files with spec d, D, Dhyper, 
    prior_misspecification and computes the averages & std dev for the 
    coefficient's means & variances"""
    
    base=("/Users/jeremiasknoblauch/Documents/OxWaSP/GVI/JASA" + 
          "/experiments_VARIABLE_DATASET/BMM1" + 
          "/prior_mis=" + str(prior_misspecification) + 
          "/model_mis=False" + 
          "/n=50" + 
          "/d=" + str(d) + 
          "/D=" + D + "_param=" + str(Dhyper) + "/")
    
    # read in files regarding coefs/means (means + variances)
    means_i = np.loadtxt(base + "mean_" + str(i) + ".txt")
    vars_i = np.loadtxt(base +  "variance_" + str(i) + ".txt")
    probs_i = np.loadtxt(base + "probabilities_" + str(i) + ".txt")
    
    # return everything
    return (means_i, vars_i, probs_i) 



def collect(d, D, Dhyper, prior_misspecification):
    """This function collects all files with spec n, d, D, Dhyper and 
    computes the averages & std dev for the coefficient's means & variances"""
    
    if D in ["KLD", "RAD"]:
        exp = "/experiments_VARIABLE_DATASET"
    else:
        exp = "/experiments_VARIABLE_DATASET_NEW"
        
    base=("/Users/jeremiasknoblauch/Documents/OxWaSP/GVI/JASA" + 
          exp + "/BMM1" +
          "/prior_mis=" + str(prior_misspecification) + 
          "/model_mis=False" + 
          "/n=50" + 
          "/d=" + str(d) + 
          "/D=" + D + "_param=" + str(Dhyper) + "/")
    

    print("d", d)
    all_means = np.zeros((100,d,2))
    all_variances = np.zeros((100,d,2))
    all_probabilities =np.zeros((100,50,2))
    
    # loop over all files
    for i in range(0,100):
        

        # read in files regarding coefs/means (means + variances)
        means_i = np.loadtxt(base + "mean_" + str(i) + ".txt")
        vars_i = np.loadtxt(base + "variance_" + str(i) + ".txt")
        probs_i = np.loadtxt(base + "probabilities_" + str(i) + ".txt")
        
        # figure out which column belongs to which cluster
        if np.mean(means_i[:,0]) > np.mean(means_i[:,1]):
            # if second column smaller, no change in ordering needed
            all_means[i,:,:] = means_i
            all_variances[i,:,:] = vars_i
            all_probabilities[i,:,:] = probs_i
        else:
            all_means[i,:,0] = means_i[:,1]
            all_means[i,:,1] = means_i[:,0]
            all_variances[i,:,0] = vars_i[:,1]
            all_variances[i,:,1] = vars_i[:,0]
            all_probabilities[i,:,0] = probs_i[:,1]
            all_probabilities[i,:,1] = probs_i[:,0]
        
    
    # return everything
    return (all_means, all_variances, all_probabilities) 
        
        
        
def plot_all_settings_for_one_coef(fig, ax_array, d_list, D_list, Dhyper_list, colors,
                                   coef_index = 1, i=1, JASA_plot = True):
    """For a list of divergences and dimensions, this collects all relevant
    files, computes their means/variances and plots them.
    
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
    
    
    # STEP 2: Get the labels on all x-axes
    subplotlabels = []
    for D, Dhyper in zip(D_list, Dhyper_list):
        if Dhyper == 0.0:
            subplotlabels += [D]
        else:
            if D == "RAD":
                if JASA_plot:
                    D = r'$RD^{(\alpha)}$'
                else:
                    D = r'$D_{AR}^{(\alpha)}$'
            else:
                if JASA_plot:
                    D = r'$AD^{(\alpha)}$'
                else:
                    D = r'$D_{A}^{(\alpha)}$'
            subplotlabels += [D+ ", " + r'$\alpha = $' + str(Dhyper)] 
    
    # STEP/LOOP 3: retrieve all settings & plot them
    for misspec_count, misspec in zip([0,1], [True, False]):
        for d, plot_count in zip(d_list, range(0,len(d_list))):
            
            #set up the objects that will contain all_means, all_vars etc
            all_means = np.zeros((len(D_list), 100, d, 2))
            all_vars = np.zeros((len(D_list), 100, d, 2))
            all_probs = np.zeros((len(D_list), 100,50, 2))
            
            # loop over D
            for D, Dhyper, count in zip(D_list, Dhyper_list, range(0,len(D_list))):
                all_means[count,:,:,:], all_vars[count,:,:,:], all_probs[count,:,:,:] = collect(
                    d,D, Dhyper, prior_misspecification = misspec)
            
            # summmarize the information by averaging over both n and d
            # and subtracting the cluster means that are the truth
            all_biases = all_means.copy()
            all_biases[:,:,:,0] -= 2.0
            all_biases[:,:,:,1] += 2.0
            D_specific_avg_bias = np.mean(np.abs(all_biases), axis=(1,2,3))
            D_specific_avg_var = np.mean((all_vars), axis=(1,2,3))
            
            # would be interesting to see the uncertainty in uncertainty
            D_specific_variance_bias = (np.var(all_biases, axis=(1,2,3)))
            D_specific_variance_var =(np.var(np.sqrt(all_vars), axis=(1,2,3)))
            
           
            
            # having collected all results with prior misspec for this choice of d,
            # proced to plot them all
            # set whisker plot stuff
            ax = ax_array[misspec_count, plot_count]
            xpos = np.linspace(1,len(D_list),len(D_list),dtype=int)
            
            print("xpos", xpos.shape, "y", D_specific_avg_bias.shape, "y err", 
                   D_specific_avg_var.shape)

            
            ax.errorbar(x=xpos, y=D_specific_avg_bias, yerr = D_specific_avg_var, fmt = 'none', ecolor = colors)
            ax.scatter(xpos[0],D_specific_avg_bias[0],s=50, c=colors[0], marker = "D") #marker
            ax.scatter(xpos[1:],D_specific_avg_bias[1:],s=50, c=colors[1:]) #marker
            
            if JASA_plot: # check empirical variance of bias + variance
                xpos_ = xpos + 0.3
                ax.scatter(xpos_[1:],D_specific_avg_bias[1:],s=50, c=colors[1:], zorder = 10)
                ax.scatter(xpos_[0],D_specific_avg_bias[0],s=50, c=colors[0], zorder = 10, marker= "D")
                
                
                err = ax.errorbar(x=xpos_, y=D_specific_avg_bias, 
                            yerr = D_specific_variance_bias, 
                            linestyle = ':',
                            fmt = 'none', ecolor = 'black', 
                            zorder = 0)
                err[-1][0].set_linestyle('--')
                 #marker
                
#                xpos_ = xpos + 0.5
#                ax.errorbar(x=xpos_, y= (D_specific_avg_bias + D_specific_variance_bias), yerr = D_specific_variance_var, fmt = 'none', ecolor = 'black')
#                ax.scatter(xpos_,(D_specific_avg_bias + D_specific_variance_bias),s=40, c='black') #marker
#                ax.errorbar(x=xpos_, y=D_specific_avg_bias - D_specific_variance_bias, yerr = D_specific_variance_var, fmt = 'none', ecolor = 'black')
#                ax.scatter(xpos_,(D_specific_avg_bias - D_specific_variance_bias),s=40, c='black')
                
            
            ax.set_xlim(xpos[0]-0.35, xpos[-1]+0.65)
            ax.axhline(0.0, linestyle = "--", color='grey')
            if misspec:
                ax.set_title('d=' + str(d), size = 14)
                ax.set_xticks([])
            else:
                #ax.set_xticklabels(subplotlabels, rotation=30, ha='right')
                ax.set_xticklabels(subplotlabels)
                ax.set_xticks(xpos)
            ax.tick_params(axis='x', which='both', labelsize=13)
            
            # if this is the first plot, add a description to the axis on LHS
            if plot_count == 0:
                if misspec:
                    lab = "misspecified" #  " + r'$\pi(\theta)$' #r'$\beta$' + str(coef_index + 1) 
                else:
                    lab = "well-specified" #  " + r'$\pi(\theta)$'
                ax.set_ylabel(lab, fontdict = {'size':11})
                ax.tick_params(axis='y', which='both', labelsize=12)

    # STEP 4: Return the figure
    return fig, ax_array

    

if True:
    
    # set up
    d_list = [50,100,250] #,25,10]
    D_list = ["KLD", "RAD", "RAD",  "reverseKLD",  "FD", "JeffreysD", "AD", "AD"] #, "RAD"]
    Dhyper_list = [0.0, 0.5, 2.0, 0.0, 0.0, 0.0, 0.5, 2.0] #, 2.0]
    
#    D_list = ["KLD", "RAD"] #"AD",  "FD"]
#    Dhyper_list = [0.0, 0.5] #,  0.5, 0.0]
    
#        
    D_list = ["KLD", "RAD"] #, "AD"] #,  "FD"]
    Dhyper_list = [0.0, 0.5] #,  0.5] #, 0.0]
    
    
    coef_index = 15
    i = 25
    lightblue = '#56B4E9'
    black = '#000000'
    darkblue = '#0072B2'
    green = '#009E73'
    orange = "#D55E00"
    
    titlesize = 15
            
    
    #colors = [orange, darkblue, darkblue, darkblue, darkblue, darkblue, darkblue, darkblue, darkblue]
    colors = [orange, darkblue] #, darkblue] #, darkblue]
#    colors = [orange, darkblue] #, darkblue, darkblue]#, darkblue, darkblue, darkblue, darkblue, darkblue] #, green]
#    colors = [lightblue]*3 + [black] * 3 + [green]*3
    
    fig, ax_array = plt.subplots(2, len(d_list), 
                                 figsize = (12, 5), sharey=True)
    
    # get that pic
    fig, ax_array = plot_all_settings_for_one_coef(fig, ax_array, d_list, D_list, 
                           Dhyper_list, colors, coef_index, i, True)
    fig.subplots_adjust(wspace = 0.10, hspace = 0.10, bottom = 0.35)
    
    lab = (r'$q^{\ast}_{GVI}(\mu_1^{true} { } - { } \mu_{1}|{\kappa})$')
    
    fig.text(0.04, 0.625, lab, va='center', rotation='vertical', 
             fontdict={'size': titlesize})
    fig.suptitle("GVI posteriors and prior specification",
                 size=titlesize, y=1.00)
    
    fig.savefig("/Users/jeremiasknoblauch/Documents/OxWaSP/GVI/JASA/"+
                "pics/"
                #"experiments_FIXED_DATASET/BMM1/pics/" + 
                "JASA_prior_misspec_all" + ".pdf", dpi = 400, format = 'pdf')
    
    
    
    # get pic for JASA paper
    fig, ax_array = plt.subplots(2, len(d_list), 
                                 figsize = (12, 5), sharey=True)
    
    # get that pic
    fig, ax_array = plot_all_settings_for_one_coef(fig, ax_array, d_list, D_list, 
                           Dhyper_list, colors, coef_index, i, False)
    fig.subplots_adjust(wspace = 0.10, hspace = 0.10, bottom = 0.35)
    
    
    lab = (r'$q^{\ast}_{GVI}(\theta_1^{true} { } - { } \theta_{1})$')
    
    fig.text(0.04, 0.625, lab, va='center', rotation='vertical', 
             fontdict={'size': titlesize})
    fig.suptitle("GVI posteriors and prior specification",
                 size=titlesize, y=1.00)
    
    fig.savefig("/Users/jeremiasknoblauch/Documents/OxWaSP/GVI/JASA/"+
                "pics/"
                #"experiments_FIXED_DATASET/BMM1/pics/" + 
                "prior_misspec_all" + ".pdf", dpi = 400, format = 'pdf')
    
    


    
    
    
