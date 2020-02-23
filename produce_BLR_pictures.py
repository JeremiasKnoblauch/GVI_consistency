#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 10:17:23 2019

@author: jeremiasknoblauch

Description: Extract the information from BLR experiments & produce plots
"""


import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl



def collect_i(i, n, D, Dhyper):
    """This function collects all files with spec n, d, D, Dhyper and 
    computes the averages & std dev for the coefficient's means & variances"""
    
    base=( "/Users/jeremiasknoblauch/Documents/OxWaSP/GVI/JASA/"  + #experiments_FIXED_DATASET/BLR"
          "experiment_home/BLR")
    path="/n=" + str(n) + "/D=" + str(D) + "_param=" + str(Dhyper) + "/"
    
    # read in files regarding coefs/means (means + variances)
    means_i = np.loadtxt(base + path + "means_" + str(i) + ".txt")
    vars_i = np.loadtxt(base + path + "variances_" + str(i) + ".txt")
    
    # read in files regarding sigma2/variances (mean + variance)
    mean_i = np.loadtxt(base + path + "mean_" + str(i) + ".txt")
    var_i = np.loadtxt(base + path + "variance_" + str(i) + ".txt")
    

           
    # return everything
    return (means_i, vars_i, 
            mean_i, var_i) 


def collect(n, D, Dhyper):
    """This function collects all files with spec n, d, D, Dhyper and 
    computes the averages & std dev for the coefficient's means & variances"""
    
    base=( "/Users/jeremiasknoblauch/Documents/OxWaSP/GVI/JASA/"  + #experiments_FIXED_DATASET/BLR"
          "experiment_home/BLR")
    path="/n=" + str(n) + "/D=" + str(D) + "_param=" + str(Dhyper) + "/"
    
    d = 20
    np.random.seed(10)
    all_means = np.zeros((100,d))
    all_variances = np.zeros((100,d))
    all_mean = np.zeros(100)
    all_variance = np.zeros(100)
    
    # loop over all files
    for i in range(0,100):
        

        # read in files regarding coefs/means (means + variances)
        means_i = np.loadtxt(base + path + "means_" + str(i+1) + ".txt")
        vars_i = np.loadtxt(base + path + "variances_" + str(i+1) + ".txt")
        all_means[i,:] = means_i
        all_variances[i,:] = vars_i
        
        # read in files regarding sigma2/variances (mean + variance)
        mean_i = np.loadtxt(base + path + "mean_" + str(i+1) + ".txt")
        var_i = np.loadtxt(base + path + "variance_" + str(i+1) + ".txt")
        all_mean[i] = mean_i
        all_variance[i] = var_i
        
    
    # return everything
    return (all_means, all_variances, 
            all_mean, all_variance) 
        
        
        
def plot_all_settings_for_one_coef(n_list, D_list, Dhyper_list, colors,
                                   coef_index = 1, i=1, JASA_plot = True):
    """For a list of divergences and sample sizes, this collects all relevant
    files, computes their means/variances and plots them.
    
    Plot structure: On a given panel, plot contains
        - the average mean square error of the MAP (=coef mean) for each D
        - the average variance around the coef mean for each D. 
    
    NOTE: We assume that entries in D_list and Dhyper_list are matched up.
    
    """
    
    # STEP 1: Get the true values back
    d = 20
    sigma2 = 25
    np.random.seed(10)
    coefs = np.random.normal(3, 10, d)
    
    # STEP 2: Create the plot's skeleton & create the labels for each sub-plot
    fig, ax_array = plt.subplots(1, len(n_list), 
                                 figsize = (16,5), sharey=True)
    
    subplotlabels = []
    for D, Dhyper in zip(D_list, Dhyper_list):
        if Dhyper == 0.0:
            if D == "reverseKLD":
                subplotlabels += ["Reverse KLD"]
            elif D == "JeffreysD":
                subplotlabels += ["JD"] #["Jeffrey's D"]
            elif D == "FD":
                subplotlabels += ["FD"] #["Fisher's D"]
            elif D == "ED":
                subplotlabels += ["ED"] #["Exponential D"]
            else:
                subplotlabels += [D]
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
            subplotlabels += [D_+ ", " + r'$\alpha = $' + str(Dhyper)] 
        
    # STEP 3: Retrieve for each setting the relevant data for all n, D specs
    #         and plot into the skeleton
    for n, plot_count in zip(n_list, range(0,len(n_list))):
                
        all_biases = np.zeros(len(D_list))
        all_sds = np.zeros(len(D_list))
        # STEP 3.1: Plot all D-settings into n-subplot
        for D, Dhyper, count in zip(D_list, Dhyper_list, range(0,len(D_list))):
            
            # retrieve results for this spec
            coef_mean,coef_var,sigma2_mean,sigma2_var=collect_i(i, n,D,Dhyper)
            if coef_index > -1:
                # take coef_index > -1 to mean that we have coefs at heart
                all_biases[count] = (coef_mean - coefs)[coef_index]
                all_sds[count]    = np.sqrt(coef_var)[coef_index]
            else:
                # otherwise, retrieve sigma2
                all_biases[count] = sigma2 - sigma2_mean
                all_sds[count] = np.sqrt(sigma2_var)
        
        # STEP 3.2: After we have collected all the individual results, plot
        #           them into one collection of box plots
        
#        # hand-code the whisker plot
#        item={}
#        item["label"] = subplotlabels # not required
#        #item["mean"] = all_biases # not required
#        item["med"] = all_biases
#        item["q1"] = all_biases - 0.6745 * all_sds
#        item["q3"] = all_biases + 0.6745 * all_sds
#        #item["cilo"] = 0.0 * np.ones(len(D_list)) # not required
#        #item["cihi"] = 0.0 * np.ones(len(D_list)) # not required
#        item["whislo"] = -10.0 #all_biases - 2.0 * all_sds # required
#        item["whishi"] = 10.0 #all_biases + 2.0 * all_sds # required
#        item["fliers"] = [] # required if showfliers=True            
#        stats = [item]
        
                  
        # set whisker plot stuff
        ax = ax_array[plot_count]
        #ax.bxp(stats, showfliers = False, shownotches=False)
        xpos = np.linspace(1,len(D_list),len(D_list),dtype=int)
        ax.errorbar(x=xpos, y=all_biases, yerr = all_sds, fmt = 'none', ecolor = colors) #'black', capsize = 7, zorder = 0) #, capthick = None)
        ax.scatter(xpos[1:],all_biases[1:],s=60, c=colors[1:], zorder=10) #marker
        ax.scatter(xpos[0],all_biases[0],s=60, c=colors[0], marker = "D",zorder=10)
        ax.set_title('n=' + str(n), size = 15)
        ax.set_xticklabels(subplotlabels, rotation=60, ha='right', fontdict={'size':14})
        ax.set_xticks(xpos)
        ax.set_xlim(xpos[0]-1, xpos[-1]+1)
        ax.axhline(0, c="grey", linestyle ="--")
        
        # if this is the first plot, add a description to the axis on LHS
        if plot_count == 0:
            if JASA_plot:
                if coef_index > -1:
                    lab = (r'$q^{\ast}_{GVI}($' +  
                           r'$\beta_1^{true} { } - { } \beta_{1}|{\kappa}$' +
                           r'$)$')
                           #r'$\beta$' + str(coef_index + 1) 
                else:
                    lab = r'$\sigma^2$'
                ax.set_ylabel(lab, fontdict = {'size':16})
            else:
                if coef_index > -1:
                    lab =  (r'$q^{\ast}_{GVI}($' +  
                           r'$\theta_1^{true} { } - { } \theta_1$' +
                           r'$)$')
                else:
                    lab = r'$\sigma^2$'
                ax.set_ylabel(lab, fontdict = {'size':16})

    # STEP 4: Return the figure
    return fig, ax_array

    

if True:
    
    # "#2171B5" "#6BAED6" "#BDD7E7" "#EFF3FF"
    VICol = "#D55E00"
    GVICol1 = "#0072B2"
    GVICol2 = "#56B4E9"
    GVICol3 = "#BDD7E7"
    GVICol4 =  "#EFF3FF"
    cols1 =  [GVICol1, GVICol2, GVICol3, GVICol4]*10
    cols2 = [VICol, VICol, VICol, VICol, VICol]
    cols_ = [cols1, cols2, cols1, cols1]
    
    
    
    # set up
    n_list = [10,25,50,100,250,500,1000,2500,5000,10000,25000]
    D_list = ["KLD", "AD", "AD", "RAD", "RAD",  "JeffreysD", "FD", "reverseKLD"]
    Dhyper_list = [0.0, 0.5, 2.0, 0.5 ,2.0, 0.0,  0.0, 0.0]
    
    n_list = [10,25,50,500,2500, 25000]
    D_list = ["KLD","RAD", #"RAD", #"AD", 
              "JeffreysD", "reverseKLD",  "FD"]
    Dhyper_list = [0.0, 0.5 , #0.5, 
                   0.0, 0.0, 0.0]
    
    
    # create blue colors
    n_cols = len(D_list)-1
    c = np.arange(1, n_cols + 1)
    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
    cols_blue = [cmap.to_rgba(3+i*0.7) for i in range(0, n_cols)]
    
    coef_index = 4 #2, 4, 13
    i = 1
    lightblue = '#56B4E9'
    black = '#000000'
    darkblue = '#0072B2'
    green = '#009E73'
    orange = "#D55E00"
            
    colors = [VICol] + cols_blue
#    colors = [lightblue]*3 + [black] * 3 + [green]*3
    
    # get that pic
    fig, ax_array = plot_all_settings_for_one_coef(n_list, D_list, 
                           Dhyper_list, colors, coef_index, i)
    fig.suptitle("Consistency of GVI posteriors on Bayesian linear regression", 
                 size = 20, y = 0.99)
    fig.subplots_adjust(wspace = 0.10, hspace = 0.00, bottom = 0.265, top = 0.86)
    
    fig.savefig("/Users/jeremiasknoblauch/Documents/OxWaSP/GVI/JASA/"+
                #"experiments_FIXED_DATASET/BLR/
                "pics/" + 
                "convergence_plot" + ".pdf", dpi = 400, format = 'pdf')
    
        
    # get that pic (GVI version of legend)
    fig, ax_array = plot_all_settings_for_one_coef(n_list, D_list, 
                           Dhyper_list, colors, coef_index, i, JASA_plot=False)
    fig.suptitle("Consistency of GVI posteriors on Bayesian linear regression", 
                 size = 20, y = 0.99)
    fig.subplots_adjust(wspace = 0.10, hspace = 0.00, bottom = 0.265, top = 0.86)
    
    
    fig.savefig("/Users/jeremiasknoblauch/Documents/OxWaSP/GVI/JASA/"+
                #"experiments_FIXED_DATASET/BLR/
                "pics/" + 
                "convergence_plot_GVI" + ".pdf", dpi = 400, format = 'pdf')


    
    
    