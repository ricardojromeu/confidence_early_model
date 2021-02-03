# -*- coding: utf-8 -*-
"""
Created on Wed Jan 27 18:39:55 2021

@author: ricro
"""

# Fitting the Gerit Pfuhl data using the fit_lom_sse.py module

import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

import fit_lom_sse as lom
import Clean_BT_data


full_data = Clean_BT_data.all_data


#full_data = pd.read_csv(r'C:\Users\ricro\Desktop\Year 5 PhD\Confidence Model - Beads Task\Pfuhl_Data.csv')

hc_data = full_data[full_data["Group"] == "NT"]

scz_data = full_data[full_data["Group"] == "SCZ"]

# Participants to exclude: prts 3775, 4608, both from SCZ group
# This leaves the final count at:
# HC = 46
# SCZ = 26 - 2 = 24
# ASD = 19

"""
Columns for the data:
    column_names = [
        0: "ID", 
        1: "Group", codes = NT (healthy control), SCZ, 
                                ASD (Autism Spectrum disorder)
        2: "Prt_Exclude", codes = 'n', 'y'
        3: "Sequence", codes = 1,2,3,4,5
        4: "Seq_Exclude", codes = 'n', 'y'
        5: "Bag_Origin"], codes = 1 = WHITE, 0 = BLUE
    Plus 20 columns named "1" - "20" for the trials
    These are 2-tuples, where the first component is the color bead, and
    the second component is the confidence rating, a number between 0 and 1,
    that was given on that trial
    Note: 1 = white bead, 0 = blue bead here 
    Trial k data: (bead_k, confidence_rating_k)

"""

# n_tries = 50 # how many times to try the minimization procedure, to help
# prevent getting "stuck" in a local minimum

# FITTING DATA TO HEALTHY CONTROLS:
    
def convert_params(params):
    lambda_IN = 10/(1 + np.exp(-params[0]))
    lambda_DE = 10/(1 + np.exp(params[1]))
    alpha = 1/(1 + np.exp(-params[2]))
    
    return np.array([lambda_IN, lambda_DE, alpha])
    
    
def sse_with_data(params, data, model = ["add", "bayes"]):
    
    
        
    params = convert_params(params)
    
    if model[1] == "bayes":
        params[1] = 1
    
    sse = 0
    
    cols_data = [str(x) for x in range(1,21)]
    
    n_seqs = len(data["Sequence"].unique())
    
    for j in range(n_seqs):
        x = data[cols_data].iloc[j].values
        
        #print(x)
        
        conf_ratings = np.array([y[1] for y in x])
        #print(conf_ratings)
        beads = np.array([y[0] for y in x])
        #print(beads)
        # Remember that 0 = blue, 1 = white in our Pfuhl data
        
        sse += lom.calculate_SSE(params, conf_ratings, beads, model)
        
    return sse
    
    

    
def fit_data(data_df, n_tries = 50, model = ["add", "bayes"]):

    ids = data_df["ID"].unique()
    n = len(ids)

    group_inflate = np.array([])
    group_deflate = np.array([])
    group_alpha = np.array([])
    group_sses = np.array([])

    for prt in range(n):
    
        # For each participant, we want to run throught the optimization procedure
        # n_tries times, and get the parameters that correspond to the lowest
        # SSE
    
        prt_data = data_df[data_df["ID"] == ids[prt]]
        
        #print(prt_data)
        
        prt_sses = np.array([])
        prt_inf = np.array([])
        prt_def = np.array([])
        prt_alpha = np.array([])
        
        
        for tries in range(n_tries):
            
            result = minimize(sse_with_data, x0 = np.random.normal(loc = 0, scale = 10, size = 3),
                              args = (prt_data, model))
            
            # result.fun gives the SSE value (function eval)
            # result.x gives array of parameter values that fit best
            
            prt_sses = np.append(prt_sses, result.fun)
            p = convert_params(result.x)
            
            prt_inf = np.append(prt_inf, p[0])
            prt_def = np.append(prt_def, p[1])
            prt_alpha = np.append(prt_alpha, p[2])
            
            
        min_sse_idx = prt_sses.argmin()
        
        group_sses = np.append(group_sses, prt_sses[min_sse_idx])

        group_inflate = np.append(group_inflate, prt_inf[min_sse_idx])
        group_deflate = np.append(group_deflate, prt_def[min_sse_idx])
        group_alpha = np.append(group_alpha, prt_alpha[min_sse_idx])
        
        print("Subj #" + str(prt+1) + " Finished Fitting")
        
    return group_sses, group_inflate, group_deflate, group_alpha



#hc_results = fit_data(hc_data)

#print("Healthy Controls Fits")
#print(hc_results[0])

#scz_results = fit_data(scz_data)

#print("SCZ Fits")
#print(scz_results[0])

# Plotting distributions of parameter estimates

# hc_titles = ["HC Inf", "HC Def", "HC Alpha"]
# scz_titles = ["SCZ Inf", "SCZ Def", "SCZ Alpha"]

# Healthy controls

#fig, axs = plt.subplots(1,3)
#for n in range(3):
#    axs[n].hist(hc_results[n + 1])
#    axs[n].set_title(hc_titles[n])
#    axs[n].set_axis([0,2])
    
#fig, axs = plt.subplots(1,3)
#for n in range(3):
#    axs[n].hist(scz_results[n + 1])
#    axs[n].set_title(scz_titles[n])
#    axs[n].set_axis([0,2])





        
        
    
    

