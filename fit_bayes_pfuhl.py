# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 22:40:32 2021

@author: ricro
"""

"""

Since the Bayes model can blow up so easily, we need to use an easier method of fitting the data:
    
    THE BAYES-LIKE MODEL ALLOWS ONLY FOR NON-ZERO INFLATION AND UNIT 
    DEFLATION
    
    CAN SIMPLIFY THE FITTING PROCESS BY NOTING THAT:
        
        c_n = 1 / ( 1 + inflate^(white - blue))
        
        where white = number of times history favors white jar
        blue = number of times history favors blue jar

"""

# Importing the needed data from other files:
    
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy
from scipy.optimize import minimize

import fitting_pfuhl_data_sse as fds # Fitting Data SSE


hc_data = fds.hc_data
scz_data = fds.scz_data

# Fitting options:
model = "avg"
n_tries = 200


## SPLITTING DATA INTO TRAIN-TEST SETS; USING THE 5TH SEQUENCE AS TEST SEQ

## Also need to consider that some sequences within participants are to be
# excluded, but this only happens in healthy control data

# Better to just remove HC with any missing sequences, especially since
# all with any missing sequence have more than one missing sequence

hc_ids_to_remove = hc_data[hc_data["Seq_Exclude"] == 'y']["ID"]
# This gives a series where the index is the index value in full DF corres-
# ponding to ID to remove; the column gives IDs to remove

hc_ids_to_remove = hc_ids_to_remove.unique()

hc_rows_to_remove = np.array([])
for prt in hc_ids_to_remove:
    rows = hc_data[hc_data["ID"] == prt].index.values
    
    hc_rows_to_remove = np.append(hc_rows_to_remove, rows)

hc_data = hc_data.drop(hc_rows_to_remove)

# Now do the same to SCZ data:
scz_rows_to_remove = scz_data[scz_data["Prt_Exclude"] == 'y']["ID"].index.values
scz_data = scz_data.drop(scz_rows_to_remove)

# This leaves us with:
    # N_hc = 42
    # N_scz = 24
    
### FOR CROSSVALIDATION, WE WANT TO POP OUT SEQUENCE 5 DATA AND STORE AS TEST
# DATA, FITTING ONLY TO SEQUENCES 1 - 4

test_rows = hc_data["Sequence"] == 5
test_rows = test_rows.values

train_rows = ~test_rows

train_hc_data = hc_data.iloc[train_rows]
test_hc_data = hc_data.iloc[test_rows]


# Same for scz_data
test_rows = scz_data["Sequence"] == 5
test_rows = test_rows.values

train_rows = ~test_rows

train_scz_data = scz_data.iloc[train_rows]
test_scz_data = scz_data.iloc[test_rows]


"""
Building simpler fitting functions for Bayes model and then fiting the model to data

"""

def generate_history(alpha, beads, model = "add"):
    # Generates the values of the history for all the beads, to avoid repeat
    # calculations; 
    # allows for specification of which kind of model to use
    
    # Converting beads codes to one more useful for the history tally
    
    beads_signed = beads.copy()
    #print(beads_signed)
    beads_signed[beads_signed == 1] = -1
    beads_signed[beads_signed == 0] = 1
    # Remember that with the Pfuhl data, 0 = blue, 1 = white
    
    if model == "add":
        H = np.array([0])
        for n in range(len(beads)):
            h = beads_signed[n] + alpha*H[n]
            H = np.append(H, h)
            
    else:
        H = np.array([0])
        for n in range(1,len(beads) + 1):
            h = ((n-1)*alpha*H[n-1] + beads_signed[n-1])/n
            H = np.append(H, h)
            
    return H


def generate_events(hist):
    # Use the history of beads to generate array of events to use:
    # 1 = history generally favors blue
    # 0 = history generally favors white
    
    events = hist[1:] >= 0
    events = events.astype(int)
    
    return events

def extract_data(df_prt):
    
    # Data must be a data frame
    
    beads = np.array([])
    conf = np.array([])
    
    cols = [str(x) for x in range(1,21)]
    
    data = df_prt[cols]
    
    for n in range(1,21):
        
        x = data[str(n)].values[0]
        #print(x)
        
        beads = np.append(beads, x[0])
        conf = np.append(conf, x[1])
        
    
    return beads, conf


def bayes_conf(params, beads, model = "add"):
    
    alpha = params[1]
    inflate = params[0]
    
    hist = generate_history(alpha, beads, model)
    events = generate_events(hist) # In events, 1 = blue favorable, 0 = white favorable
    
    n_trials = np.linspace(1,20,num=20)
    
    blue_events_n = events.cumsum()
    white_events_n = n_trials - blue_events_n
    
    
    conf = 1/(1 + inflate**(white_events_n - blue_events_n)) # This gives confidence for blue bag
    
    conf = 1 - conf # now becomes confidence for white bag
    
    return conf

def convert_params(params):
    lambda_IN = 10/(1 + np.exp(-params[0]))
    alpha = 1/(1 + np.exp(-params[1]))
    
    return np.array([lambda_IN, alpha])

def calc_bayes_sse_seq(params, conf_ratings, beads, model = "add"):
    
    # This function calculates SSE for a single sequence
    
    # Need to first generate predictions from the model, then return sse compared to conf_ratings
    
    
    
    pred = bayes_conf(params, beads, model) # Gives array of white confidence
    
    seq_sse = sum((conf_ratings - pred)**2)
    
    return seq_sse
    
    
    
    

def sse_bayes_prt(params, data, model = "add"):
    
    # This function calculates SSE for all sequences in a participant's dataframe
    
    # Putting in participant data into this data argument
        
    params = convert_params(params) # First slot is inflate, second is alpha
    
    sse = 0
    
    cols_data = [str(x) for x in range(1,21)]
    
    n_seqs = len(data["Sequence"].unique())
    
    for j in range(n_seqs):
        
        x = data[cols_data].iloc[j]
        beads = np.array([y[0] for y in x])
        conf_ratings = np.array([y[1] for y in x])
        
        sse += calc_bayes_sse_seq(params, conf_ratings, beads, model)
        
    return sse


    
    


def fit_bayes_model(data_df, n_tries = 50, model = "add"):
    
    ids = data_df["ID"].unique()
    Nprt = len(ids)
    
    
    group_inflate = np.array([])
    group_alpha = np.array([])
    group_sses = np.array([])

    for prt in range(Nprt):
    
        # For each participant, we want to run throught the optimization procedure
        # n_tries times, and get the parameters that correspond to the lowest
        # SSE
    
        prt_data = data_df[data_df["ID"] == ids[prt]]
        
        #print(prt_data)
        
        prt_sses = np.array([])
        prt_inf = np.array([])
        prt_alpha = np.array([])
        
        
        for tries in range(n_tries):
            
            result = minimize(sse_bayes_prt, x0 = np.random.normal(loc = 0, scale = 10, size = 2),
                              args = (prt_data, model))
            
            # result.fun gives the SSE value (function eval)
            # result.x gives array of parameter values that fit best
            
            prt_sses = np.append(prt_sses, result.fun)
            p = convert_params(result.x)
            
            prt_inf = np.append(prt_inf, p[0])
            prt_alpha = np.append(prt_alpha, p[1])
            
            
        min_sse_idx = prt_sses.argmin()
        
        group_sses = np.append(group_sses, prt_sses[min_sse_idx])

        group_inflate = np.append(group_inflate, prt_inf[min_sse_idx])
        group_alpha = np.append(group_alpha, prt_alpha[min_sse_idx])
        
        print("Subj #" + str(prt+1) + " Finished Fitting")
        
    return group_sses, group_inflate, group_alpha



# Now to build a dataframe of fit values and parameters

def build_results_df(df,results):
    
    ids = df["ID"].unique()
    Nprt = len(ids)
    
    cols = ["ID", "Group", "SSE", "Inflate", "Memory"]
    index = list(range(Nprt))
    
    returned_df = pd.DataFrame(None, index = index, columns = cols)
    
    for n in range(Nprt):
        
        
        # Need to change group codes to 1 = NT/controls, 2 = SCZ,
        # since arrays need the same data type throughout
        group = df[df["ID"] == ids[n]]["Group"].unique()
        if group == "NT":
            group = 1
        else:
            group = 2
        
        x = np.array([])
        x = np.append(x, [ids[n], group])
        
        for m in range(3):
            x = np.append(x, results[m][n])
            
        returned_df.iloc[n] = x
        
    return returned_df


# CROSS VALIDATION FUNCTIONS:
    
def max_possible_sse(conf):
    # Calculate the largest possible SSE for the given data, defined
    # as making a prediction of 1 or 0 for confidence rating that maximally 
    # generates error
    
    error = 1 - conf
    error = np.round(error)
    
    max_sse = np.sum( (conf - error)**2 )
    
    return max_sse


def cross_val(df_params, df_data, model = "add"):
    # Columns from df_params: [ID, Group, SSE, Inflate, Deflate, Memory]
    # Group: 1 = HC, 2 = SCZ
    
    # df_data should be a dataframe with ONLY sequence 5 data from prts
    
    
    
    ids = df_params["ID"]
    Nprt = len(ids)
    
    new_columns = ["ID", "Group", "Seq5_SSE", "Max_SSE"]
    
    # Set up empty dataframe for updating 
    sse_df = pd.DataFrame(None, index = list(range(Nprt)),
                          columns = new_columns)
    
    for n in range(Nprt):
        
        params = df_params[df_params["ID"] == ids[n]][["Inflate","Memory"]].values
        params = params[0]
        
        #print(params)
        
        data = df_data[df_data["ID"] == ids[n]]
        
        beads, conf = extract_data(data)
        
        conf_sse = sse_bayes_prt(params, data, model = model)
        
        max_sse = max_possible_sse(conf)
        
        to_add = data[["ID", "Group"]].values
        sse_data = np.array([conf_sse, max_sse])
        to_add = np.append(to_add, sse_data)
        
        sse_df.iloc[n] = to_add
        
    return sse_df



"""

NOW THAT THE FUNCTIONS ARE SET, WE CAN START FITTING THE MODEL TO THE DATA


RUNNING CROSS VALIDATION: FIRST FITTING TO TRAINING DATA SET AND THEN 
NEED TO COMPARE TO SEQUENCE 5

"""

print("Starting to Fit HC Data...")
hc_results = fit_bayes_model(train_hc_data, n_tries = n_tries, model = model)
print("Finished Fitting HC Data!")

print("Starting to Fit SCZ Data...")
scz_results = fit_bayes_model(train_scz_data, n_tries = n_tries, model = model)
print("Finished Fitting SCZ Data!")

# FIRST LOOKING AT FIT DATA ON TRAINING DATA


hc_params = build_results_df(hc_data, hc_results)
scz_params = build_results_df(scz_data, scz_results)

fig, axs = plt.subplots(1,3)
# Comparing SCZ and HC results

cols = hc_params.columns

for plot in range(3):
    
    axs[plot].hist(hc_params[cols[2 + plot]], alpha = 0.5, label = "HC")
    axs[plot].hist(scz_params[cols[2 + plot]], alpha = 0.5, label = "SCZ")
    axs[plot].set_title(cols[2 + plot])
    
axs[0].legend(loc = "upper right")

# Looking at some basic statistics

print("Mean SSE (SD) for each group: ")
print("HC: [Mean SD]")
print([np.mean(hc_params["SSE"]), np.std(hc_params["SSE"])])

print("SCZ: [Mean SD]")
print([np.mean(scz_params["SSE"]), np.std(scz_params["SSE"])])



print("T-test comparing INFLATE: HC -SCZ")
hc_inf = hc_params["Inflate"].values
scz_inf = scz_params["Inflate"].values

print(scipy.stats.ttest_ind(hc_inf[hc_inf <= 3], scz_inf[scz_inf <= 3]))

# Finally, we can save the results as text files

hc_params.to_csv("bayes_" + model + "_" + "hc_params.txt", index = False)
hc_data.to_csv("bayes_" + model + "_" + "hc_data.txt", index = False)

scz_params.to_csv("bayes_" + model+ "_" + "scz_params.txt", index = False)
scz_data.to_csv("bayes_" + model + "_" + "scz_data.txt", index = False)

# NOW LOOKING AT HOW WELL THE MODEL DOES IN CROSS VALIDATION TEST FOR EACH GROUP


hc_seq5_sse = cross_val(hc_params, test_hc_data, model = model)

scz_seq5_sse = cross_val(scz_params, test_scz_data, model = model)

# Should plot examples of best and worst cases for each group

fig, axs = plt.subplots(1,2)

# Healthy controls, first plot best case scenario
perc_sse_hc = hc_seq5_sse["Seq5_SSE"]/hc_seq5_sse["Max_SSE"]
best = perc_sse_hc.values.argmin()

worst = perc_sse_hc.values.argmax()

best_data = test_hc_data[test_hc_data["ID"] == hc_seq5_sse["ID"].iloc[best]]
best_beads, best_conf = extract_data(best_data)

worst_data = test_hc_data[test_hc_data["ID"] == hc_seq5_sse["ID"].iloc[worst]]
worst_beads, worst_conf = extract_data(worst_data)

best_params = hc_params[["Inflate", "Memory"]].iloc[best].values
worst_params = hc_params[["Inflate", "Memory"]].iloc[worst].values

best_pred = bayes_conf(best_params, best_beads)
worst_pred = bayes_conf(worst_params, worst_beads)

# Now to plot

Trials = list(range(1,21))

# Best
axs[0].plot(Trials, best_conf, 'bo-', label = "Data")
axs[0].plot(Trials, best_pred, 'r--*', label = "Pred")
axs[0].set_title('Best Case: HC')
axs[0].set_xticklabels(best_beads)
axs[0].set_ylabel('Confidence')
axs[0].set_xlabel('Trials')
axs[0].legend(loc = "upper right")

# Worst

axs[1].plot(Trials, worst_conf, 'bo-', label = "Data")
axs[1].plot(Trials, worst_pred, 'r--*', label = "Pred")
axs[1].set_title('Worst Case: HC')
axs[1].set_xticklabels(worst_beads)
axs[1].set_ylabel('Confidence')
axs[1].set_xlabel('Trials')


# Now to Plot SCZ data

fig, axs = plt.subplots(1,2)

# SCZ, first plot best case scenario
perc_sse_scz = scz_seq5_sse["Seq5_SSE"]/scz_seq5_sse["Max_SSE"]
best = perc_sse_scz.values.argmin()

worst = perc_sse_scz.values.argmax()

best_data = test_scz_data[test_scz_data["ID"] == scz_seq5_sse["ID"].iloc[best]]
best_beads, best_conf = extract_data(best_data)

worst_data = test_scz_data[test_scz_data["ID"] == scz_seq5_sse["ID"].iloc[worst]]
worst_beads, worst_conf = extract_data(worst_data)

best_params = scz_params[["Inflate", "Memory"]].iloc[best].values
worst_params = scz_params[["Inflate", "Memory"]].iloc[worst].values

best_pred = bayes_conf(best_params, best_beads)
worst_pred = bayes_conf(worst_params, worst_beads)

# Now to plot

Trials = list(range(1,21))

# Best
axs[0].plot(Trials, best_conf, 'bo-', label = "Data")
axs[0].plot(Trials, best_pred, 'r--*', label = "Pred")
axs[0].set_title('Best Case: SCZ')
axs[0].set_xticklabels(best_beads)
axs[0].set_ylabel('Confidence')
axs[0].set_xlabel('Trials')
axs[0].legend(loc = "upper left")


# Worst

axs[1].plot(Trials, worst_conf, 'bo-', label = "Data")
axs[1].plot(Trials, worst_pred, 'r--*', label = "Pred")
axs[1].set_title('Worst Case: SCZ')
axs[1].set_xticklabels(worst_beads)
axs[1].set_ylabel('Confidence')
axs[1].set_xlabel('Trials')

plt.show()

print("Modeling Results based on Crossvalidation: HCs")
print("Model = "+ model[0] + ", " + model[1])
print("Avg and SD of SSE:")
hc_sse = hc_seq5_sse["Seq5_SSE"].values
print([np.mean(hc_sse), np.std(hc_sse)])


print("Modeling Results based on Crossvalidation: SCZ")
print("Model = "+ model[0] + ", " + model[1])
print("Avg and SD of SSE:")
scz_sse = scz_seq5_sse["Seq5_SSE"].values
print([np.mean(scz_sse), np.std(scz_sse)])