# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 16:15:14 2021

@author: ricro
"""

# Fitting both the healthy controls and schizophrenia data using the 
# "full," inflation-deflation model, and fitting using SSE

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy

import fitting_pfuhl_data_sse as fds # Fitting Data SSE


hc_data = fds.hc_data
scz_data = fds.scz_data

# Fitting options:
model = ["avg", "bayes"]
n_tries = 200


"""

FIRST, WE WILL FIT THE MODEL USING BOTH INFLATION AND DEFLATION PARAMETERS
THIS IS KNOWN AS THE 'FULL' MODEL IN THE AMPC PRESENTATION


FULL MODEL FIT

"""

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


print("Starting to Fit HC Data...")
hc_results = fds.fit_data(train_hc_data, n_tries = n_tries, model = model)
print("Finished Fitting HC Data!")

print("Starting to Fit SCZ Data...")
scz_results = fds.fit_data(train_scz_data, n_tries = n_tries, model = model)
print("Finished Fitting SCZ Data!")

# Now to build a dataframe of fit values and parameters

def build_results_df(df,results):
    
    ids = df["ID"].unique()
    Nprt = len(ids)
    
    cols = ["ID", "Group", "SSE", "Inflate", "Deflate", "Memory"]
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
        
        for m in range(4):
            x = np.append(x, results[m][n])
            
        returned_df.iloc[n] = x
        
    return returned_df

hc_params = build_results_df(hc_data, hc_results)
scz_params = build_results_df(scz_data, scz_results)

# Need to remove certain participants due to fitting errors or missing data


fig, axs = plt.subplots(1,4)
# Comparing SCZ and HC results

cols = hc_params.columns

for plot in range(4):
    
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

print("T-test comparing DEFLATE: HC - SCZ")
hc_def = hc_params["Deflate"].values
scz_def = scz_params["Deflate"].values

print(scipy.stats.ttest_ind(hc_def[hc_def <= 3], scz_def[scz_def <= 3]))

print("T-test comparing INFLATE: HC -SCZ")
hc_inf = hc_params["Inflate"].values
scz_inf = scz_params["Inflate"].values

print(scipy.stats.ttest_ind(hc_inf[hc_inf <= 3], scz_inf[scz_inf <= 3]))

# Finally, we can save the results as text files

hc_params.to_csv(model[0]+"_"+model[1] + "_" + "hc_params.txt", index = False)
hc_data.to_csv(model[0]+"_"+model[1] + "_" + "hc_data.txt", index = False)

scz_params.to_csv(model[0]+"_"+model[1] + "_" + "scz_params.txt", index = False)
scz_data.to_csv(model[0]+"_"+model[1] + "_" + "scz_data.txt", index = False)

# Finding a strong correlation, between .96 - .99, between inflation and 
# deflation parameters

# Need to try fitting to some data and then predicting last sequence and 
# seeing how well the model does compared to a "Bayes-like" model that
# has deflation = 0. 

    
"""

Testing the FULL MODEL through crossvalidation using SSE

We'll generate predictions of the model using the fitted parameters from the
training set, and compare them to the true data

We then get an SSE score for each participant. We can compare this 
measure person-by-person with the same scores from the Bayes model
to see which was more effective at accounting for data

"""  

# Building a function that builds a dataframe of Seq5_SSEs

# First, need to extract beads for Seq5 for each person

def extract_data(df_prt):
    
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
        
        params = df_params[df_params["ID"] == ids[n]][["Inflate","Deflate","Memory"]].values
        params = params[0]
        
        #print(params)
        
        data = df_data[df_data["ID"] == ids[n]]
        
        #print(data)
        
        beads, conf = extract_data(data)
        
        conf_sse = fds.lom.calculate_SSE(params, conf, beads, model = model)
        
        max_sse = max_possible_sse(conf)
        
        to_add = data[["ID", "Group"]].values
        sse_data = np.array([conf_sse, max_sse])
        to_add = np.append(to_add, sse_data)
        
        sse_df.iloc[n] = to_add
        
    return sse_df



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

best_params = hc_params[["Inflate", "Deflate","Memory"]].iloc[best].values
worst_params = hc_params[["Inflate", "Deflate","Memory"]].iloc[worst].values

best_pred = fds.lom.generate_predictions(best_params, best_beads)
worst_pred = fds.lom.generate_predictions(worst_params, worst_beads)

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

best_params = scz_params[["Inflate", "Deflate","Memory"]].iloc[best].values
worst_params = scz_params[["Inflate", "Deflate","Memory"]].iloc[worst].values

best_pred = fds.lom.generate_predictions(best_params, best_beads)
worst_pred = fds.lom.generate_predictions(worst_params, worst_beads)

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






