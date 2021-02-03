# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 11:18:38 2021

@author: ricro
"""

# Giving a graphical example of the difference between the full and the 
# bayes-like model

import numpy as np
import matplotlib.pyplot as plt

full_params = np.array([1.15, .83, .47]) # Inflate, Deflate, Memory
bayes_params = np.array([1.15, 1, .47]) # Inflate, Deflate, Memory

np.random.seed(999999)
beads = np.random.normal(loc = 1, scale = 2, size = 20) >= 0
beads = beads.astype(int)

#beads = np.ones(19)
#beads = np.append(beads, 0)

# Here, we re-write the functions to correspond with the code: blue = 1, white = 0

def build_operator(params):
    
    lambda_1 = params[0]
    lambda_2 = params[1]
    
    T = np.array([
        [lambda_1, 0],
        [0, lambda_2]
        ])
    
    return T


def generate_history(alpha, beads, model = "add"):
    
    # Generates the values of the history for all the beads, to avoid repeat
    # calculations; 
    # allows for specification of which kind of model to use
    
    # Converting beads codes to one more useful for the history tally
    
    beads_signed = beads.copy()
    #print(beads_signed)
    beads_signed[beads_signed == 1] = 1
    beads_signed[beads_signed == 0] = -1
    # Remember that with the Pfuhl data, 0 = blue, 1 = white
    
    if model == "add":
        H = np.array([0])
        for n in range(len(beads)):
            h = beads_signed[n] + alpha*H[n]
            H = np.append(H, h)
            
    else:
        H = np.array([0])
        for n in range(1,len(beads)+1):
            h = ((n-1)*alpha*H[n-1] + beads_signed[n-1])/n
            H = np.append(H, h)
            
    return H

def generate_events(hist):
    # Use the history of beads to generate array of events to use:
    # 1 = history generally favors blue
    # 0 = history generally favors white
    
    events = hist >= 0
    events = events.astype(int)
    
    return events

def generate_predictions(params, beads, model = "add"):
    # params is an array as follows:
    # params = [lambda_IN, lambda_DE, alpha]
    # beads is an array of 1's for blue and 0's for white
    
    # setting the params:
    lambda_IN = params[0] # Inflation of congruent hypothesis;
    #   Should be >= 0 
    lambda_DE = params[1] # Deflation of incongruent hypothesis
    # Should be >= 0
    alpha = params[2] # Memory applied to history of beads
    # Shold be in [0,1]
    
    # beads will be an array of 1's and 0's, where 1 = BLUE, 0 = WHITE
    # in  NON-Pfuhl data. The "generate_history" function will do the necessary
    # conversions
    
    history = generate_history(alpha, beads, model)
    
    events = generate_events(history)
    
    # operator for inflation of BLUE hypothesis:
    T_blue = build_operator([lambda_IN, lambda_DE])
    # This operator is used when the history is congruent with a blue hyp
    
    # operator for inflation of WHITE hypothesis:
    T_white = build_operator([lambda_DE, lambda_IN])
    # This operator is used when the histry is congruent with a white hyp
    
    pred = np.array([])
    
    z = np.array([[1,1]]) # saving the vectors representing the evidence
    #z[0] = z[0]/np.linalg.norm(z[0])
    # for each hypothesis; starting with a unit vector of equal evidence
    # for each hypothesis
    
    N_trials = len(beads)    
    for trial in range(N_trials):
        
        if events[trial] == 1:
            old_z = z[trial]
            # Working with unit vectors to keep computations reasonable;
            # Does not affect final confidence predictions; see notes
            new_z = np.dot(T_blue, old_z)
            
            #new_z = new_z/np.linalg.norm(new_z)
            
            z = np.vstack([z, new_z])
            
            conf = new_z[0]/sum(new_z)
            
            pred = np.append(pred, conf)
            
        else:
             old_z = z[trial]
             new_z = np.dot(T_white, old_z)
             
             #new_z = new_z/np.linalg.norm(old_z)
             
             z = np.vstack([z, new_z])
             
             conf = new_z[0]/sum(new_z)
             
             pred = np.append(pred, conf)
             
    return pred, z


full_pred, full_z = generate_predictions(full_params, beads)
bayes_pred, bayes_z = generate_predictions(bayes_params, beads)

fig, axs = plt.subplots(1,2)

# First plot the z's for each model
# Use z[:,0] to get all x coordinates, z[:,1] for all y coordinates

axs[0].plot(full_z[:,0], full_z[:,1], 'r-*', label = "Full")
axs[0].plot(bayes_z[:,0], bayes_z[:,1], 'b-o', label = "Bayes")
axs[0].set_title('Evidence Vectors')


Trials = list(range(1, len(beads) + 1))
axs[1].plot(Trials, full_pred, 'r-*', label = "Full")
axs[1].plot(Trials, bayes_pred, 'b-o', label = "Bayes")
axs[1].set_title('Confidence Ratings Predictions')
axs[1].set_xticks(Trials)
axs[1].set_xticklabels(beads)
axs[1].legend(loc="upper left")


fig, axs = plt.subplots()

# Plotting additive vs averaging histories

alpha = full_params[2]
add = generate_history(alpha, beads, model = "add")
avg = generate_history(alpha, beads, model = "avg")

axs.plot(Trials, add[1:], 'r-*', label = "Add")
axs.plot(Trials, avg[1:], 'b-o', label = "Avg")
axs.plot(Trials, np.zeros(20), 'k-')
axs.set_xticks(Trials)
axs.set_xticklabels(beads)
axs.set_title('Compare: Additive vs Averaging History')
axs.legend(loc = "upper left")

print("Comparing events from Average vs Additive model")
event_add = generate_events(add)
event_avg = generate_events(avg)

print(event_add - event_avg)
