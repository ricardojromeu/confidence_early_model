# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 19:22:24 2021

@author: ricro
"""

"""

Module containing the functions to fit the Gerit Pfuhl Data using the 
Linear Operator Model (LOM). 

HERE: FIT USING SUM OF SQUARED ERROR

"""


import numpy as np


def build_operator(params, blue):
    # This function takes the parameters for inflation (first slot)
    # and deflation (second slot) to build a matrix to use for later
    if blue == 1:
        lambda_1 = params[0]
        lambda_2 = params[1]
    else:
        lambda_1 = params[1]
        lambda_2 = params[0]
    
    T = np.array([
        [lambda_1, 0],
        [0, lambda_2]
        ])
    
    # Each list is a row in the matrix; equivalent to MATLAB [p 0; 0 p]
    
    return T


def generate_history(alpha, beads, model = ["add", "bayes"]):
    # Generates the values of the history for all the beads, to avoid repeat
    # calculations; 
    # allows for specification of which kind of model to use
    
    # Converting beads codes to one more useful for the history tally
    
    beads_signed = beads.copy()
    #print(beads_signed)
    beads_signed[beads_signed == 1] = -1
    beads_signed[beads_signed == 0] = 1
    # Remember that with the Pfuhl data, 0 = blue, 1 = white
    
    if model[0] == "add":
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
    
    events = hist >= 0
    events = events.astype(int)
    
    return events

# Need to use np.dot(Matrix, vector) to obtain array of vector output

def generate_predictions(params, beads, model = ["add", "bayes"]):
    # params is an array as follows:
    # params = [lambda_IN, lambda_DE, alpha]
    # beads is an array of 1's for blue and 0's for white
    
    # setting the params:
    lambda_IN = params[0] # Inflation of congruent hypothesis;
    #   Should be >= 0 
    #lambda_DE = params[1] # Deflation of incongruent hypothesis
    # Should be >= 0
    if model[1] == "bayes":
        lambda_DE = 1
    else:
        lambda_DE = params[1]
    
    alpha = params[2] # Memory applied to history of beads
    # Shold be in [0,1]
    
    # beads will be an array of 1's and 0's, where 1 = white, 0 = blue
    # in our Pfuhl data. The "generate_history" function will do the necessary
    # conversions
    
    history = generate_history(alpha, beads, model)
    
    events = generate_events(history)
    
    # operator for inflation of BLUE hypothesis:
    T_blue = build_operator([lambda_IN, lambda_DE], 1)
    # This operator is used when the history is congruent with a blue hyp
    
    # operator for inflation of WHITE hypothesis:
    T_white = build_operator([lambda_IN, lambda_DE], 0)
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
            
            conf = new_z[1]/sum(new_z) 
            # the second slot corresponds to evidence for white, 
            # how the Pfuhl data are recorded
            
            pred = np.append(pred, conf)
            
        else:
             old_z = z[trial]
             new_z = np.dot(T_white, old_z)
             
             #new_z = new_z/np.linalg.norm(old_z)
             
             z = np.vstack([z, new_z])
             
             conf = new_z[1]/sum(new_z)
             
             pred = np.append(pred, conf)
             
    return pred


def calculate_SSE(params, data, beads, model = ["add", "bayes"]):
    
    pred = generate_predictions(params, beads, model)
    
    SSE = np.sum((data - pred)**2)
    
    return SSE
    
    
    
        
        
    
    