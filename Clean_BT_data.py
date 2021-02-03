# -*- coding: utf-8 -*-
"""
Created on Mon Jan 25 14:20:56 2021

@author: ricro
"""

# Cleaning up and arranging Gerit Pfuhl beads data for later analysis

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


"""
Basics of data variables:
    (1) No name - row number of data
    (2) ID of prts
    (3) Group:
        NT - healthy controls
        ASD - autism spectrum disorder
        SCZ - schizophrenia
    (4) Trial - 1 to 20
    (5) Seq1_color: Bead Color
        1 - white
        0 - blue
    (6) Seq1_p_more_white: 
        Between 0 and 1
    (7) Seq1_origin_bag
        0 - blue
        1 - white
        Using an 80-20 proportion here
        
    (8) Seq2_color
    (9) Seq2_p_more_white
    (10) Seq2_origin_bag
    
    (11) Seq3_color
    (12) Seq3_p_more_white
    (13) Seq3_origin_bag
    
    (14) Seq4_color
    (15) Seq4_p_more_white
    (16) Seq4_origin_bag
    
    (17) Seq5_color
    (18) Seq5_p_more_white
    (19) Seq5_origin_bag

    (20) - (24) excl.t.seq1 - excl.t.seq5:
        'n' or 'y' to exclude data
    (25) excl.t.ID:
        'n' ("no") or 'y' ("yes")
        
ALL MISSING VALUES ARE LABELED AS -999

"""


complete_data = pd.read_csv("BT.csv", index_col = 0)

healthy_data = complete_data[complete_data["Group"] == "NT"]

asd_data = complete_data[complete_data["Group"] == "ASD"]

scz_data = complete_data[complete_data["Group"] == "SCZ"]


# Build a general function that takes the data above and returns a 
# dataframe with;
# column names:
    # (1) ID
    # (2) Group
    # (3) Excluded?
    # (4) Sequence Code (1 -5)
# The columns will then be
    # (1) Excluded? At Sequence Level
    # Followed by Trials 1 - 20
    
    
def rearrange_data(df):
    # df should have columns as in "complete_data" above
    
    # First, need to build the multiIndex
    ids = df["ID"].unique()
    Nprts = len(ids)
    Seqs = np.tile([1,2,3,4,5], Nprts)
    
    IDs = np.repeat(ids, 5)
    
    # Then need to build group array; can't use unique here since values are
    # repeated.
    
    Group_ID = np.array([])
    Excluded = np.array([])
    Excluded_Seq = np.array([])
    Bag_Color = np.array([])
    
    for prt in ids:
        
        prt_data = df[df["ID"] == prt]
        grp_code = prt_data["Group"].unique()
        
        exclude_code = prt_data["excl.ID"].unique()
        
        Group_ID = np.append(Group_ID, np.repeat(grp_code, 5))
        Excluded = np.append(Excluded, np.repeat(exclude_code, 5))
        
        exclude_seq = np.array([])
        bag_color = np.array([])
        
        for s in range(1,6):
            
            col = "excl.t.seq" + str(s)
            value = prt_data[col].unique()
            
            
            bag_column = "Seq" + str(s) + "_origin_bag"
            color = prt_data[[bag_column]].values[0]
            
            # Since the bags can switch, we'll just stick to the bag
            # used on te first trial of the sequence
            
            exclude_seq = np.append(exclude_seq, value)
            bag_color = np.append(bag_color, color)
            
            #if prt == 3384:
            #    print(bag_color)
            
        Excluded_Seq = np.append(Excluded_Seq, exclude_seq)
        Bag_Color = np.append(Bag_Color, bag_color)
        
    
    
    # Build without using MuliIndex, because MI is too complicated
    # Each row will be a different sequence for each participant
    
    column_names = ["ID", "Group", "Prt_Exclude", "Sequence",
                    "Seq_Exclude", "Bag_Origin"]
    
    
    column_names = column_names + [str(x) for x in range(1,21)]
    # These latter numbers indicate the trial number
    
    final_df = pd.DataFrame(columns = column_names)
    
    total_N = Nprts * 5
    
    for j in range(total_N):
        
        all_data = df[df["ID"] == IDs[j]]
        
        s_bead_idx = "Seq" + str(Seqs[j]) + "_colour"
        s_conf_idx = "Seq" + str(Seqs[j]) + "_p_more_white"
        
        zipped_data = list(zip(all_data[s_bead_idx].values,
                               all_data[s_conf_idx].values))
        
        to_add_data = [IDs[j], Group_ID[j],
                           Excluded[j], Seqs[j],
                           Excluded_Seq[j], Bag_Color[j]] + zipped_data
        
        #print(to_add_data)
        
        final_df.loc[j] = to_add_data
        
        # Use final_df.loc[n] to obtain data from the nth row
        
    
    return final_df
    
        
        
# Participants to exclude: prts 3775, 4608, both from SCZ group
# This leaves the final count at:
# HC = 46
# SCZ = 26 - 2 = 24
# ASD = 19
        
    
scz_data = rearrange_data(scz_data).head()
healthy_data = rearrange_data(healthy_data).head()

cols = [str(x) for x in range(1,21)]

# To extract all the confidence values from a sequence, use:
    # [x[1] for x in df[cols].iloc[nth-row].values]
    
    
# fig, axs = plt.subplots(1,2, sharex = True, sharey = True)

Trials = list(range(1,21))


#for n in range(5):
#    # First subplot:
#    conf = [x[1] for x in scz_data[cols].iloc[n].values]
#    axs[0].plot(Trials, conf, 'b-')
#    
#    # Second subplot:
#    conf = [x[1] for x in healthy_data[cols].iloc[n].values]
#    axs[1].plot(Trials, conf, 'r-')
    
#axs[0].set_title('SCZ')

#axs[1].set_title('HC')

all_data = rearrange_data(complete_data)




    
    