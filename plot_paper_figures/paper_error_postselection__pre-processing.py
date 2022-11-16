# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 13:49:50 2022
"""
import os
import numpy as np 
"""
In this pre-processing we: 
    1. Post-select data on correct state initialization (m0=g)
    2. Combine datasets where (+1 -> g, -1 -> e) and (+1 -> e, -1 -> g)
    3. Correct for Pauli frame flipping in SBS
"""

datadir = r'E:\data\large_datasets\raw_binary_data_latest'
pdatadir = r'E:\data\paper_data\error_postselection_dataset\pre_processed_data'

all_files = os.listdir(datadir)
all_rounds, all_states = [], []
for fname in all_files:
    all_rounds.append(int(fname.split('_')[1]))
rounds = np.unique(all_rounds)

for reps in rounds:
    fnames = [f for f in all_files if 'reps_'+str(reps)+'_' in f]
    pdata = {}
    for j, f in enumerate(fnames):
        data = np.load(os.path.join(datadir, f))
        for s in ['+Z', '+Y']:
            # For step (1), see docstring
            ig = np.where((1-data['m0_'+s+'_g_s0'])*(1-data['m0_'+s+'_g_s1']))[0]
            ie = np.where((1-data['m0_'+s+'_e_s0'])*(1-data['m0_'+s+'_e_s1']))[0]
            
            # For step (3), see docstring
            flip = True if (s!='+Y' and reps % 4 == 2) else False
            
            # For step (2), see docstring
            if j==0:
                if not flip:
                    pdata['m2_'+s] = np.concatenate(
                        [data['m2_'+s+'_g_s0'][ig], 
                         1-data['m2_'+s+'_e_s0'][ie]])
                else:
                    pdata['m2_'+s] = np.concatenate(
                        [1-data['m2_'+s+'_g_s0'][ig], 
                         data['m2_'+s+'_e_s0'][ie]])
                    
                if reps>0:
                    pdata['mi_'+s+'_s0'] = np.concatenate(
                        [data['mi_'+s+'_g_s0'][ig], 
                         data['mi_'+s+'_e_s0'][ie]])
        
                    pdata['mi_'+s+'_s1'] = np.concatenate(
                        [data['mi_'+s+'_g_s1'][ig], 
                         data['mi_'+s+'_e_s1'][ie]])
            else:
                if not flip:
                    pdata['m2_'+s] = np.concatenate(
                        [pdata['m2_'+s],
                         data['m2_'+s+'_g_s0'][ig], 
                         1-data['m2_'+s+'_e_s0'][ie]])
                else:
                    pdata['m2_'+s] = np.concatenate(
                        [pdata['m2_'+s],
                         1-data['m2_'+s+'_g_s0'][ig], 
                         data['m2_'+s+'_e_s0'][ie]])
                
                if reps>0:
                    pdata['mi_'+s+'_s0'] = np.concatenate(
                        [pdata['mi_'+s+'_s0'], 
                         data['mi_'+s+'_g_s0'][ig], 
                         data['mi_'+s+'_e_s0'][ie]])
        
                    pdata['mi_'+s+'_s1'] = np.concatenate(
                        [pdata['mi_'+s+'_s1'], 
                         data['mi_'+s+'_g_s1'][ig], 
                         data['mi_'+s+'_e_s1'][ie]])                

    np.savez(os.path.join(pdatadir, 'cycles_%d' %reps), **pdata)