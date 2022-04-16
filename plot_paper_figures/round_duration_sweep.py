# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 13:20:23 2022

@author: qulab
"""
import os
import plot_config
import matplotlib.pyplot as plt
import numpy as np

SAVE_FIGURE = True
MARKERSIZE = 3

palette = plt.get_cmap('tab10')
color = {'X' : palette(0), 
        'Y' : palette(1), 
        'Z' : palette(2)}


tau_rounds = {}
tau_rounds['X'] = np.array([171, 105, 76, 175, 120, 80, 92, 77, 161, 89, 54, 75, 
                            74, 60, 90, 113, 165])
tau_rounds['Y'] = np.array([135, 84, 56, 127, 86, 59, 61, 57, 108, 60, 35, 56, 
                            49, 40, 69, 92, 123])
tau_rounds['Z'] = np.array([185, 111, 87, 175, 123, 86, 93, 80, 156, 90, 61, 79, 
                            80, 56, 86, 123, 160])

round_duration = np.array([4.96, 6.95, 13.9, 4.96, 6.95, 13.9, 10.9, 13.9, 4.96, 
                           10.9, 16.9, 13.9, 13.9, 16.9, 10.9, 6.95, 4.96])



tau_rounds = {}
tau_rounds['X'] = np.array([53, 54, 89, 96, 106, 111])
tau_rounds['Y'] = np.array([31, 35, 51, 56, 66, 71])
tau_rounds['Z'] = np.array([56, 54, 81, 97, 106, 111])

round_duration = np.array([17.3, 15.3, 11.3, 9.3, 7.3])


ind = np.argsort(round_duration)
round_duration = round_duration[ind]
tau_rounds = {k: v[ind] for k,v in tau_rounds.items()}


fig, axes = plt.subplots(2,1, dpi=600, sharex=True, figsize=(3,2.5))
ax = axes[0]
ax.grid(True)
ax.set_ylabel('Lifetime (us)')

medians = {'X':[], 'Y':[], 'Z':[]}
for t in np.unique(round_duration):
    ind = np.where(round_duration==t)[0]
    for s in ['X', 'Y', 'Z']:
        medians[s].append(np.median(tau_rounds[s][ind]*round_duration[ind]))

for s in ['X', 'Y', 'Z']:
    ax.plot(round_duration, round_duration*tau_rounds[s], marker='o', 
            linestyle='none', color=color[s], markersize=MARKERSIZE)
    ax.plot(np.unique(round_duration), medians[s], color=color[s])


ax = axes[1]
ax.grid(True)
ax.set_xlabel('QEC step duration (us)')
ax.set_ylabel('Lifetime (steps)')    

medians = {'X':[], 'Y':[], 'Z':[]}
for t in np.unique(round_duration):
    ind = np.where(round_duration==t)[0]
    for s in ['X', 'Y', 'Z']:
        medians[s].append(np.median(tau_rounds[s][ind]))

for s in ['X', 'Y', 'Z']:
    ax.plot(round_duration, tau_rounds[s], marker='o', linestyle='none', 
            color=color[s], markersize=MARKERSIZE)
    ax.plot(np.unique(round_duration), medians[s], color=color[s])


plt.tight_layout()
    
# Save figure
if SAVE_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\fig4_noise_injection_experiments'
    savename = 'round_time_sweep'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')