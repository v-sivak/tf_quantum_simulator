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
LEGEND = False
MARKERSIZE = 2

palette = plt.get_cmap('tab10')
color = {'X' : palette(0), 
        'Y' : palette(1), 
        'Z' : palette(2)}


# Last point here is from my very first ever break-even last September
tau_rounds = {}
tau_rounds['X'] = np.array([171, 105, 76, 175, 120, 80, 92, 77, 161, 89, 54, 75, 
                            74, 60, 90, 113, 165, 227])
tau_rounds['Y'] = np.array([135, 84, 56, 127, 86, 59, 61, 57, 108, 60, 35, 56, 
                            49, 40, 69, 92, 123, 154])
tau_rounds['Z'] = np.array([185, 111, 87, 175, 123, 86, 93, 80, 156, 90, 61, 79, 
                            80, 56, 86, 123, 160, 234])

round_duration = np.array([4.96, 6.95, 13.9, 4.96, 6.95, 13.9, 10.9, 13.9, 4.96, 
                           10.9, 16.9, 13.9, 13.9, 16.9, 10.9, 6.95, 4.96, 3.9])



ind = np.argsort(round_duration)
round_duration = round_duration[ind]
tau_rounds = {k: v[ind] for k,v in tau_rounds.items()}


fig, axes = plt.subplots(2,1, dpi=600, sharex=True, figsize=(3,2.5))
ax = axes[0]
ax.set_ylim(0,1300)
ax.set_xticks([3,5,10,15])
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


# ax = axes[1]
# ax.grid(True)
# ax.set_xlabel('QEC step duration (us)')
# ax.set_ylabel('Lifetime (steps)')    

# medians = {'X':[], 'Y':[], 'Z':[]}
# for t in np.unique(round_duration):
#     ind = np.where(round_duration==t)[0]
#     for s in ['X', 'Y', 'Z']:
#         medians[s].append(np.median(tau_rounds[s][ind]))

# for s in ['X', 'Y', 'Z']:
#     ax.plot(round_duration, tau_rounds[s], marker='o', linestyle='none', 
#             color=color[s], markersize=MARKERSIZE)
#     ax.plot(np.unique(round_duration), medians[s], color=color[s])


ax = axes[1]
ax.grid(True)
ax.set_xlabel('QEC step duration (us)')
ax.set_ylabel(r'Error prob. $p_{\rm step}$')    

medians = {'X':[], 'Y':[], 'Z':[]}
for t in np.unique(round_duration):
    ind = np.where(round_duration==t)[0]
    for s in ['X', 'Y', 'Z']:
        p_L = 0.5*(1-np.exp(-1/tau_rounds[s][ind]))
        medians[s].append(np.median(p_L))

for s in ['X', 'Y', 'Z']:
    p_L = 0.5*(1-np.exp(-1/tau_rounds[s]))
    ax.plot(round_duration, p_L, marker='o', linestyle='none', 
            color=color[s], markersize=MARKERSIZE, label=r'$|+$'+s+r'$\rangle$')
    ax.plot(np.unique(round_duration), medians[s], color=color[s])

ax.set_yscale('log')
ax.grid(True)
ax.set_ylim([8e-4, 2e-2])
ax.set_yticks([1e-3, 1e-2])

if LEGEND:
    legend = ax.legend()
    legend.legendHandles[0]._legmarker.set_markersize(4)
    legend.legendHandles[1]._legmarker.set_markersize(4)
    legend.legendHandles[2]._legmarker.set_markersize(4)

plt.tight_layout()
    
# Save figure
if SAVE_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\idle_time_sweep'
    savename = 'round_time_sweep'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')