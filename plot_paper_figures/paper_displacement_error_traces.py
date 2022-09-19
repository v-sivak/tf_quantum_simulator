# -*- coding: utf-8 -*-
"""
Created on Thu Sep 15 19:02:57 2022
"""
import os
import plot_config
import numpy as np 
import matplotlib.pyplot as plt



SAVE_FIGURE = True
    
corr_data = np.load(r'E:\data\paper_data\displacement_error_traces\measurements.npz')
msmts, delta_range = corr_data['msmts'], corr_data['delta_range']

msmts_avg = np.mean(msmts, axis=1)

N_delta = 30
N_round = 22

fig, ax = plt.subplots(1, 1, dpi=600, figsize=(1.8,2))
ax.set_ylabel('Time after error (cycles)')
ax.set_xlabel(r'Displacement error, $\varepsilon\,/\,l_S$')
ax.set_xticks([0,0.25,0.5])
vmin = np.min(msmts_avg[:N_delta,:N_round].transpose())
vmax = np.max(msmts_avg[:N_delta,:N_round].transpose())
p = ax.pcolormesh(delta_range[:N_delta]/np.sqrt(2*np.pi), np.arange(N_round), 
              msmts_avg[:N_delta,:N_round].transpose(), cmap='gist_heat_r',
              vmin=vmin, vmax=1.0)

# cbar = fig.colorbar(p, ax=ax)
# cbar.set_ticks([0.1, 0.7])
# ax.collections[0].colorbar.ax.set_ylim(0.1,0.7)

plt.tight_layout()

if SAVE_FIGURE:
    savename = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\fig4_characterization\error_traces'
    fig.savefig(savename, fmt='pdf')