# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:43:04 2022
"""
import numpy as np
import matplotlib.pyplot as plt
import plot_config
from scipy.optimize import curve_fit
import os

datadir = r'E:\data\paper_data\cavity_cooling'

SAVE_FIGURE = True

colors = plt.get_cmap('tab10')

# cooling from GKP state
fig, ax = plt.subplots(1, 1, dpi=600, figsize=(5, 2.8))
ax.set_ylabel(r'Prob. of $|0\rangle$ (unnormalized)')
ax.set_xlabel(r'Time (cycles)')
ax.set_xscale('log')
ax.set_ylim(0.25,0.95)

for fname in [os.path.join(datadir, 'dissipative_cooling_1.npz'),
              os.path.join(datadir, 'dissipative_cooling_2.npz')]:
    data = dict(np.load(fname))
    rounds = data.pop('rounds')
    m00 = float(data.pop('m00'))
    m01 = float(data.pop('m01'))
    
    for i, eps in enumerate(np.sort(list(data))):
        label = eps if '2' in fname else None
        ax.plot(rounds, data[eps], marker='.', linestyle='-', 
                label=label, color=colors(i))

rounds = [0,1200]
ax.plot(rounds, np.ones_like(rounds)*m00, linestyle='--', color=colors(0))
ax.plot(rounds, np.ones_like(rounds)*m01, linestyle='--', color='black')


plt.tight_layout()
savename = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\cavity_cooling\gkp_ecdc_cooling.pdf'
if SAVE_FIGURE: fig.savefig(savename)