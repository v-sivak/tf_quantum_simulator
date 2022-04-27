# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 09:14:02 2022

@author: qulab
"""
import os
import plot_config
import numpy as np 
import matplotlib.pyplot as plt
from math import sqrt, pi

SAVE_FIGURE = False

d = np.load(r'Z:\shared\tmp\for Vlad\from_vlad\readout_characterization\readout.npz')

data = dict(g=d['data_g'], e=d['data_e'])
axis = dict(g=d['axis_g'], e=d['axis_e'])
threshold = d['threshold']

fig, ax = plt.subplots(1,1, dpi=200, figsize=(3.375, 2.5))
ax.set_xlabel('Integrated signal')
ax.set_ylabel('Counts')
ax.set_yscale('log')
for s in ['g', 'e']:
    ax.plot(axis[s], data[s], linestyle='none', marker='.')
ax.plot(np.ones(2)*threshold, [1, np.max(data['g'])], linestyle='--', color='k')
plt.tight_layout()
    
if SAVE_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\readout_characterization'
    savename = 'readout'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')