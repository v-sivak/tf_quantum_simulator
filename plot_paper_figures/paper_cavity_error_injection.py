# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 14:17:13 2022
"""
import numpy as np
import matplotlib.pyplot as plt
import plot_config
import os

SAVE_FIGURE = False

datadir = r'E:\data\paper_data\cavity_error_injection'

### ROTATION ERRORS
data = np.load(os.path.join(datadir, 'rotation_sweep.npz'))
phi_range, times = data['phi_range'], data['times']
pauli = {'X':0, 'Y':0, 'Z':0}
pauli = {s:data[s] for s in pauli.keys()}

### PLOT ON A CIRCLE
fig, ax = plt.subplots(1,1, figsize=(1.75,1.75), dpi=600)

ax.axis('off')
ax.set_xlim(-1,1)
ax.set_ylim(-1,1)
ax.set_aspect('equal')
cmaps = {'Z':plt.get_cmap('Greens'), 'Y':plt.get_cmap('Oranges')}

for s in ['Z', 'Y']:
    for t in range(pauli[s].shape[1])[1:]:
        pauli_c = 0.5*(1+pauli[s][:,t]) * np.exp(1j*phi_range)
        ax.plot(pauli_c.real, pauli_c.imag, 
                color=cmaps[s](1-float(t-1)/pauli[s].shape[1]))

ax.plot(np.exp(1j*phi_range).real, np.exp(1j*phi_range).imag, 
        linestyle='-', color='k')
ax.plot(0.5*np.exp(1j*phi_range).real, 0.5*np.exp(1j*phi_range).imag, 
        linestyle='--', color='k')
ax.plot([0], [0], color='k', marker='.')

plt.tight_layout()

# Save figure
if SAVE_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\noise_injection'
    savename = 'rotation_errors'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')
    


### DISPLACEMENT ERRORS
data = np.load(os.path.join(datadir, 'displacement_sweep.npz'))
delta_range, times = data['delta_range'], data['times']
pauli = {'X':0, 'Y':0, 'Z':0}
pauli = {s:data[s] for s in pauli.keys()}
S = np.sqrt(2*np.pi)

### PLOT 
fig, ax = plt.subplots(1,1, figsize=(1.9,1.9), dpi=600)
ax.set_xlabel(r'Error amplitude, $\varepsilon/l_S$')
ax.set_ylabel('Probability (unnormalized)')
ax.set_xlim(0,2)
ax.set_yticks([0,0.5,1.0])

from matplotlib.ticker import AutoMinorLocator
minor_locator = AutoMinorLocator(4)
ax.xaxis.set_minor_locator(minor_locator)

cmaps = {'Z':plt.get_cmap('Greens'), 'Y':plt.get_cmap('Oranges'), 'X':plt.get_cmap('Blues')}

for s in ['X', 'Z', 'Y']:
    for t in range(pauli[s].shape[1])[1:]:
        pauli_c = 0.5*(1+pauli[s][:,t])
        ax.plot(delta_range/S, pauli_c, color=cmaps[s](1-float(t-1)/pauli[s].shape[1]))


d = 0.05
for i in range(2):
    ax.plot([d+i+0, d+i+1/4], [1,1], linestyle='--', color='k')
    ax.plot([d+i+1/4, d+i+1/4], [1,0], linestyle='--', color='k')
    ax.plot([d+i+1/4, d+i+3/4], [0,0], linestyle='--', color='k')
    ax.plot([d+i+3/4, d+i+3/4], [0,1], linestyle='--', color='k')
    ax.plot([d+i+3/4, d+i+1], [1,1], linestyle='--', color='k')

plt.tight_layout()

# Save figure
if SAVE_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\noise_injection'
    savename = 'displacement_errors'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')