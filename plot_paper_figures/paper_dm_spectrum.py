# -*- coding: utf-8 -*-
"""
Created on Sat Sep 17 17:02:14 2022
"""

# append parent directory to path 
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import tensorflow as tf
from tensorflow import complex64 as c64
from math import pi, sqrt
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as hf
from utils import density_matrix
import plot_config

SAVE_SPECTRUM_FIGURE = False
SAVE_WIGNER_FIGURE = False

states = ['+Z', '-Z']
rounds = np.array([0,100, 200, 400, 800])
datadir = r'E:\data\paper_data\density_matrix_fit'

# Load data
dms = {'+Z':[], '-Z':[]}
for state in states:
    for i, t in enumerate(rounds):
        fname = state+'_n='+str(t)+'.npz'
        data = np.load(os.path.join(datadir, fname))
        rho = data['rho_re'] + 1j*data['rho_im']
        rho = tf.cast(rho, c64)
        dms[state].append(rho)


# Extract spectrum of reconstructed density matrix
spectrum = {'+Z':[], '-Z':[]}
for state in states:
    for i, t in enumerate(rounds):
        eigvals, eigvecs = tf.linalg.eigh(dms[state][i])
        eigvals = np.flip(eigvals)[:4].numpy()
        if t == 100 or t==12:
            eigvals = eigvals[np.array([0,2,1,3])]
        spectrum[state].append(eigvals)
spectrum = {k:np.array(v).real for (k,v) in spectrum.items()} # [T,4]


# Make the plot of this spectrum
fig, ax = plt.subplots(1,1,dpi=600, figsize=(3,2.5))
ax.set_ylabel('Spectrum of density matrix')
ax.set_xlabel('Time (cycles)')
ax.set_yscale('log')
ax.set_ylim(6e-3,1)
ax.set_xlim(0,850)
colors = {0:'#219ebc', 2:'#fb8500', 1:'#219ebc', 3:'#fb8500'}

for state in ['+Z']:
    for i in range(4):
        ax.plot(rounds[1:], spectrum[state][1:,i], 
                marker='.', color=colors[i])
    
    ax.plot(rounds[1:], spectrum[state][1:,0]+spectrum[state][1:,1], 
            linestyle='--', color=colors[0])
    ax.plot(rounds[1:], spectrum[state][1:,2]+spectrum[state][1:,3], 
            linestyle='--', color=colors[2])
# ax.set_xticks(np.log([100,200,400,800]))
# ax.set_xticklabels([100,200,400,800])

plt.tight_layout()

if SAVE_SPECTRUM_FIGURE:
    savename = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\spectrum_of_dm\spectrum'
    fig.savefig(savename, fmt='pdf')



# Get Wigners of different subspaces, only for T=800

rho = (dms['-Z'][-1]+dms['+Z'][-1])/2

# Eigenvalues
eigvals, eigvecs = tf.linalg.eigh(rho)
eigvals = tf.sort(tf.math.real(eigvals), direction='DESCENDING')
print(f'Eigenvalues {eigvals}')

# Plot wigner of projector onto 1st pair of eigenvalues
state0 = tf.concat([eigvecs[:,-1], tf.zeros(140-eigvecs.shape[0], dtype=c64)], axis=0)
state1 = tf.concat([eigvecs[:,-2], tf.zeros(140-eigvecs.shape[0], dtype=c64)], axis=0)
(x_C, y_C, W_C) = hf.plot_phase_space(density_matrix(state0)+density_matrix(state1), 
                    False, phase_space_rep='wigner', op=True)

# Plot wigner of projector onto 2nd pair of eigenvalues
state0 = tf.concat([eigvecs[:,-3], tf.zeros(140-eigvecs.shape[0], dtype=c64)], axis=0)
state1 = tf.concat([eigvecs[:,-4], tf.zeros(140-eigvecs.shape[0], dtype=c64)], axis=0)
(x_E, y_E, W_E) = hf.plot_phase_space(density_matrix(state0)+density_matrix(state1), 
                    False, phase_space_rep='wigner', op=True)




# Make a figure for the paper 
fig, axes = plt.subplots(2,1, figsize=(3.2,2.18), dpi=600, sharex=True)
plot_kwargs = dict(cmap='seismic', vmin=-2/np.pi, vmax=2/np.pi)
axes[0].set_aspect('equal')
axes[1].set_aspect('equal')
axes[0].set_xticks([])
axes[0].set_yticks([])
axes[1].set_xticks([])
axes[1].set_yticks([])
# axes[0].set_title(r'$\Pi_{\cal C}$')
# axes[1].set_title(r'$E\,\Pi_{\cal C}E^\dagger$')

axes[0].pcolormesh(x_C, y_C, np.transpose(W_C), **plot_kwargs)
axes[1].pcolormesh(x_E, y_E, np.transpose(W_E), **plot_kwargs)

plt.tight_layout()

if SAVE_WIGNER_FIGURE:
    savename = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures_working\spectrum_of_dm\wigners'
    fig.savefig(savename, fmt='pdf')