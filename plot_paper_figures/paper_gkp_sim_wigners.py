# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 09:58:40 2022

@author: qulab
"""
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import tensorflow as tf
from tensorflow import complex64 as c64
from math import pi, sqrt
import numpy as np
import operators as ops
import utils
import matplotlib.pyplot as plt
from plot_paper_figures import plot_config
from utils import density_matrix
import qutip as qt


from scipy.special import hermite 
from math import factorial

matmul = tf.linalg.matmul
matvec = tf.linalg.matvec
trace = tf.linalg.trace

SAVE_FIGURE = True

### TOOLS FOR GKP IN FOCK BASIS
J_max = 12
N_max = 60

def pos_coeffs(n, x):
    c = np.sqrt(2**n * factorial(n) * np.sqrt(np.pi))
    return np.exp(-x**2/2) * hermite(n)(x) / c

def gkp_coeffs(n, s, Delta):
    offs = 0 if s==0 else 1/2
    a = 0
    for j in range(-J_max, J_max+1, 1):
        a += pos_coeffs(n, 2*np.sqrt(np.pi)*(j+offs))
    a *= np.exp(-Delta**2 * n)
    return a

def gkp_state(Delta, s, N):
    psi = np.zeros(N)
    for n in range(N):
        if n < N_max:
            psi[n] = gkp_coeffs(n, s, Delta)
        else:
            n = 0
    norm = np.sqrt(np.sum(psi**2))
    return (psi / norm).reshape([1,N])
        
    
### CREATE GKP STATES OF DIFFERENT DELTA
Delta = [0.1, 0.2, 0.3, 0.4]
Ns = 120 # dimension of the states

states = {'+Z':[], '-Z':[]}

for d in Delta:
    states['+Z'].append(tf.cast(gkp_state(d, 0, Ns), c64))
    states['-Z'].append(tf.cast(gkp_state(d, 1, Ns), c64))


### CREATE WIGNERS
import helper_functions as hf

W = {'+Z':[], '-Z':[]}

for s  in ['+Z', '-Z']:
    for i, d in enumerate(Delta):
        (x, y, W_) = hf.plot_phase_space(states[s][i], False, 
                            pts=111, phase_space_rep='wigner', op=False)
        W[s].append(W_)


### Plot Wigners
fig, axes = plt.subplots(2, 4, sharex=True, sharey='row', figsize=(5, 3), dpi=600)

for j, d in enumerate(Delta):
    ### ----------------------------------------------------------------------
    # 1st row: Wigner of +Z state
    ax = axes[0,j]
    title = 'Ideal code' if j==0 else '$\Delta=$%.1f' %d
    ax.set_title(title)
    ax.set_aspect('equal')
    
    plot_kwargs = dict(cmap='seismic', vmin=-2/pi, vmax=2/pi)
    ax.pcolormesh(x, y, np.transpose(W['+Z'][j]), **plot_kwargs, rasterized=True)

    ax.set_yticks(np.arange(-3,4,1)*np.sqrt(np.pi/2))
    ax.set_yticklabels(np.arange(-3,4,1))


    ### ----------------------------------------------------------------------
    # 2nd row: Wigner of -Z state
    ax = axes[1,j]
    ax.set_aspect('equal')

    plot_kwargs = dict(cmap='seismic', vmin=-2/pi, vmax=2/pi)
    ax.pcolormesh(x, y, np.transpose(W['-Z'][j]), **plot_kwargs, rasterized=True)

    ax.set_yticks(np.arange(-3,4,1)*np.sqrt(np.pi/2))
    ax.set_yticklabels(np.arange(-3,4,1))

    ax.set_xticks(np.arange(-3,4,1)*np.sqrt(np.pi/2))
    ax.set_xticklabels(np.arange(-3,4,1))
    

axes[0,0].set_ylabel(r'${\rm Im}[\alpha]\,/\,\sqrt{\pi\,/\,2}$')
axes[1,0].set_ylabel(r'${\rm Im}[\alpha]\,/\,\sqrt{\pi\,/\,2}$')
axes[1,2].set_xlabel(r'${\rm Re}[\alpha]\,/\,\sqrt{\pi\,/\,2}$')

plt.tight_layout()


if SAVE_FIGURE:
    savename = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures_working\gkp_and_sbs_intro\wigners'
    fig.savefig(savename, fmt='pdf')