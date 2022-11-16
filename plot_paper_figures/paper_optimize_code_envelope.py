# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:01:42 2021
"""
# append parent directory to path 
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

from scipy.special import hermite 
from math import factorial

matmul = tf.linalg.matmul
matvec = tf.linalg.matvec
trace = tf.linalg.trace

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
Delta = np.linspace(0.1, 0.5, 41)
Ns = 200 # dimension of the states

states = {'+Z':[], '-Z':[]}

for d in Delta:
    states['+Z'].append(gkp_state(d, 0, Ns))
    states['-Z'].append(gkp_state(d, 1, Ns))

states = {s : tf.concat(states[s], 0) for s in states.keys()}
states = {s : tf.cast(states[s], c64) for s in states.keys()}

# state projectors
projectors = {s : utils.outer_product(states[s], states[s]) 
              for s in states.keys()}

# Create finite-energy code projector for array of Delta
Pc = projectors['+Z'] + projectors['-Z']


### COMPUTE EXPECTATIONS OF FINITE-ENERGY CODE OPERATORS
rounds = np.array([0, 100, 200, 400, 800])

SAVE_FIGURE = True

purity = np.zeros_like(rounds, dtype=float)
nbar = np.zeros_like(rounds, dtype=float)
projected_purity = {}
avg_projector = {'+Z':{}, '-Z':{}, 'code':{}}

for i, t in enumerate(rounds):
    # load data
    fname = '+Z_n='+str(t)+'.npz'
    datadir = r'E:\data\paper_data\density_matrix_fit'
    data = np.load(os.path.join(datadir, fname))
    rho = data['rho_re'] + 1j*data['rho_im']
    rho = tf.cast(rho, c64)
    
    N = rho.shape[0] # dimension of the reconstructed density matrix
    nbar[i] = trace(matmul(ops.num(N), rho)) # avg photon number
    
    if t == 100:
        ### SLIGHTLY DISPLACE THE CODE
        # this is needed because the agent chose to stabilize a displaced code.
        # the displacement "eps" is found from data. 
        D = ops.DisplacementOperator(Ns)
        eps = np.ones(len(Delta)) * (0.08-0.12j) * np.sqrt(np.pi/2)
        eps = tf.cast(eps, c64)
        
        states = {s : matvec(D(eps), states[s]) for s in states.keys()}
        projectors = {s : utils.outer_product(states[s], states[s]) 
                      for s in states.keys()}
        Pc = projectors['+Z'] + projectors['-Z']
    
    # Density matrix projected onto the code space
    rho_c = matmul(matmul(Pc[:,:N,:N], rho), Pc[:,:N,:N])
    projected_purity[t] = (trace(matmul(rho_c,rho_c))/trace(rho_c)**2).numpy().real
    avg_projector['+Z'][t] = trace(matmul(projectors['+Z'][:,:N,:N], rho)).numpy().real
    avg_projector['-Z'][t] = trace(matmul(projectors['-Z'][:,:N,:N], rho)).numpy().real
    avg_projector['code'][t] = avg_projector['+Z'][t] + avg_projector['-Z'][t]
    purity[i] = float(trace(matmul(rho, rho)).numpy().real)
    

### PLOT DIFFERENT THINGS
fig, axes = plt.subplots(1,3, dpi=600, gridspec_kw={'width_ratios': [1.5,1,1]},
                         figsize=(7,2))

# Plot fidelity to |+Z> and |-Z> states of the finite-energy code and 
# expectation value of the code projector.
ax = axes[0]
colors = plt.get_cmap('seismic')
ax.set_ylabel('Operator expectation value')
ax.set_xlabel(r'Code envelope, $\Delta$')

ax.set_xlim(0.1-0.02, 0.605)

for i, t in enumerate(rounds):
    ax.plot(Delta, avg_projector['+Z'][t], 
            label=r'$|+Z\rangle\langle+Z|$', color=colors(0.5*(1-(-1+i/5))))

    ax.plot(Delta, avg_projector['-Z'][t], 
            label=r'$|-Z\rangle\langle-Z|$', color=colors(0.5*(1-(1-i/5))))
    
    if t != 0: 
        ax.plot(Delta, avg_projector['+Z'][t]+avg_projector['-Z'][t], 
                label=r'Projector, $\Pi_\Delta$', linestyle=':', color='k')

# Plot code projector expectation from completely independent measurement
# ax.plot(Delta, np.ones_like(Delta)*0.806, linestyle='--')


Delta_i = Delta[np.argmax(avg_projector['+Z'][0])]
print('Max initial fidelity: %.3f, at Delta=%.3f' 
      %(max(avg_projector['+Z'][0]), Delta_i))

ind = np.argmax(avg_projector['code'][200])
Delta_f = Delta[ind]
max_code_projectors = [avg_projector['code'][t][ind] for t in rounds[1:]]
print('Max code projector: %.3f +- %.3f, at Delta=%.3f' 
      %(np.mean(max_code_projectors), np.std(max_code_projectors), Delta_f))


optimal_projected_purity = []
for t in rounds:
    p = projected_purity[t][ind]
    optimal_projected_purity.append(p)


ax = axes[1]
ax. set_ylabel('Purity')
ax.set_xlabel('Time (cycles)')
ax.plot(rounds, np.ones_like(rounds)*0.5, linestyle='--', color='k')
ax.plot(rounds, np.ones_like(rounds)*1.0, linestyle='--', color='k')
ax.plot(rounds, purity, linestyle='none', marker='.', 
        label=r'${\rm Tr}\,[\rho^2]$')
ax.plot(rounds, optimal_projected_purity, linestyle='none', marker='.',
        label=r'${\rm Tr}\,[\rho_{\Delta}^2]$')
ax.legend()


ax = axes[2]
ax.set_xlabel('Time (cycles)')
ax.set_ylabel('Average photon number')
ax.plot(rounds, nbar, linestyle='--', color='k', marker='.')

plt.tight_layout()

if SAVE_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures_working\state_reconstruction'
    fig.savefig(os.path.join(savedir, 'qec_sweep'), fmt='.pdf')

