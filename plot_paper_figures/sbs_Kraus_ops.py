# -*- coding: utf-8 -*-
"""
Created on Fri Sep 30 13:00:09 2022
"""
# append parent directory to path 
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))

import tensorflow as tf
from tensorflow import complex64 as c64
import numpy as np
import matplotlib.pyplot as plt
import helper_functions as hf
from utils import density_matrix
import plot_config
import operators as ops
import utils
import qutip as qt


SAVE_FIGURE = False
SAVE_PROJECTOR_FIGURES = False


Nc = 50 # cavity hilbert space 
Delta = 0.34
a_1 = np.sqrt(2*np.pi) / 2
eps1 = np.sqrt(2*np.pi) * Delta**2 / 2 / 2

def D(x,y):
    return qt.displace(Nc, 1j*x+y)

# Expressions from Shraddha
Kg_x = 1/4*(D(eps1,0)*D(0,a_1)*D(eps1,0)-1j*D(eps1,0)*D(0,a_1)*D(-eps1,0)
          +D(eps1,0)*D(0,-a_1)*D(eps1,0)+1j*D(eps1,0)*D(0,-a_1)*D(-eps1,0)
          +1j*D(-eps1,0)*D(0,a_1)*D(eps1,0)+D(-eps1,0)*D(0,a_1)*D(-eps1,0)
          -1j*D(-eps1,0)*D(0,-a_1)*D(eps1,0)+D(-eps1,0)*D(0,-a_1)*D(-eps1,0))

Kg_p = 1/4*(D(0,eps1)*D(-a_1,0)*D(0,eps1)-1j*D(0,eps1)*D(-a_1,0)*D(0,-eps1)
          +D(0,eps1)*D(a_1,0)*D(0,eps1)+1j*D(0,eps1)*D(a_1,0)*D(0,-eps1)
          +1j*D(0,-eps1)*D(-a_1,0)*D(0,eps1)+D(0,-eps1)*D(-a_1,0)*D(0,-eps1)
          -1j*D(0,-eps1)*D(a_1,0)*D(0,eps1)+D(0,-eps1)*D(a_1,0)*D(0,-eps1))

Ke_x = 1/4*(D(eps1,0)*D(0,a_1)*D(eps1,0)-1j*D(eps1,0)*D(0,a_1)*D(-eps1,0)
          +D(eps1,0)*D(0,-a_1)*D(eps1,0)+1j*D(eps1,0)*D(0,-a_1)*D(-eps1,0)
          -1j*D(-eps1,0)*D(0,a_1)*D(eps1,0)-D(-eps1,0)*D(0,a_1)*D(-eps1,0)
          +1j*D(-eps1,0)*D(0,-a_1)*D(eps1,0)-D(-eps1,0)*D(0,-a_1)*D(-eps1,0))

Ke_p = 1/4*(D(0,eps1)*D(-a_1,0)*D(0,eps1)-1j*D(0,eps1)*D(-a_1,0)*D(0,-eps1)
          +D(0,eps1)*D(a_1,0)*D(0,eps1)+1j*D(0,eps1)*D(a_1,0)*D(0,-eps1)
          -1j*D(0,-eps1)*D(-a_1,0)*D(0,eps1)-D(0,-eps1)*D(-a_1,0)*D(0,-eps1)
          +1j*D(0,-eps1)*D(a_1,0)*D(0,eps1)-D(0,-eps1)*D(a_1,0)*D(0,-eps1))

Kgg = Kg_p * Kg_x
eigvals, U = np.linalg.eig(np.array(Kgg.dag()*Kgg))

idx = np.flip(eigvals.argsort())
eigvals = eigvals[idx]
U1 = tf.cast(U[:,np.flip(idx)], c64)
U = tf.cast(U[:,idx], c64)

# plot eigenspectrum
fig, ax = plt.subplots(1,1, dpi=600)
plt.plot(eigvals,'.')

K = {}
K['gg'] = tf.cast(np.array(Kg_p * Kg_x), c64)
K['ge'] = tf.cast(np.array(Kg_p * Ke_x), c64)
K['eg'] = tf.cast(np.array(Ke_p * Kg_x), c64)
K['ee'] = tf.cast(np.array(Ke_p * Ke_x), c64)

### Plot Kraus matrices
fig, axes = plt.subplots(1,4, figsize=(5,2), dpi=600, sharey=True)
labels = [r'$K_{gg}$', r'$K_{ge}$', r'$K_{eg}$', r'$K_{ee}$']
for i, s in enumerate(['gg', 'ge', 'eg', 'ee']):
    ax = axes[i]
    p = ax.set_title(labels[i])
    Ki = tf.linalg.adjoint(U1) @ K[s] @ U1
    p = ax.imshow(np.abs(Ki[-12:,-12:]), vmin=0, vmax=1, cmap='Reds')
    ax.set_xticks(np.arange(-0.5,12.5,2))
    ax.set_yticks(np.arange(-0.5,12.5,2))
    ax.grid(color='r', linestyle='--', linewidth=0.5)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(axis="y", color='w')
    ax.tick_params(axis="x", color='w')
plt.tight_layout()

savename = os.path.join(plot_config.save_root_dir, 
                        r'gkp_and_sbs_intro\Kraus.pdf')
if SAVE_FIGURE: fig.savefig(savename)




def lines(ax, color='r'):
    for level in np.arange(0, N_max, 2):
        ax.plot([0,9],[level,level], linestyle='--', color=color,
                dashes=(2.5, 1.5),linewidth=0.5)

### Plot quantum trajectories
fig, axes = plt.subplots(1,6, figsize=(5,1.5), sharey=True, sharex=True, dpi=600)
N_max = 32
init_state = 0

# 1st panel: no errors
ax = axes[0]
states = []
state = U[:,init_state]
states.append(state)
for s in ['gg', 'gg', 'gg', 'gg', 'gg', 'gg', 'gg', 'gg']: 
    state = tf.linalg.matvec(K[s], state)
    state, _ = utils.normalize(state)
    states.append(state)
print(tf.math.abs(utils.batch_dot(states[-1], U[:,init_state]))**2)
states = tf.linalg.matvec(tf.linalg.adjoint(U), tf.convert_to_tensor(states))
ax.pcolormesh(np.abs(states)[:,:N_max].transpose(), cmap='magma_r', vmin=0, vmax=1)
lines(ax, color='r') 


# 2nd panel: a^dagger error
ax = axes[1]
states = []
state = U[:,init_state]
state = tf.linalg.matvec(ops.create(Nc), state)
state, _ = utils.normalize(state)
states.append(state)
for s in ['eg', 'gg', 'gg', 'gg', 'gg', 'gg', 'gg', 'gg']: 
    state = tf.linalg.matvec(K[s], state)
    state, _ = utils.normalize(state)
    states.append(state)
print(tf.math.abs(utils.batch_dot(states[-1], U[:,init_state]))**2)
states = tf.linalg.matvec(tf.linalg.adjoint(U), tf.convert_to_tensor(states))
ax.pcolormesh(np.abs(states)[:,:N_max].transpose(), cmap='magma_r', vmin=0, vmax=1)
lines(ax, color='r')

# 3rd panel: a^dagger error
ax = axes[2]
states = []
state = U[:,init_state]
state = tf.linalg.matvec(ops.create(Nc), state)
state, _ = utils.normalize(state)
states.append(state)
for s in ['gg', 'gg', 'ge', 'gg', 'gg', 'gg', 'gg', 'gg']: 
    state = tf.linalg.matvec(K[s], state)
    state, _ = utils.normalize(state)
    states.append(state)
print(tf.math.abs(utils.batch_dot(states[-1], U[:,init_state]))**2)
states = tf.linalg.matvec(tf.linalg.adjoint(U), tf.convert_to_tensor(states))
ax.pcolormesh(np.abs(states)[:,:N_max].transpose(), cmap='magma_r', vmin=0, vmax=1)
lines(ax, color='r') 

# 4th panel: (a^dagger)^2 error
ax = axes[3]
states = []
state = U[:,init_state]
state = tf.linalg.matvec(ops.create(Nc) @ ops.create(Nc), state)
state, _ = utils.normalize(state)
states.append(state)
for s in ['eg', 'ge', 'gg', 'gg', 'gg', 'gg', 'gg', 'gg']: 
    state = tf.linalg.matvec(K[s], state)
    state, _ = utils.normalize(state)
    states.append(state)
print(tf.math.abs(utils.batch_dot(states[-1], U[:,init_state]))**2)
states = tf.linalg.matvec(tf.linalg.adjoint(U), tf.convert_to_tensor(states))
ax.pcolormesh(np.abs(states)[:,:N_max].transpose(), cmap='magma_r', vmin=0, vmax=1)
lines(ax, color='r') 

# # 5th panel: (a^dagger)^3 error and then a^dagger error
# ax = axes[4]
# states = []
# state = U[:,init_state]
# state = tf.linalg.matvec(ops.create(Nc) @ ops.create(Nc) @ ops.create(Nc), state)
# state, _ = utils.normalize(state)
# states.append(state)
# for i, s in enumerate(['eg', 'ge', 'eg', 'gg', 'ge', 'gg', 'gg', 'gg']): 
#     state = tf.linalg.matvec(K[s], state)
#     state, _ = utils.normalize(state)
#     if i == 2:
#         state = tf.linalg.matvec(ops.destroy(Nc), state)
#         state, _ = utils.normalize(state)
#     states.append(state)
# print(tf.math.abs(utils.batch_dot(states[-1], U[:,init_state]))**2)
# states = tf.linalg.matvec(tf.linalg.adjoint(U), tf.convert_to_tensor(states))
# ax.pcolormesh(np.abs(states)[:,:N_max].transpose(), cmap='magma_r', vmin=0, vmax=1)
# lines(ax, color='r') 

# 5th panel: (a^dagger)^3 error and then a^dagger error
ax = axes[4]
states = []
state = U[:,init_state]
state = tf.linalg.matvec(ops.destroy(Nc) @ ops.destroy(Nc) @ ops.destroy(Nc), state)
state, _ = utils.normalize(state)
states.append(state)
for i, s in enumerate(['eg', 'ge', 'eg', 'ge', 'gg', 'gg', 'gg', 'gg']): 
    state = tf.linalg.matvec(K[s], state)
    state, _ = utils.normalize(state)
    if i == 1:
        state = tf.linalg.matvec(ops.destroy(Nc), state)
        state, _ = utils.normalize(state)
    states.append(state)
print(tf.math.abs(utils.batch_dot(states[-1], U[:,init_state]))**2)
states = tf.linalg.matvec(tf.linalg.adjoint(U), tf.convert_to_tensor(states))
ax.pcolormesh(np.abs(states)[:,:N_max].transpose(), cmap='magma_r', vmin=0, vmax=1)
lines(ax, color='r') 

# 6th panel: (a^dagger)^4 error
ax = axes[5]
states = []
state = U[:,init_state]
state = tf.linalg.matvec(ops.create(Nc) @ ops.create(Nc) @ ops.create(Nc) @ ops.create(Nc), state)
state, _ = utils.normalize(state)
states.append(state)
for i, s in enumerate(['eg', 'ge', 'eg', 'ge', 'eg', 'ge', 'gg', 'gg']): 
# for i, s in enumerate(['ge', 'eg', 'gg', 'ge', 'eg', 'gg', 'gg', 'gg']): 
    state = tf.linalg.matvec(K[s], state)
    state, _ = utils.normalize(state)
    states.append(state)
print(tf.math.abs(utils.batch_dot(states[-1], U[:,init_state]))**2)
states = tf.linalg.matvec(tf.linalg.adjoint(U), tf.convert_to_tensor(states))
ax.pcolormesh(np.abs(states)[:,:N_max].transpose(), cmap='magma_r', vmin=0, vmax=1)
lines(ax, color='r') 


axes[5].set_yticks([])
axes[5].set_xticks([0.5,3.5,6.5])
axes[5].set_xticklabels([1,4,7])
ax.set_yticklabels([])
plt.tight_layout()
    
savename = os.path.join(plot_config.save_root_dir, 
                        r'gkp_and_sbs_intro\correction_demo.pdf')
if SAVE_FIGURE: fig.savefig(savename)



if 0:
    N = 100
    # Plot wigner of projectors onto subspaces generated by <a> and <a^d>
    state0 = tf.concat([U[:,0], tf.zeros(N-U.shape[0], dtype=c64)], axis=0)
    state0 = tf.linalg.matvec(ops.create(N), state0)
    state1 = tf.concat([U[:,1], tf.zeros(N-U.shape[0], dtype=c64)], axis=0)
    state1 = tf.linalg.matvec(ops.create(N), state1)
    (x, y, W) = hf.plot_phase_space(density_matrix(state0)+density_matrix(state1), 
                        False, phase_space_rep='wigner', op=True, pts=141)
    
    fig, ax = plt.subplots(1,1, figsize=(1,1), dpi=600)
    plot_kwargs = dict(cmap='seismic', vmin=-2/np.pi, vmax=2/np.pi, rasterized=True)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.pcolormesh(x, y, np.transpose(W), **plot_kwargs)
    plt.tight_layout()
    
    if SAVE_PROJECTOR_FIGURES:
        savename = os.path.join(plot_config.save_root_dir, 
                                r'gkp_and_sbs_intro\subspaces\C_gain.pdf')
        fig.savefig(savename)

    state0 = tf.concat([U[:,0], tf.zeros(N-U.shape[0], dtype=c64)], axis=0)
    state0 = tf.linalg.matvec(ops.destroy(N), state0)
    state1 = tf.concat([U[:,1], tf.zeros(N-U.shape[0], dtype=c64)], axis=0)
    state1 = tf.linalg.matvec(ops.destroy(N), state1)
    (x, y, W) = hf.plot_phase_space(density_matrix(state0)+density_matrix(state1), 
                        False, phase_space_rep='wigner', op=True, pts=141)
    
    fig, ax = plt.subplots(1,1, figsize=(1,1), dpi=600)
    plot_kwargs = dict(cmap='seismic', vmin=-2/np.pi, vmax=2/np.pi, rasterized=True)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.pcolormesh(x, y, np.transpose(W), **plot_kwargs)
    plt.tight_layout()
    
    if SAVE_PROJECTOR_FIGURES:
        savename = os.path.join(plot_config.save_root_dir, 
                                r'gkp_and_sbs_intro\subspaces\C_loss.pdf')
        fig.savefig(savename)

    # Plot wigner of projectors onto error subspaces
    for i in range(6):
        
        state0 = tf.concat([U[:,2*i], tf.zeros(N-U.shape[0], dtype=c64)], axis=0)
        state1 = tf.concat([U[:,1+2*i], tf.zeros(N-U.shape[0], dtype=c64)], axis=0)
        (x, y, W) = hf.plot_phase_space(density_matrix(state0)+density_matrix(state1), 
                            False, phase_space_rep='wigner', op=True, pts=141)
        
        fig, ax = plt.subplots(1,1, figsize=(2,2), dpi=600)
        plot_kwargs = dict(cmap='seismic', vmin=-2/np.pi, vmax=2/np.pi, rasterized=True)
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.pcolormesh(x, y, np.transpose(W), **plot_kwargs)
        plt.tight_layout()

        if SAVE_PROJECTOR_FIGURES:
            savedir = os.path.join(plot_config.save_root_dir, r'gkp_and_sbs_intro\subspaces')
            fig.savefig(os.path.join(savedir, 'C'+str(i)), fmt='.pdf')