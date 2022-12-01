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
from tensorflow import float32 as f32
from math import pi, sqrt
import numpy as np
import operators as ops
# Use the GitHub version of TFCO
# !pip install git+https://github.com/google-research/tensorflow_constrained_optimization
import tensorflow_constrained_optimization as tfco
import matplotlib.pyplot as plt
from plot_paper_figures import plot_config
from scipy.optimize import curve_fit
import helper_functions as hf
from utils import density_matrix
    
matmul = tf.linalg.matmul
real = tf.math.real
imag = tf.math.imag
trace = tf.linalg.trace

"""
Here, the density matrix is parameterized as rho = (C^dag C) / Tr[(C^dag C)], 
which imposes the positive semidefinite structure with trace = 1. Matrix C can 
be any real matrix, so we represent it as C = A + iB with real matrices A and B 
whose coefficients are optimized.

"""
# TODO: add analytic construction of displaced parity

SAVE_FIGURE = False
SAVE_DENSITY_MATRIX = False
fname = r'+Z_n=400.npz'
datadir = os.path.join(plot_config.data_root_dir, 'Wigner_tomography')
dataname = os.path.join(datadir, fname)
figsavedir = os.path.join(plot_config.save_root_dir, 'state_reconstruction')
dmsavedir = os.path.join(plot_config.data_root_dir, 'density_matrix_fit')
N = 32 # dimension of the reconstructed density matrix


### Load experimental Wigner -------------------------------------------------
#-----------------------------------------------------------------------------
scale = 2/pi

def get_exp_wigner(fname):
    data = np.load(fname)
    W = data['W'] * scale
    xs, ys = data['W_xs'], data['W_ys']
    return W, xs, ys


def get_exp_contrast(fname):
    data = np.load(fname)
    P = data['sx']
    P = (P+np.flip(P))/2
    xs = data['P_xs']

    def quadratic(beta, a, b):
        return a - b * beta**2
    popt, pcov = curve_fit(quadratic, xs, P, p0=(0.8,0))

    fig, ax = plt.subplots(1,1, dpi=100)
    ax.plot(xs, P)
    ax.plot(xs, quadratic(xs, *popt))
    
    contrast = lambda beta: np.sqrt(quadratic(beta, *popt))
    return contrast


def normalize_wigner(W_exp, xs, ys):
    # find the area of elementary square in phase space
    area = (xs[-1] - xs[0]) / (len(xs)-1) * (ys[-1] - ys[0]) / (len(ys)-1)
    norm = np.sum(W_exp) * area
    W_normalized = W_exp / norm
    return W_normalized


W_exp, xs, ys = get_exp_wigner(dataname)
# W_exp = normalize_wigner(W_exp, xs, ys)
W_flat = tf.reshape(W_exp, [-1])
W_flat = tf.cast(W_flat, f32)

xs_mesh, ys_mesh = np.meshgrid(xs, ys, indexing='ij')
grid = tf.cast(xs_mesh + 1j*ys_mesh, c64)
grid_flat = tf.reshape(grid, [-1])

contrast = get_exp_contrast(dataname)


### Create displaced parity operator matrices for Wigner tomography ----------
#-----------------------------------------------------------------------------
def create_displaced_parity_tf():
    N_large = 140 # dimension used to compute the tomography matrix
    D = ops.DisplacementOperator(N_large)
    P = ops.parity(N_large)
    displaced_parity = matmul(matmul(D(grid_flat), P), D(-grid_flat))
    # Convert to lower-dimentional Hilbert space; shape=[N_alpha,N,N]
    displaced_parity_re = real(displaced_parity[:,:N,:N])
    displaced_parity_im = imag(displaced_parity[:,:N,:N])
    return (displaced_parity_re, displaced_parity_im)

disp_parity_re, disp_parity_im = create_displaced_parity_tf()

### Create parameterization of the density matrix ----------------------------
#-----------------------------------------------------------------------------
A = tf.Variable(tf.random.uniform([N,N]), dtype=tf.float32, name='A')
B = tf.Variable(tf.random.uniform([N,N]), dtype=tf.float32, name='B')

def loss_fn():
    rho_trace = trace(matmul(tf.transpose(A), A) + matmul(tf.transpose(B), B))
    rho_im = (matmul(tf.transpose(A), B) - matmul(tf.transpose(B), A)) / rho_trace
    rho_re = (matmul(tf.transpose(A), A) + matmul(tf.transpose(B), B)) / rho_trace
    W = scale * trace(matmul(rho_re, disp_parity_re) - matmul(rho_im, disp_parity_im))
    W *= contrast(tf.math.abs(grid_flat))
    loss = tf.reduce_mean((W_flat - W)**2)
    return loss

### Create constrainted minimization problem (not using constraints here) ----
#-----------------------------------------------------------------------------
class ReconstructionMLE(tfco.ConstrainedMinimizationProblem):
    def __init__(self, loss_fn, weights):
        self._loss_fn = loss_fn
        self._weights = weights

    @property
    def num_constraints(self):
        return 0

    def objective(self):
        return loss_fn()

    def constraints(self):
        A, B = self._weights
        # it works with inequality constraints
        # rho_re = matmul(tf.transpose(A), A) + matmul(tf.transpose(B), B)
        # trace_le_1 = trace(rho_re) - 1
        # trace_gr_1 = 1 - trace(rho_re)
        # constraints = tf.stack([trace_le_1, trace_gr_1])
        # return constraints
        return []

problem = ReconstructionMLE(loss_fn, [A, B])

### Solve the minimization problem -------------------------------------------
#-----------------------------------------------------------------------------
optimizer = tfco.LagrangianOptimizer(
    optimizer=tf.optimizers.Adam(learning_rate=0.1),
    num_constraints=problem.num_constraints)

for i in range(401):
    optimizer.minimize(problem, var_list=[A, B])
    if i % 200 == 0:
        print(f'step = {i}')
        print(f'loss = {loss_fn()}')
        print(f'constraints = {problem.constraints()}')


### Get reconstructed density matrix and Wigner
rho_trace = trace(matmul(tf.transpose(A), A) + matmul(tf.transpose(B), B))
rho_im = (matmul(tf.transpose(A), B) - matmul(tf.transpose(B), A)) / rho_trace
rho_re = (matmul(tf.transpose(A), A) + matmul(tf.transpose(B), B)) / rho_trace
rho = tf.cast(rho_re, c64) + 1j*tf.cast(rho_im, c64)

W_reconstructed = scale * trace(matmul(rho_re, disp_parity_re) - matmul(rho_im, disp_parity_im))
W_reconstructed = tf.reshape(W_reconstructed, grid.shape)

if SAVE_DENSITY_MATRIX:
    np.savez(os.path.join(dmsavedir, fname), 
             rho_re=np.array(rho_re).real, rho_im=np.array(rho_im).real)

### Plot 2D wigner and density matrix ----------------------------------------
#-----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 3, dpi=300, figsize=(5.25,3))
for ax in axes.ravel(): ax.set_aspect('equal')

plot_kwargs = dict(cmap='seismic', vmin=-scale, vmax=scale)
p = axes[0].pcolormesh(xs, ys, np.transpose(W_exp), **plot_kwargs)
p = axes[1].pcolormesh(xs, ys, np.transpose(W_reconstructed), **plot_kwargs)

# plt.colorbar(p)

lim = np.max(np.abs(rho_re))
plot_kwargs = dict(cmap='seismic', vmin=-0.3, vmax=0.3)
axes[2].pcolormesh(np.transpose(rho_re), **plot_kwargs)
        
for i in [0,1]:
    # Set ticks
    ax = axes[i]
    ax.set_xticks(np.arange(-2,3,1)*sqrt(pi/2))
    ax.set_yticks(np.arange(-2,3,1)*sqrt(pi/2))
    ax.set_xticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$'])
    ax.set_yticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$'])

    # Add a grid 
    base = np.arange(-4,5) * sqrt(pi/2)
    X, Y = np.meshgrid(base, base)
    ax.plot(X, Y, color=plt.get_cmap('gray')(0.25), linestyle=':')
    ax.plot(Y, X, color=plt.get_cmap('gray')(0.25), linestyle=':')
    
    lim = 3.2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    
    ax.set_xlabel(r'${\rm Re}\,(\alpha)\,/\,\sqrt{\pi\,/\,2}$')
    ax.set_ylabel(r'${\rm Im}\,(\alpha)\,/\,\sqrt{\pi\,/\,2}$')

dm_ticks = [0,10,20]
axes[2].set_xticks(dm_ticks)
axes[2].set_xticks(dm_ticks)

axes[2].set_xlabel(r'Photon number')
axes[2].set_ylabel(r'Photon number')

plt.tight_layout()

if SAVE_FIGURE:
    figname = os.path.join(figsavedir, 'wigner_reconstruction.pdf')
    fig.savefig(figname)


# Plot contrast calibration measurement
fig, ax = plt.subplots(1, 1, dpi=300, figsize=(1.775,1.775))
ax.set_xlabel(r'${\rm Re}\,(\alpha)\,/\,\sqrt{\pi\,/\,2}$')
ax.set_xticks([-4,-2,0,2,4])
ax.set_ylabel(r'Average msmt. result')
data = np.load(dataname)
ax.plot(data['P_xs']/sqrt(pi/2), data['sx'], marker='x', linestyle='none')
ax.plot(data['P_xs']/sqrt(pi/2), data['sy'], marker='x', linestyle='none')
ax.plot(data['P_xs']/sqrt(pi/2), data['sz'], marker='x', linestyle='none')

ax.plot(data['P_xs']/sqrt(pi/2), contrast(data['P_xs']), color='k', linestyle=':')
ax.plot(data['P_xs']/sqrt(pi/2), contrast(data['P_xs'])**2, color='k', linestyle='--')

plt.tight_layout()

if SAVE_FIGURE:
    figname = os.path.join(figsavedir, 'wigner_contrast.pdf')
    fig.savefig(figname)

### Print some metrics -------------------------------------------------------
#-----------------------------------------------------------------------------

# Purity
purity = real(trace(matmul(rho, rho)))
print(f'Purity {purity}')

# Trace
trace_ = real(trace(rho))
print(f'Trace {trace_}')

# Photon number
nbar = real(trace(matmul(rho,ops.num(N))))
print(f'Photon number {nbar}')

# Eigenvalues
eigvals, eigvecs = tf.linalg.eigh(rho)
eigvals = tf.sort(real(eigvals), direction='DESCENDING')
print(f'Eigenvalues {eigvals}')


# fig, ax = plt.subplots(1,1,dpi=600, figsize=(3,3))
# ax.set_ylabel('Eigenvalue')
# ax.set_xlabel('Index of eigenalue')
# # ax.set_ylim(1e-5,1)
# ax.plot(eigvals.numpy().real, marker='.')
# ax.set_yscale('log')


# # Plot wigner of projector onto 1st pair of eigenvalues
# state0 = tf.concat([eigvecs[:,-1], tf.zeros(140-eigvecs[0,:].shape[0], dtype=c64)], axis=0)
# state0 = tf.expand_dims(state0, 0)
# state1 = tf.concat([eigvecs[:,-2], tf.zeros(140-eigvecs[0,:].shape[0], dtype=c64)], axis=0)
# state1 = tf.expand_dims(state1, 0)
# hf.plot_phase_space(density_matrix(state0)+density_matrix(state1), 
#                     False, phase_space_rep='wigner', op=True)

# # Plot wigner of projector onto 2nd pair of eigenvalues
# state0 = tf.concat([eigvecs[:,-3], tf.zeros(140-eigvecs[0,:].shape[0], dtype=c64)], axis=0)
# state0 = tf.expand_dims(state0, 0)
# state1 = tf.concat([eigvecs[:,-4], tf.zeros(140-eigvecs[0,:].shape[0], dtype=c64)], axis=0)
# state1 = tf.expand_dims(state1, 0)
# hf.plot_phase_space(density_matrix(state0)+density_matrix(state1), 
#                     False, phase_space_rep='wigner', op=True)
