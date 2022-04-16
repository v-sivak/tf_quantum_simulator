# -*- coding: utf-8 -*-
"""
Created on Mon Apr 26 11:01:42 2021
"""
import tensorflow as tf
from tensorflow import complex64 as c64
from math import pi, sqrt
import numpy as np
import operators as ops
# Use the GitHub version of TFCO
# !pip install git+https://github.com/google-research/tensorflow_constrained_optimization
import tensorflow_constrained_optimization as tfco
import matplotlib.pyplot as plt
from gkp_exp_analysis import plot_config

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
# TODO: fix a transposition error.

SAVE_DM = True
SAVE_FIGURE = True
dataname = r'Z:\shared\tmp\for Vlad\from_vlad\20220415_0.npz'
savename = r'E:\data\gkp_sims\PPO\ECD\EXP_Vlad\density_matrix\test.npz'
figname = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\state_reconstruction\reconstruction.pdf'
N = 25 # dimension of the reconstructed density matrix

# fname = r'Z:\\shared\\tmp\\for Vlad\\from_vlad\\20220308_T12.npz'
# data = np.load(r'Z:\\shared\\tmp\\for Vlad\\from_vlad\\20220305_.npz')
# data = np.load(r'Z:\\shared\\tmp\\for Vlad\\from_vlad\\20220309_T280.npz')
# data = np.load(r'Z:\\shared\\tmp\\for Vlad\\from_vlad\\20220308_T4.npz')
# data = np.load(r'Z:\\shared\\tmp\\for Vlad\\from_vlad\\20220307_T80.npz')




### Load experimental CF -----------------------------------------------------
#-----------------------------------------------------------------------------
def get_exp_CF(fname):
    data = np.load(fname)
    CF = data['CF']
    xs, ys = data['i'], data['q']
    return CF, xs, ys

def normalize_CF(CF):
    i0 = int(CF.shape[0]/2)
    contrast = CF[i0,i0]
    return CF / contrast

CF_exp, xs, ys = get_exp_CF(dataname)
CF_exp = normalize_CF(CF_exp)
grid = tf.cast(xs + 1j*ys, c64)

grid_flat = tf.reshape(grid, [-1])
CF_flat = tf.reshape(CF_exp, [-1])
CF_flat = tf.cast(CF_flat, c64)


### Create displacement operator matrices for CF tomography ------------------
#-----------------------------------------------------------------------------
def create_disp_op_tf():
    N_large = 140 # dimension used to compute the tomography matrix
    D = ops.DisplacementOperator(N_large)
    # Convert to lower-dimentional Hilbert space; shape=[N_alpha,N,N]
    disp_op = D(grid_flat)[:,:N,:N] 
    return real(disp_op), imag(disp_op)

disp_re, disp_im = create_disp_op_tf()


### Create parameterization of the density matrix ----------------------------
#-----------------------------------------------------------------------------
A = tf.Variable(tf.random.uniform([N,N]), dtype=tf.float32, name='A')
B = tf.Variable(tf.random.uniform([N,N]), dtype=tf.float32, name='B')

def loss_fn():
    rho_trace = trace(matmul(tf.transpose(A), A) + matmul(tf.transpose(B), B))
    rho_im = (matmul(tf.transpose(A), B) - matmul(tf.transpose(B), A)) / rho_trace
    rho_re = (matmul(tf.transpose(A), A) + matmul(tf.transpose(B), B)) / rho_trace
    CF_re = trace(matmul(rho_re, disp_re) - matmul(rho_im, disp_im))
    CF_im = trace(matmul(rho_re, disp_im) + matmul(rho_im, disp_re))
    loss_re = tf.reduce_mean((real(CF_flat) - CF_re)**2)
    loss_im = tf.reduce_mean((imag(CF_flat) - CF_im)**2)
    loss_trace = (trace(rho_re) - 1)**2
    return loss_re + loss_im #+ loss_trace


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

for i in range(1000):
    optimizer.minimize(problem, var_list=[A, B])
    if i % 200 == 0:
        print(f'step = {i}')
        print(f'loss = {loss_fn()}')
        print(f'constraints = {problem.constraints()}')


### Get reconstructed density matrix and CF
rho_trace = trace(matmul(tf.transpose(A), A) + matmul(tf.transpose(B), B))
rho_im = (matmul(tf.transpose(A), B) - matmul(tf.transpose(B), A)) / rho_trace
rho_re = (matmul(tf.transpose(A), A) + matmul(tf.transpose(B), B)) / rho_trace
rho = tf.cast(rho_re, c64) + 1j*tf.cast(rho_im, c64)

CF_re = trace(matmul(rho_re, disp_re) - matmul(rho_im, disp_im))
CF_im = trace(matmul(rho_re, disp_im) + matmul(rho_im, disp_re))

CF_re_reconstructed = tf.reshape(CF_re, grid.shape)
CF_im_reconstructed = tf.reshape(CF_im, grid.shape)


### Plot 2D CF and density matrix --------------------------------------------
#-----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, dpi=300, figsize=(4.5,3))
for ax in axes.ravel(): ax.set_aspect('equal')

plot_kwargs = dict(cmap='seismic', vmin=-1, vmax=1)
axes[0,0].pcolormesh(xs, ys, np.transpose(real(CF_exp)), **plot_kwargs)
axes[1,0].pcolormesh(xs, ys, np.transpose(imag(CF_exp)), **plot_kwargs)
axes[0,1].pcolormesh(xs, ys, np.transpose(CF_re_reconstructed), **plot_kwargs)
axes[1,1].pcolormesh(xs, ys, np.transpose(CF_im_reconstructed), **plot_kwargs)

plot_kwargs = dict(cmap='seismic', vmin=-0.5, vmax=0.5)
axes[0,2].pcolormesh(np.transpose(rho_re), **plot_kwargs)
axes[1,2].pcolormesh(np.transpose(rho_im), **plot_kwargs)
        
for i in [0,1]:
    for j in [0,1]:
        # Set ticks
        ax = axes[i,j]
        ax.set_xticks(np.arange(-2,3,1)*sqrt(2*pi))
        ax.set_yticks(np.arange(-2,3,1)*sqrt(2*pi))
        ax.set_xticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$'])
        ax.set_yticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$'])

        # Add a grid 
        base = np.arange(-4,5) * sqrt(2*pi)
        X, Y = np.meshgrid(base, base)
        axes[i,j].plot(X, Y, color='grey', linestyle=':')
        axes[i,j].plot(Y, X, color='grey', linestyle=':')
        
        lim = 5.8
        axes[i,j].set_xlim(-lim,lim)
        axes[i,j].set_ylim(-lim,lim)
        
        if i == 1: ax.set_xlabel(r'${\rm Re}\,(\beta)\,/\,\sqrt{2\pi}$')
        if j == 0: ax.set_ylabel(r'${\rm Im}\,(\beta)\,/\,\sqrt{2\pi}$')

        dm_ticks = [0,10,20]
        axes[i,2].set_xticks(dm_ticks)
        axes[i,2].set_xticks(dm_ticks)

axes[1,2].set_xlabel(r'Photon number')

plt.tight_layout()

if SAVE_FIGURE:
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
eigvals, _ = tf.linalg.eigh(rho)
eigvals = tf.sort(real(eigvals), direction='DESCENDING')
print(f'Eigenvalues {eigvals}')


if SAVE_DM:
    np.savez(savename, rho_re=rho_re.numpy(), rho_im=rho_im.numpy())