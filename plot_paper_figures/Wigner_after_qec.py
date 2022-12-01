# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:58:24 2021
"""
import os
import plot_config
import numpy as np 
import matplotlib.pyplot as plt
from math import sqrt, pi

def create_grid(ax, xs):
    ax.plot(xs, np.zeros_like(xs), linestyle=':', color='k')
    ax.plot(np.zeros_like(xs), xs, linestyle=':', color='k')
    ax.plot(xs, np.ones_like(xs)*np.sqrt(np.pi/2), linestyle=':', color='k')
    ax.plot(xs, -np.ones_like(xs)*np.sqrt(np.pi/2), linestyle=':', color='k')
    ax.plot(np.ones_like(xs)*np.sqrt(np.pi/2), xs, linestyle=':', color='k')
    ax.plot(-np.ones_like(xs)*np.sqrt(np.pi/2), xs, linestyle=':', color='k')
    ax.plot(xs, np.ones_like(xs)*np.sqrt(2*np.pi), linestyle=':', color='k')
    ax.plot(xs, -np.ones_like(xs)*np.sqrt(2*np.pi), linestyle=':', color='k')
    ax.plot(np.ones_like(xs)*np.sqrt(2*np.pi), xs, linestyle=':', color='k')
    ax.plot(-np.ones_like(xs)*np.sqrt(2*np.pi), xs, linestyle=':', color='k')


wigner_dir = os.path.join(plot_config.data_root_dir, 'Wigner_tomography')
rounds = np.array([0, 100, 200, 400, 800])
round_time_us = 4.924
contrast = 0.63

SAVE_FIGURE = False


# Plot 2D Wigner
fig, axes = plt.subplots(4, 5, sharex=True, sharey='row', figsize=(7, 6.7), dpi=600)

for j, n in enumerate(rounds):
    ### ----------------------------------------------------------------------
    # Load +Z state
    fname = os.path.join(wigner_dir, '+Z_n='+str(n)+'.npz')
    data = np.load(fname)
    W, xs, ys = data['W'], data['W_xs'], data['W_ys']

    # 1st row: Wigner of +Z state
    ax = axes[0,j]
    ax.set_title('%d cycles; %.2f ms' %(n, n*round_time_us*1e-3))
    ax.set_aspect('equal')
    
    lim = np.max(np.abs(W))
    print('Max wigner contrast, %.2f' %lim)
    plot_kwargs = dict(cmap='seismic', vmin=-contrast, vmax=contrast)
    ax.pcolormesh(xs, ys, np.transpose(W), **plot_kwargs, rasterized=True)

    ax.set_yticks(np.arange(-2,3,1)*np.sqrt(np.pi/2))
    ax.set_yticklabels(np.arange(-2,3,1))

    # 3rd row: Position wavefunction
    ax = axes[2, j]
    if j==0: ax.set_ylabel('Position prob. density')
    ax.plot(xs, W.sum(axis=1))

    # 4th row: Momentum wavefunction    
    ax = axes[3, j]
    if j==0: ax.set_ylabel('Momentum prob. density')
    ax.plot(xs, W.sum(axis=0))

    ax.set_xticks(np.arange(-2,3,1)*np.sqrt(np.pi/2))
    ax.set_xticklabels(np.arange(-2,3,1))
    ### ----------------------------------------------------------------------


    ### ----------------------------------------------------------------------
    # Load -Z state
    fname = os.path.join(wigner_dir, '-Z_n='+str(n)+'.npz')
    data = np.load(fname)
    W, xs, ys = data['W'], data['W_xs'], data['W_ys']
    
    # 1st row: Wigner of -Z state
    ax = axes[1,j]
    ax.set_aspect('equal')

    lim = np.max(np.abs(W))
    print('Max wigner contrast, %.2f' %lim)
    plot_kwargs = dict(cmap='seismic', vmin=-contrast, vmax=contrast)
    ax.pcolormesh(xs, ys, np.transpose(W), **plot_kwargs, rasterized=True)

    ax.set_yticks(np.arange(-2,3,1)*np.sqrt(np.pi/2))
    ax.set_yticklabels(np.arange(-2,3,1))

    # 3rd row: Position wavefunction
    ax = axes[2, j]
    ax.set_yticks([0,10,20])
    if j==0: ax.set_ylabel('Position prob. density')
    ax.plot(xs, W.sum(axis=1))

    # 4th row: Momentum wavefunction
    ax = axes[3, j]
    ax.set_yticks([0,5,10])
    if j==0: ax.set_ylabel('Momentum prob. density')
    ax.plot(xs, W.sum(axis=0))

    ax.set_xticks(np.arange(-2,3,1)*np.sqrt(np.pi/2))
    ax.set_xticklabels(np.arange(-2,3,1))
    ### ----------------------------------------------------------------------


axes[0,0].set_ylabel(r'${\rm Im}[\alpha]\,/\,\sqrt{\pi\,/\,2}$')
axes[1,0].set_ylabel(r'${\rm Im}[\alpha]\,/\,\sqrt{\pi\,/\,2}$')

# axes[0,2].set_xlabel(r'${\rm Re}[\alpha]\,/\,\sqrt{\pi\,/\,2}$')
# axes[1,2].set_xlabel(r'${\rm Re}[\alpha]\,/\,\sqrt{\pi\,/\,2}$')
axes[2,2].set_xlabel(r'${\rm Re}[\alpha]\,/\,\sqrt{\pi\,/\,2}$')
axes[3,2].set_xlabel(r'${\rm Im}[\alpha]\,/\,\sqrt{\pi\,/\,2}$')

plt.tight_layout()
savename = os.path.join(plot_config.save_root_dir, 
                        r'wigner_tomo_after_qec\big.pdf')
if SAVE_FIGURE: fig.savefig(savename)

