# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:58:24 2021
"""
import os
import plot_config
import numpy as np 
import matplotlib.pyplot as plt
from math import sqrt, pi

datadir = r'E:\data\paper_data\Wigner_tomography'

wigner_files = [
    os.path.join(datadir, '+Z_n12.npz'), 
    os.path.join(datadir, '+Y_n12.npz'), 
    os.path.join(datadir, '-X_n12.npz')
    ]

names = [
    r'$|+Z\,\rangle$',
    r'$|+Y\,\rangle$',
    r'$|-X\,\rangle$',
    ]

SAVE_FIGURE = False

# Plot 2D Wigner
fig, axes = plt.subplots(1, 3, sharey=True, sharex=True, 
                         figsize=(3.5, 2.4), dpi=600)

for j in range(3):
    data = np.load(wigner_files[j])
    W, i, q = data['W'], data['W_xs'], data['W_ys']
    print(np.max(W))

    ax = axes.ravel()[j] 
    if j == 1: ax.set_xlabel(r'${\rm Re}\,(\alpha)\,/\,\sqrt{\pi\,/\,2}$')
    if j == 0: ax.set_ylabel(r'${\rm Im}\,(\alpha)\,/\,\sqrt{\pi\,/\,2}$')
    ax.set_aspect('equal')
    # ax.text(-3., 2.4, names[j])
    ax.set_xticks(np.arange(-2,3,1)*sqrt(pi/2))
    ax.set_yticks(np.arange(-2,3,1)*sqrt(pi/2))
    ax.set_xticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$'])
    ax.set_yticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$'])
    
    plot_kwargs = dict(cmap='seismic', vmin=-0.63, vmax=0.63, rasterized=True)
    p = ax.pcolormesh(i, q, np.transpose(W), **plot_kwargs)
    
    # Add a grid 
    base = np.arange(-4,5) * sqrt(pi/2)
    xs, ys = base, base
    X, Y = np.meshgrid(xs, ys)
    ax.plot(X+0.1 * sqrt(pi/2), Y, color=plt.get_cmap('Greys')(0.75), linestyle=':')
    ax.plot(Y, X-0.1 * sqrt(pi/2), color=plt.get_cmap('Greys')(0.75), linestyle=':')
    
    lim = 3.2
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)


plt.tight_layout()

if SAVE_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\fig1_system_and_tomography'
    savename = 'CF_tomo_square_code'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')


# fig, ax = plt.subplots(1, 1, figsize=(4,3.375), dpi=600)
# plt.colorbar(p)
# ax.remove()
# savename = 'CF_tomo_square_code_colorbar'
# fig.savefig(os.path.join(savedir, savename), fmt='pdf')