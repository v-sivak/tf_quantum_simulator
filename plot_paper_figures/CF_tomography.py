# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:58:24 2021
"""
import os
import plot_config
import numpy as np 
import matplotlib.pyplot as plt
from math import sqrt, pi

CF_files = [
    r'Z:\\shared\\tmp\\for Vlad\\from_vlad\\20220303_.npz',
    r'Z:\\shared\\tmp\\for Vlad\\from_vlad\\20220302.npz',
    r'Z:\\shared\\tmp\\for Vlad\\from_vlad\\20220305_.npz',
    ]

names = [
    r'$|+Z\rangle_{\rm sqr}$',
    r'$|+Y\rangle_{\rm sqr}$',
    r'$|-X\rangle_{\rm sqr}$',
    ]

SAVE_FIGURE = False

# Plot 2D Wigner
fig, axes = plt.subplots(1, 3, sharey=True, sharex=True, 
                         figsize=(3.375, 2.4), dpi=600)
for j in range(3):
    data = np.load(CF_files[j])
    CF, i, q = data['CF'], data['i'], data['q']
    # Alec-style, use the property CF(x) = CF*(-x) to average the data
    # CF = (np.rot90(CF, 2) + CF)/2

    ax = axes.ravel()[j] 
    if j == 1: ax.set_xlabel(r'${\rm Re}\,(\beta)\,/\,\sqrt{2\pi}$')
    if j == 0: ax.set_ylabel(r'${\rm Im}\,(\beta)\,/\,\sqrt{2\pi}$')
    ax.set_aspect('equal')
    ax.text(-5.2, 3.7, names[j])
    ax.set_xticks(np.arange(-2,3,1)*sqrt(2*pi))
    ax.set_yticks(np.arange(-2,3,1)*sqrt(2*pi))
    ax.set_xticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$'])
    ax.set_yticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$'])
    
    plot_kwargs = dict(cmap='RdBu_r', vmin=-1, vmax=1)
    p = ax.pcolormesh(i, q, np.transpose(CF), **plot_kwargs)
    
    # Add a grid 
    if j in [0, 1, 2]:
        base = np.arange(-4,5) * sqrt(2*pi)
        xs = ys = base
        X, Y = np.meshgrid(xs, ys)
        ax.plot(X, Y, color='grey', linestyle=':')
        ax.plot(Y, X, color='grey', linestyle=':')
    
    lim = 5.5
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