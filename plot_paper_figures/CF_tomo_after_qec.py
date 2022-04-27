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
    r'Z:\\shared\\tmp\\for Vlad\\from_vlad\\20220305_.npz',
    r'Z:\\shared\\tmp\\for Vlad\\from_vlad\\20220308_T4.npz',
    r'Z:\\shared\\tmp\\for Vlad\\from_vlad\\20220308_T12.npz',
    r'Z:\\shared\\tmp\\for Vlad\\from_vlad\\20220307_T80.npz',
    r'Z:\\shared\\tmp\\for Vlad\\from_vlad\\20220309_T280.npz',
    ]

names = [
    r'$T=0$',
    r'$T=4$',
    r'$T=12$',
    r'$T=80$',
    r'$T=200$',
    ]

SAVE_FIGURE = False
SHOW_TICKS = False

# Plot 2D Wigner
fig, axes = plt.subplots(1, 5, sharey=True, sharex=True, figsize=(7, 2.4), dpi=600)
for j in range(5):
    data = np.load(CF_files[j])
    CF, i, q = data['CF'], data['i'], data['q']
    # Alec-style, use the property CF(x) = CF*(-x) to average the data
    # CF = (np.rot90(CF, 2) + CF)/2

    ax = axes.ravel()[j] 
    ax.set_aspect('equal')
    ax.text(-5.2, 4, names[j])
    
    if SHOW_TICKS:
        if j == 2: ax.set_xlabel(r'${\rm Re}\,(\beta)\,/\,\sqrt{2\pi}$')
        if j == 0: ax.set_ylabel(r'${\rm Im}\,(\beta)\,/\,\sqrt{2\pi}$')

        ax.set_xticks(np.arange(-2,3,1)*sqrt(2*pi))
        ax.set_yticks(np.arange(-2,3,1)*sqrt(2*pi))
        ax.set_xticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$'])
        ax.set_yticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$'])
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        
    # plot_kwargs = dict(cmap='RdBu_r', vmin=-1, vmax=1)
    plot_kwargs = dict(cmap='seismic', vmin=-1, vmax=1)
    p = ax.pcolormesh(i, q, np.transpose(CF), **plot_kwargs)
    
    # Add a grid 
    base = np.arange(-4,5) * sqrt(2*pi)
    xs = ys = base
    X, Y = np.meshgrid(xs, ys)
    ax.plot(X, Y, color='grey', linestyle=':')
    ax.plot(Y, X, color='grey', linestyle=':')
    
    lim = 5.5
    ax.set_xlim(-lim,lim)
    ax.set_ylim(-lim,lim)


plt.tight_layout()

fig.savefig(r'E:\VladGoogleDrive\Qulab\Talks\22_04_29_xxx\tomo.png', fmt='png')

if SAVE_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\CF_tomo_after_qec'
    savename = 'CF_tomo_after_qec'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')


# fig, ax = plt.subplots(1, 1, figsize=(4,3.375), dpi=600)
# plt.colorbar(p)
# ax.remove()
# savename = 'CF_tomo_square_code_colorbar'
# fig.savefig(os.path.join(savedir, savename), fmt='pdf')