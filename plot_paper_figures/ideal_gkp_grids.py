# -*- coding: utf-8 -*-
"""
Created on Sun Oct  3 10:45:53 2021
"""
import numpy as np
import matplotlib.pyplot as plt
import plot_config
from math import pi, sqrt

### Square code
fig, axes = plt.subplots(2,3, dpi=200, figsize=(7,4.7), sharey=True, sharex=True)
colors = plt.get_cmap('tab10')

l = sqrt(2*pi)
base = np.arange(-4,5) * l

xs = ys = base
X, Y = np.meshgrid(xs, ys)

blob_size = 36 

for ax in axes.ravel(): 
    ax.set_aspect('equal')
    ax.set_xticks([-2*l,-l, 0, l,2*l])
    # ax.set_xticklabels([r'$-\sqrt{2\pi}$', r'$0$', r'$\sqrt{2\pi}$'])
    ax.set_xticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$'])
    ax.set_yticks([-2*l,-l, 0, l,2*l])
    # ax.set_yticklabels([r'$-\sqrt{2\pi}$', r'$0$', r'$\sqrt{2\pi}$'])
    ax.set_yticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$'])
    ax.scatter(X, Y, color=colors(3), s=blob_size)
    # ax.plot(X, Y, color='grey')
    # ax.plot(Y, X, color='grey')
    
    lim = sqrt(2*pi)*2.4
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

# |+X>
ax = axes[0,0]
ax.set_title(r'$|+X\rangle$')

xs = base + np.sqrt(pi/2)
ys = base
X, Y = np.meshgrid(xs, ys)
ax.scatter(X, Y, color=colors(0), s=blob_size)

xs = base + sqrt(pi/2)
ys = 2*base
X, Y = np.meshgrid(xs, ys)
ax.scatter(X, Y, color=colors(3), s=blob_size)


# |-X>
ax = axes[1,0]
ax.set_title(r'$|-X\rangle$')

xs = base + np.sqrt(pi/2)
ys = base
X, Y = np.meshgrid(xs, ys)
ax.scatter(X, Y, color=colors(3), s=blob_size)

xs = base + sqrt(pi/2)
ys = 2*base
X, Y = np.meshgrid(xs, ys)
ax.scatter(X, Y, color=colors(0), s=blob_size)


# |+Z>
ax = axes[0,2]
ax.set_title(r'$|+Z\rangle$')

ys = base + np.sqrt(pi/2)
xs = base
X, Y = np.meshgrid(xs, ys)
ax.scatter(X, Y, color=colors(0), s=blob_size)

ys = base + sqrt(pi/2)
xs = 2*base
X, Y = np.meshgrid(xs, ys)
ax.scatter(X, Y, color=colors(3), s=blob_size)



# |-Z>
ax = axes[1,2]
ax.set_title(r'$|-Z\rangle$')

ys = base + np.sqrt(pi/2)
xs = base
X, Y = np.meshgrid(xs, ys)
ax.scatter(X, Y, color=colors(3), s=blob_size)

ys = base + sqrt(pi/2)
xs = 2*base
X, Y = np.meshgrid(xs, ys)
ax.scatter(X, Y, color=colors(0), s=blob_size)


# |+Y>
ax = axes[0,1]
ax.set_title(r'$|+Y\rangle$')

ys = 2 * base + np.sqrt(pi/2)
xs = 2 * base + np.sqrt(pi/2)
X, Y = np.meshgrid(xs, ys)
ax.scatter(X, Y, color=colors(3), s=blob_size)

ys = 2 * base - np.sqrt(pi/2)
xs = 2 * base - np.sqrt(pi/2)
X, Y = np.meshgrid(xs, ys)
ax.scatter(X, Y, color=colors(3), s=blob_size)

ys = 2 * base - np.sqrt(pi/2)
xs = 2 * base + np.sqrt(pi/2)
X, Y = np.meshgrid(xs, ys)
ax.scatter(X, Y, color=colors(0), s=blob_size)

ys = 2 * base + np.sqrt(pi/2)
xs = 2 * base - np.sqrt(pi/2)
X, Y = np.meshgrid(xs, ys)
ax.scatter(X, Y, color=colors(0), s=blob_size)


# |-Y>
ax = axes[1,1]
ax.set_title(r'$|-Y\rangle$')

ys = 2 * base + np.sqrt(pi/2)
xs = 2 * base + np.sqrt(pi/2)
X, Y = np.meshgrid(xs, ys)
ax.scatter(X, Y, color=colors(0), s=blob_size)

ys = 2 * base - np.sqrt(pi/2)
xs = 2 * base - np.sqrt(pi/2)
X, Y = np.meshgrid(xs, ys)
ax.scatter(X, Y, color=colors(0), s=blob_size)

ys = 2 * base - np.sqrt(pi/2)
xs = 2 * base + np.sqrt(pi/2)
X, Y = np.meshgrid(xs, ys)
ax.scatter(X, Y, color=colors(3), s=blob_size)

ys = 2 * base + np.sqrt(pi/2)
xs = 2 * base - np.sqrt(pi/2)
X, Y = np.meshgrid(xs, ys)
ax.scatter(X, Y, color=colors(3), s=blob_size)


# plot the grid lines, solid for stabilizer lattice and dashed for Pauli lattice
for ax in axes.ravel(): 

    xs = ys = base
    X, Y = np.meshgrid(xs, ys)
    ax.plot(X, Y, color='grey', zorder=0)
    ax.plot(Y, X, color='grey', zorder=0)

    xs = ys = base + np.sqrt(pi/2)
    X, Y = np.meshgrid(xs, ys)
    ax.plot(X, Y, color='grey', linestyle='--', zorder=0)
    ax.plot(Y, X, color='grey', linestyle='--', zorder=0)


# plot arrows for Paulis
for ax in axes.ravel(): 
    r = l/2
    ax.arrow(0, 0, r.real, r.imag, head_width=0.5, head_length=0.5, head_starts_at_zero=False, 
             length_includes_head=True, color='black')

    r = l/2 * np.exp(1j*np.pi/2)
    ax.arrow(0, 0, r.real, r.imag, head_width=0.5, head_length=0.5, head_starts_at_zero=False, 
             length_includes_head=True, color='black')

    r = l/2 * np.sqrt(2) * np.exp(1j*np.pi/4)
    ax.arrow(0, 0, r.real, r.imag, head_width=0.5, head_length=0.5, head_starts_at_zero=False, 
             length_includes_head=True, color='black')


plt.tight_layout()

savename = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\ideal_gkp_grids\gkp_ideal_grids_sqr'
fig.savefig(savename, dpi=600, fmt='.pdf')





### Hexagonal code
fig, axes = plt.subplots(2,3, dpi=200, figsize=(7,4.7), sharey=True, sharex=True)

l = np.sqrt(4*np.pi/np.sqrt(3)) 

points = np.linspace(-10, 10, 11)



for ax in axes.ravel(): 
    ax.set_aspect('equal')
    ax.set_xticks([-2*l,-l, 0, l,2*l])
    ax.set_xticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$'])
    ax.set_yticks([-2*l,-l, 0, l,2*l])
    ax.set_yticklabels([r'$-2$', r'$-1$', r'$0$', r'$1$', r'$2$'])
    
    
    for i in np.arange(-4,5,1):
        for j in np.arange(-4,5,1):
            r = i*l+j*l*np.exp(1j*np.pi/3)
            ax.scatter(r.real, r.imag, color=colors(3), s=blob_size)
    
    
    lim = l*2.4
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

# plot the grid lines, solid for stabilizer lattice and dashed for Pauli lattice
for ax in axes.ravel(): 
    d = points
    for j in np.arange(-4,5,1):
        ax.plot(d.real, d.imag+l*j*np.sin(np.pi/3), color='grey', zorder=0)
        ax.plot(d.real, d.imag+l*j*np.sin(np.pi/3)+l*np.sin(np.pi/3)/2, color='grey', linestyle='--', zorder=0)
    
    d = points*np.exp(1j*np.pi/3.0)
    for j in np.arange(-4,5,1):
        ax.plot(d.real+l*j, d.imag, color='grey', zorder=0)
        ax.plot(d.real+l*j+l/2, d.imag, color='grey', linestyle='--', zorder=0)

    d = points*np.exp(2j*np.pi/3.0)
    for j in np.arange(-4,5,1):
        ax.plot(d.real+l*j, d.imag, color='grey', zorder=0)
        ax.plot(d.real+l*j+l/2, d.imag, color='grey', linestyle='--', zorder=0)
    

# plot arrows for Paulis
for ax in axes.ravel(): 
    r = l/2
    ax.arrow(0, 0, r.real, r.imag, head_width=0.5, head_length=0.5, head_starts_at_zero=False, 
             length_includes_head=True, color='black')

    r = l/2 * np.exp(1j*np.pi/3)
    ax.arrow(0, 0, r.real, r.imag, head_width=0.5, head_length=0.5, head_starts_at_zero=False, 
             length_includes_head=True, color='black')

    r = l/2 * np.exp(2j*np.pi/3)
    ax.arrow(0, 0, r.real, r.imag, head_width=0.5, head_length=0.5, head_starts_at_zero=False, 
             length_includes_head=True, color='black')


# |+X>
ax = axes[0,0]
ax.set_title(r'$|+X\rangle$')

for i in np.arange(-4,5,1):
    for j in np.arange(-4,5,1):
        c = 3 if j%2 == 0 else 0
        r = i*l+j*l*np.exp(1j*np.pi/3) + l/2
        ax.scatter(r.real, r.imag, color=colors(c), s=blob_size)


# |-X>
ax = axes[1,0]
ax.set_title(r'$|-X\rangle$')

for i in np.arange(-4,5,1):
    for j in np.arange(-4,5,1):
        c = 3 if j%2 == 1 else 0
        r = i*l+j*l*np.exp(1j*np.pi/3) + l/2
        ax.scatter(r.real, r.imag, color=colors(c), s=blob_size)


# |+Z>
ax = axes[0,2]
ax.set_title(r'$|+Z\rangle$')

for i in np.arange(-4,5,1):
    for j in np.arange(-4,5,1):
        c = 3 if j%2 == 0 else 0
        r = (i*l+j*l*np.exp(1j*np.pi/3) + l/2)*np.exp(2j*np.pi/3)
        ax.scatter(r.real, r.imag, color=colors(c), s=blob_size)


# |-Z>
ax = axes[1,2]
ax.set_title(r'$|-Z\rangle$')

for i in np.arange(-4,5,1):
    for j in np.arange(-4,5,1):
        c = 3 if j%2 == 1 else 0
        r = (i*l+j*l*np.exp(1j*np.pi/3) + l/2)*np.exp(2j*np.pi/3)
        ax.scatter(r.real, r.imag, color=colors(c), s=blob_size)


# |+Y>
ax = axes[0,1]
ax.set_title(r'$|+Y\rangle$')

for i in np.arange(-4,5,1):
    for j in np.arange(-4,5,1):
        c = 3 if j%2 == 0 else 0
        r = (i*l+j*l*np.exp(1j*np.pi/3) + l/2)*np.exp(1j*np.pi/3)
        ax.scatter(r.real, r.imag, color=colors(c), s=blob_size)


# |-Y>
ax = axes[1,1]
ax.set_title(r'$|-Y\rangle$')

for i in np.arange(-4,5,1):
    for j in np.arange(-4,5,1):
        c = 3 if j%2 == 1 else 0
        r = (i*l+j*l*np.exp(1j*np.pi/3) + l/2)*np.exp(1j*np.pi/3)
        ax.scatter(r.real, r.imag, color=colors(c), s=blob_size)


plt.tight_layout()

savename = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\ideal_gkp_grids\gkp_ideal_grids_hex'
fig.savefig(savename, dpi=600, fmt='.pdf')