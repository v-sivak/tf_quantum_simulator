# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 09:14:02 2022
"""
import os
import plot_config
import numpy as np 
import matplotlib.pyplot as plt
from math import sqrt, pi

datadir = os.path.join(plot_config.data_root_dir, 'readout')

SAVE_FIGURE = False
SAVE_T1_VS_NBAR = False

data = np.load(os.path.join(datadir, 'readout.npz'))
thresh0 = data['thresh0']
thresh1 = data['thresh1']


### FIGURE 1: READOUT HISTOGRAMS

fig, axes = plt.subplots(1, 3, dpi=600, sharey=True, figsize=(5.5,2.5))
# fig, axes = plt.subplots(1, 3, dpi=600, sharey=True, figsize=(7,3))
max_val = 0
for i, s in enumerate(['g', 'e', 'f']):
    x = data[s].real
    y = data[s].imag
    
    x_range = [-0.024,0.024]
    y_range = [-0.024,0.024]
    
    H, xedges, yedges = np.histogram2d(x, y, bins=[101,101], range=[x_range,y_range])    
    max_val = max([max_val, np.max(H)])
    
    ax = axes[i]
    ax.set_aspect('equal')
    p = ax.pcolormesh(xedges, yedges, np.log10(1+H.transpose()), 
                      cmap='Reds', rasterized=True, vmin=0, vmax=np.log10(max_val))
    
    ax.plot([thresh0, thresh0], y_range, color=plt.get_cmap('Set2')(0), linestyle='--')
    ax.plot(x_range, [thresh1, thresh1], color=plt.get_cmap('Set2')(0), linestyle='--')
    
    ax.set_xticks(np.linspace(*x_range, 3))
    ax.set_yticks(np.linspace(*x_range, 3))
    
axes[0].set_ylabel('Q (ADC units)')
axes[1].set_xlabel('I (ADC units)')

# plt.colorbar(p)

plt.tight_layout()
savename = os.path.join(plot_config.save_root_dir, 
                        r'readout_characterization\histograms.pdf')
if SAVE_FIGURE: fig.savefig(savename)


m = np.zeros([3,3])

Ng = len(data['g'])
m[0,0] = np.sum((data['g'].real < thresh0)) / Ng
m[0,1] = np.sum(np.logical_and(data['g'].real > thresh0, data['g'].imag < thresh1)) / Ng
m[0,2] = np.sum(np.logical_and(data['g'].real > thresh0, data['g'].imag > thresh1)) / Ng

Ne = len(data['e'])
m[1,0] = np.sum((data['e'].real < thresh0)) / Ne
m[1,1] = np.sum(np.logical_and(data['e'].real > thresh0, data['e'].imag < thresh1)) / Ne
m[1,2] = np.sum(np.logical_and(data['e'].real > thresh0, data['e'].imag > thresh1)) / Ne

Nf = len(data['f'])
m[2,2] = np.sum((data['f'].imag > thresh1)) / Nf
m[2,1] = np.sum(np.logical_and(data['f'].real > thresh0, data['f'].imag < thresh1)) / Nf
m[2,0] = np.sum(np.logical_and(data['f'].real < thresh0, data['f'].imag < thresh1)) / Nf



### FIGURE 2: MARKOV MATRIX

from math import floor, log10

def fexp(f):
    return int(floor(log10(abs(f)))) if f != 0 else 0

def fman(f):
    return f/10**fexp(f)

fig, ax = plt.subplots(1,1, dpi=600, figsize=(2.0,2.0))
ax.matshow(np.zeros([3,3]), cmap='Reds')
# ax.set_ylabel('Start state')
ax.set_xticks([0,1,2])
ax.set_xticklabels(['g', 'e', 'f'])
# ax.set_xlabel('End state')
ax.set_yticks([0,1,2])
ax.set_yticklabels(['g', 'e', 'f'])

ax.plot(np.ones(2)*0.5, [-0.5, 2.5], color=plt.get_cmap('tab10')(0))
ax.plot(np.ones(2)*1.5, [-0.5, 2.5], color=plt.get_cmap('tab10')(0))
ax.plot([-0.5, 2.5], np.ones(2)*0.5, color=plt.get_cmap('tab10')(0))
ax.plot([-0.5, 2.5], np.ones(2)*1.5, color=plt.get_cmap('tab10')(0))

ofs_x = -0.4
ofs_y = 0.075
ax.text(ofs_x+0., ofs_y+0., r'  %.4f' %m[0,0])
ax.text(ofs_x+1., ofs_y+0., r'%.1f$\times 10^{%d}$' %(fman(m[0,1]), fexp(m[0,1])))
ax.text(ofs_x+2., ofs_y+0., r'%.1f$\times 10^{%d}$' %(fman(m[0,2]), fexp(m[0,2])))

ax.text(ofs_x+0., ofs_y+1., r'%.1f$\times 10^{%d}$' %(fman(m[1,0]), fexp(m[1,0])))
ax.text(ofs_x+1., ofs_y+1., r'  %.4f' %m[1,1])
ax.text(ofs_x+2., ofs_y+1., r'%.1f$\times 10^{%d}$' %(fman(m[1,2]), fexp(m[1,2])))

ax.text(ofs_x+0., ofs_y+2., r'%.1f$\times 10^{%d}$' %(fman(m[2,0]), fexp(m[2,0])))
ax.text(ofs_x+1., ofs_y+2., r'%.1f$\times 10^{%d}$' %(fman(m[2,1]), fexp(m[2,1])))
ax.text(ofs_x+2., ofs_y+2., r'  %.4f' %m[2,2])

plt.tight_layout()
savename = os.path.join(plot_config.save_root_dir, 
                        r'readout_characterization\Markov.pdf')
if SAVE_FIGURE: fig.savefig(savename)


### FIGURE 3: T1 VS NBAR
data = np.load(os.path.join(datadir, 'T1_vs_nbar_sweep.npz'))
T1, xs, ys = data['T1'], data['xs'], data['ys']
hours = (ys-ys[0])*24


fig, ax = plt.subplots(1,1, dpi=200, figsize=(1.7,3.75))
ax.set_xlabel(r'$\sqrt{\overline{n}}$ (DAC units)')
ax.set_ylabel('Time (hours)')
hours = (ys-ys[0])*24
p = ax.pcolormesh(xs, hours, T1, cmap='RdYlGn', vmin=0)
ax.set_xticks([0,0.2,0.4])
ax.set_yticks(np.arange(0,max(hours),5))
plt.colorbar(p, orientation='horizontal', label=r'$T_1^{\, t}\,\rm (\mu s)$',
             ticks=[0,50,100,150,200,250])
plt.tight_layout()

savename = os.path.join(plot_config.save_root_dir, 
                        r'readout_characterization\T1_vs_nbar.pdf')
if SAVE_FIGURE: fig.savefig(savename)