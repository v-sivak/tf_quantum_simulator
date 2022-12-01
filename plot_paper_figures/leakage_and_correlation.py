# -*- coding: utf-8 -*-
"""
Created on Tue Aug  9 16:29:03 2022
"""
import os
import plot_config
import numpy as np 
import matplotlib.pyplot as plt
from math import sqrt, pi
from scipy.optimize import curve_fit

datadir = os.path.join(plot_config.data_root_dir, 'measurement_outcomes')

### LOAD DATA
data = dict(np.load(os.path.join(datadir, 'sample_for_analysis.npz')))

SAVE_FIGURE = False

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
### Analyse correlation between measurements 
fig, axes = plt.subplots(1,3, dpi=600, figsize=(7,3))

ax = axes[0]
ax.set_xticks([0,10,20,30])
ax.set_yticks([0,10,20,30])
ax.set_ylabel('Time (cycles)')
r = np.corrcoef(data['g'].transpose())
ax.set_aspect('equal')
p = ax.pcolormesh(r[:30,:30], cmap='seismic', vmin=-1, vmax=1, rasterized=True)

ax = axes[1]
ax.set_xticks([0,500,1000])
ax.set_yticks([0,500,1000])
ax.set_xlabel('Time (cycles)')
r = np.corrcoef(data['g'].transpose())
ax.set_aspect('equal')
p = ax.pcolormesh(r, cmap='seismic', vmin=-0.1, vmax=0.1, rasterized=True)

ax = axes[2]
ax.set_xticks([0,500,1000])
ax.set_yticks([0,500,1000])
# # 1) index of trajectories that have any leakage event at any time step
# ind = np.where(np.any(data['f'], axis=1)==False)[0]
# 2) index of trajectories that have two consecutive leakage events anywhere
two_leakage_events = np.logical_and(data['f'], np.roll(data['f'], 1, axis=1))
ind = np.where(np.any(two_leakage_events, axis=1)==False)[0]
r = np.corrcoef(data['g'][ind].transpose())
ax.set_aspect('equal')
p = ax.pcolormesh(r[:-1,:-1], cmap='seismic', vmin=-0.1, vmax=0.1, rasterized=True)

plt.tight_layout()
 
savename = os.path.join(plot_config.save_root_dir, 
                        r'measurements_statistics\correlation.pdf')
if SAVE_FIGURE: fig.savefig(savename)



### Analyse correlation between measurements 
fig, axes = plt.subplots(1,2, dpi=600, figsize=(7,1.8), sharey=True)

avg_window = 20
locations = [800,200,40]

ax = axes[0]
r = np.corrcoef(data['g'].transpose())
ax.set_xscale('log')

for j in locations:
    r_avg = np.mean([r[i-j:i,i] for i in range(j,j+avg_window)], axis=0)
    ax.plot(1+np.arange(j), np.flip(r_avg), marker='.', markersize=3.5,
            label=str(j))

ax.plot(np.zeros(800))
# ax.set_xlabel('Separation (cycles)')
ax.set_ylabel('Correlation coefficient, '+r'$r_{ij}$')

ax.legend(loc='upper right', 
          title='Location of '+ r'$2^{\rm nd}$' + '\n' + 'measurement, '+r'$j$')

ax = axes[1]
# # 1) index of trajectories that have any leakage event at any time step
# ind = np.where(np.any(data['f'], axis=1)==False)[0]
# 2) index of trajectories that have two consecutive leakage events anywhere
two_leakage_events = np.logical_and(data['f'], np.roll(data['f'], 1, axis=1))
ind = np.where(np.any(two_leakage_events, axis=1)==False)[0]
r = np.corrcoef(data['g'][ind].transpose())
ax.set_xscale('log')

for j in locations:
    r_avg = np.mean([r[i-j:i,i] for i in range(j,j+avg_window)], axis=0)
    ax.plot(1+np.arange(j), np.flip(r_avg), marker='.', markersize=3.5)
ax.plot(np.zeros(800))

# Fit logcal maxima to exponential decay to extract correlation length
def exp(t, a, T):
    return a * np.exp(-t/T)
even_ind = np.where((1+np.arange(j)) % 2 == 0)[0]
popt, pcov = curve_fit(exp, (1+np.arange(j))[even_ind], np.flip(r_avg)[even_ind])
print('Correlation length %.1f cycles' %popt[-1])
# ax.plot(1+np.arange(j), exp(1+np.arange(j), *popt), color='k', linestyle='--')


plt.tight_layout()
savename = os.path.join(plot_config.save_root_dir, 
                        r'measurements_statistics\corr_avg.pdf')
if SAVE_FIGURE: fig.savefig(savename)


Pi = (data['g']*np.roll(data['g'],1,axis=1)).mean(axis=0)
Pi_prod = data['g'].mean(axis=0)**2
print('Expectation of code projector: %.3f +- %.3f' %(Pi.mean(), Pi.std()))
print('Projector from product: %.3f +- %.3f' %(Pi_prod.mean(), Pi_prod.std()))
