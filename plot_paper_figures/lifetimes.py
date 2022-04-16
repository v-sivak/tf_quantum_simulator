# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:58:24 2021
"""
import os
import plot_config
import numpy as np 
import matplotlib.pyplot as plt
from math import sqrt, pi

### Load data
round_time_us = 13.3
times_us = np.load(r'Z:\shared\tmp\for Vlad\from_vlad\lifetimes\times_us.npz')
state = np.load(r'Z:\shared\tmp\for Vlad\from_vlad\lifetimes\state.npz')
fit_params = np.load(r'Z:\shared\tmp\for Vlad\from_vlad\lifetimes\fit_params.npz')
fit_stdev = np.load(r'Z:\shared\tmp\for Vlad\from_vlad\lifetimes\fit_stdev.npz')

USE_LEGEND = True
SAVE_FIGURE = True

### Fit functions
def exp_decay(t, a, b, c):
    return a * np.exp(-t/b) + c

def cos_exp_decay(t, a, b, c, omega, phi):
    return a * np.exp(-t/b) * np.cos(omega*t + phi) + c


palette = plt.get_cmap('tab10')

color = {'transmon_T1' : palette(9), 
        'transmon_T2E' : palette(5), 
        'transmon_T1_n' : palette(7), 
        'fock_T1' : palette(3), 
        'fock_T2' : palette(4), 
        'gkp_X' : palette(0), 
        'gkp_Y' : palette(1), 
        'gkp_Z' : palette(2)}
    
### Plot all this shit
fig, axes = plt.subplots(3, 1, sharex=True, dpi=600, figsize=(4,3))

# Plot transmon
ax = axes[0]
ax.set_ylim(-0.05, 1.05)
ax.set_yticks([0,0.5,1.0])
ax.grid(True)
ax.set_ylabel('P(e)')
for i, bit in enumerate(['transmon_T1', 'transmon_T2E']):
    if bit == 'transmon_T1':
        ax.plot(times_us[bit], exp_decay(times_us[bit], *fit_params[bit]), zorder=0,
                marker='None', color=color[bit], linewidth=0.75)
        ax.plot(times_us[bit], state[bit], marker='.',
                linestyle='none', color=color[bit], markersize=2.0,
                label=r'$T_1^{\,t}$=%.0f $\pm$ %.0f' %(fit_params[bit][1], fit_stdev[bit][1]))

    if bit == 'transmon_T2E':
        ax.plot(times_us[bit], cos_exp_decay(times_us[bit], *fit_params[bit]), zorder=0,
                marker='None', color=color[bit], linewidth=0.75)
        ax.plot(times_us[bit], state[bit], marker='.',
                linestyle='none', color=color[bit], markersize=2.0,
                label=r'$T_{2e}^{\,t}$=%.0f $\pm$ %.0f' %(fit_params[bit][1], fit_stdev[bit][1]))
if USE_LEGEND: ax.legend(loc='right', fontsize=6.5, title='Transmon')

# Plot Fock
ax = axes[1]
ax.set_ylim(-0.05, 1.05)
ax.set_yticks([0,0.5,1.0])
ax.grid(True)
ax.set_ylabel('P(e)')
for i, bit in enumerate(['fock_T1', 'fock_T2']):
    if bit == 'fock_T1':
        ax.plot(times_us[bit], exp_decay(times_us[bit], *fit_params[bit]), zorder=0, 
                marker='None', color=color[bit], linewidth=0.75)
        ax.plot(times_us[bit], state[bit], marker='.', 
                linestyle='none', color=color[bit], markersize=3.0,
                label=r'$T_1^{\,c}$=%.0f $\pm$ %.0f' %(fit_params[bit][1], fit_stdev[bit][1]))
    if bit == 'fock_T2':
        ax.plot(times_us[bit], cos_exp_decay(times_us[bit], *fit_params[bit]), zorder=0, 
                marker='None', color=color[bit], linewidth=0.75)                
        ax.plot(times_us[bit], state[bit], marker='.', 
                linestyle='none', color=color[bit], markersize=3.0,
                label=r'$T_2^{\,c}$=%.0f $\pm$ %.0f' %(fit_params[bit][1], fit_stdev[bit][1]))
if USE_LEGEND: ax.legend(loc='right', fontsize=6.5, title='Fock')

# Plot GKP
ax = axes[2]
ax.set_ylim(-0.05, 1.05)
ax.set_yticks([0,0.5,1.0])
ax.grid(True)
ax.set_ylabel('P(g)')
for i, bit in enumerate(['gkp_X', 'gkp_Y', 'gkp_Z']):
    s = bit[-1]
    ax.plot(times_us[bit], exp_decay(times_us[bit], *fit_params[bit]), zorder=0, 
            marker='None', color=color[bit], linewidth=0.75)
    ax.plot(times_us[bit], state[bit], marker='.', 
            linestyle='none', color=color[bit], markersize=3.0,
            label=r'$T_'+s+r'$=%.0f $\pm$ %.0f' %(fit_params[bit][1], fit_stdev[bit][1]))
if USE_LEGEND: ax.legend(loc='right', fontsize=6.5, title='GKP')


ax.set_xlim(-100,4000)
ax.set_xlabel(r'Time ($\mu$s)')
#------------------------------------------------------------------------------


### Calculate QEC Gain
gamma_gkp  = 1/3.0 * sum([1/fit_params[s][1] for s in ['gkp_X', 'gkp_Y', 'gkp_Z']])
delta_gamma_gkp = 1/3.0 * np.sqrt(sum([(fit_stdev[s][1]/fit_params[s][1]**2)**2 for s in ['gkp_X', 'gkp_Y', 'gkp_Z']]))

gamma_fock = 1/3.0 * sum([1/fit_params[s][1] for s in ['fock_T1', 'fock_T2', 'fock_T2']])
delta_gamma_fock = 1/3.0 * np.sqrt(sum([(fit_stdev[s][1]/fit_params[s][1]**2)**2 for s in ['fock_T1', 'fock_T2', 'fock_T2', 'fock_T2', 'fock_T2']]))

gain = gamma_fock / gamma_gkp
delta_gain = gain * np.sqrt((delta_gamma_gkp/gamma_gkp)**2+(delta_gamma_fock/gamma_fock)**2)

print('QEC Gain: %.2f +- %.2f' %(gain, delta_gain))

# Save figure
if SAVE_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\fig3_lifetimes_and_stability'
    savename = 'Lifetimes'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')