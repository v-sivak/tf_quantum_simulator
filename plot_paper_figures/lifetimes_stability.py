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
times = dict(np.load(r'Z:\shared\tmp\for Vlad\from_vlad\lifetimes_stability\times.npz'))
lifetimes = dict(np.load(r'Z:\shared\tmp\for Vlad\from_vlad\lifetimes_stability\lifetimes.npz'))
errorbars = dict(np.load(r'Z:\shared\tmp\for Vlad\from_vlad\lifetimes_stability\errorbars.npz'))
Pe = np.load(r'Z:\shared\tmp\for Vlad\from_vlad\lifetimes_stability\Pe.npz')['Pe']

SAVE_FIGURE = True
USE_LEGEND = False


# Calculate QEC gain
gain, delta_gain = [], []
for i_gkp, t in enumerate(times['gkp_X']):
    i_fock_T1 = np.argmin(np.abs(np.array(times['fock_T1'])-t))
    i_fock_T2 = np.argmin(np.abs(np.array(times['fock_T2'])-t))
    
    gamma_fock = 1./3.*(1./lifetimes['fock_T1'][i_fock_T1] + 2./lifetimes['fock_T2'][i_fock_T2])
    gamma_gkp = 1./3.*(1./lifetimes['gkp_X'][i_gkp] + 1./lifetimes['gkp_Y'][i_gkp] + 1./lifetimes['gkp_Z'][i_gkp])
 
    delta_gamma_gkp = 1/3.0 * np.sqrt((errorbars['gkp_X'][i_gkp]/lifetimes['gkp_X'][i_gkp]**2)**2+(errorbars['gkp_Y'][i_gkp]/lifetimes['gkp_Y'][i_gkp]**2)**2+(errorbars['gkp_Z'][i_gkp]/lifetimes['gkp_Z'][i_gkp]**2)**2)
    delta_gamma_fock = 1/3. * np.sqrt(4*(errorbars['fock_T2'][i_fock_T2]/lifetimes['fock_T2'][i_fock_T2]**2)**2+(errorbars['fock_T1'][i_fock_T1]/lifetimes['fock_T1'][i_fock_T1]**2)**2)
    
    this_gain = gamma_fock / gamma_gkp
    this_delta_gain = this_gain * np.sqrt((delta_gamma_gkp/gamma_gkp)**2+(delta_gamma_fock/gamma_fock)**2)
    
    gain.append(this_gain)
    delta_gain.append(this_delta_gain)


label = {'transmon_T1' : r'Transmon $T_1^{\,t}$', 
        'transmon_T2e' : r'Transmon $T_{2e}^{\,t}$', 
        'transmon_T2' : r'Transmon $T_2^{\,t}$', 
        'transmon_T1_n' : r'Transmon $T_1^{\,t}(\overline{n})$', 
        'fock_T1' : r'Fock $T_1^{\,c}$', 
        'fock_T2' : r'Fock $T_2^{\,c}$', 
        'gkp_X' : r'GKP $T_X$', 
        'gkp_Y' : r'GKP $T_Y$', 
        'gkp_Z' : r'GKP $T_Z$', }

palette = plt.get_cmap('tab10')

color = {'transmon_T1' : palette(9), 
        'transmon_T2e' : palette(5), 
        'transmon_T2' : palette(8), 
        'transmon_T1_n' : palette(7), 
        'fock_T1' : palette(3), 
        'fock_T2' : palette(4), 
        'gkp_X' : palette(0), 
        'gkp_Y' : palette(1), 
        'gkp_Z' : palette(2)}


##############################################################################
##############################################################################
##############################################################################


fig, axes = plt.subplots(2,1, dpi=600, gridspec_kw={'height_ratios': [2.23, 1]}, sharex=True,
                         figsize=(3.7,3))
ax = axes[0]
# ax.set_yscale('log')
ax.grid(True)
tau_max = np.max([lifetimes['gkp_X'],lifetimes['gkp_Z'], lifetimes['fock_T2']])
ax.set_ylim(0,tau_max+50)
ax.set_yticks(np.arange(0, tau_max+50, 200))
ax.set_ylabel(r'Lifetime ($\mu$s)')
t0 = np.min(times['gkp_X'])
t1 = np.max(times['gkp_X'])
ax.set_xlim(-0.1,t1-t0+0.1)

for op in ['fock_T1', 'fock_T2', 'gkp_X', 'gkp_Y', 'gkp_Z', 'transmon_T1', 'transmon_T2e', 'transmon_T1_n']: #'transmon_T2'
    ax.errorbar(np.sort(np.array(times[op]))-t0, np.array(lifetimes[op])[np.array(times[op]).argsort()], 
            yerr=errorbars[op], linestyle='none', color='black', capsize=1.0)
    ax.plot(np.sort(np.array(times[op]))-t0, np.array(lifetimes[op])[np.array(times[op]).argsort()], 
            marker='.', label=label[op], linestyle='-' if op in ['gkp_X', 'gkp_Y', 'gkp_Z'] else 'none', 
            markersize=3.0, color=color[op])
if USE_LEGEND: ax.legend(loc='best', fontsize=5)


ax = axes[1]
ax.grid(True)
ax.set_xlabel('Time (days)')
ax.set_ylabel('QEC gain')
ax.set_yticks([0.8, 1.0, 1.2, 1.4])
#ax.set_ylim(0.8,1.55)
x_axis = np.array(times['gkp_Z'])-t0
ax.plot(x_axis, np.ones_like(x_axis), color='red', linestyle='-')

ax.errorbar(x_axis, gain, yerr=delta_gain, linestyle='none', color='black', capsize=1.0, zorder=10)
ax.plot(x_axis, gain, marker='.', color='black', linestyle='none', markersize=3.0)



# Save figure
if SAVE_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\fig3_lifetimes_and_stability'
    savename = 'lifetimes_stability'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')