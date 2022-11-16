# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:58:24 2021
"""
import os
import plot_config
import numpy as np 
import matplotlib.pyplot as plt
from math import sqrt, pi

### LOAD DATA
datadir = r'E:\data\paper_data\lifetimes_stability'


round_time_us = 4.924
times = dict(np.load(os.path.join(datadir, 'times.npz')))
lifetimes = dict(np.load(os.path.join(datadir, 'lifetimes.npz')))
errorbars = dict(np.load(os.path.join(datadir, 'errorbars.npz')))
readout = dict(np.load(os.path.join(datadir, 'readout.npz')))
T1_vs_n = dict(np.load(os.path.join(datadir, 'T1_vs_nbar.npz')))
Pe = np.load(os.path.join(datadir, 'Pe.npz'))['Pe']


SAVE_FIGURE = False
USE_LEGEND = False
SAVE_GAIN_FIGURE = False


print('Mean Fock T1: %.2f +- %.2f' %(lifetimes['fock_T1_v1'].mean(), lifetimes['fock_T1_v1'].std()))
print('')
print('Mean Fock T2: %.2f +- %.2f' %(lifetimes['fock_T2_v1'].mean(), lifetimes['fock_T2_v1'].std()))
print('')
print('Mean Tmon T1: %.2f +- %.2f' %(lifetimes['tmon_T1'].mean(), lifetimes['tmon_T1'].std()))
print('Mean Tmon T2R: %.2f +- %.2f' %(lifetimes['tmon_T2R'].mean(), lifetimes['tmon_T2R'].std()))
print('Mean Tmon T2E: %.2f +- %.2f' %(lifetimes['tmon_T2E'].mean(), lifetimes['tmon_T2E'].std()))
print('')
print('Mean P(e): %.4f +- %.4f' %(Pe.mean(), Pe.std()))
print('')
print('Median T_phi from tmon: %.2f' %np.median(lifetimes['tmon_T1']/Pe))
print('Median T_phi: %.2f' %np.median(1/(1/lifetimes['fock_T2_v1']-1/2/lifetimes['fock_T1_v1'])))


r = np.corrcoef([1-readout['e'], 1/lifetimes['gkp_Z']])[0,1]
print('Corr. R with readout infidelity of |e>: %.3f' %r)


r = np.corrcoef([1/lifetimes['tmon_T1'], 1/lifetimes['gkp_Z']])[0,1]
print('Corr. R with transmon T1: %.3f' %r)


# SOME PLOTTING CONVENTIONS
label={'tmon_T1' : r'$T_1^{\,t}$', 
       'tmon_T2E' : r'$T_{2E}^{\,t}$', 
       'tmon_T2R' : r'$T_{2R}^{\,t}$',
       'fock_T1_v1' : r'$T_1^{\,c}$',
       'fock_T2_v1' : r'$T_2^{\,c}$',
       'fock_T1_v2' : r'Fock $T_1^{v2}$', 
       'fock_T2_v2' : r'Fock $T_2^{v2}$', 
       'gkp_X' : r'$T_X$', 
       'gkp_Y' : r'$T_Y$', 
       'gkp_Z' : r'$T_Z$'}

marker={'tmon_T1' : '.', 
        'tmon_T2E' : '.', 
        'tmon_T2R' : '.',
        'fock_T1_v1' : '.', 
        'fock_T2_v1' : '.',
        'fock_T1_v2' : '.', 
        'fock_T2_v2' : '.',
        'gkp_X' : '.', 
        'gkp_Y' : '.', 
        'gkp_Z' : '.'}

colors = {'tmon_T1' : '#bc3908', 
          'tmon_T2E' : '#4f772d', 
          'tmon_T2R' : plt.get_cmap('tab20b')(3),
          'fock_T1_v1' : '#e63946', 
          'fock_T1_v2' : plt.get_cmap('Paired')(0), 
          'fock_T2_v1' : '#457b9d',
          'fock_T2_v2' : plt.get_cmap('Paired')(2),
          'gkp_X' : '#f4a261', 
          'gkp_Y' : '#e76f51', 
          'gkp_Z' : '#2a9d8f'}


##############################################################################
##############################################################################
### PLOT LIFETIMES STABILITY
fig, axes = plt.subplots(3,1, dpi=600, gridspec_kw={'height_ratios': [2.23, 1.6, 1.1]}, 
                          sharex=True, figsize=(5,3.9)) # (3.5,2.5)

ax = axes[0]
ax.grid(True, lw=0.5)
tau_max = np.max(np.concatenate([lifetimes[k] for k in lifetimes.keys()]))
ax.set_ylim(0,(tau_max+100)*1e-3)
ax.set_yticks(np.arange(0, (tau_max+70)*1e-3, 0.4))
ax.set_ylabel(r'Lifetime (ms)')
# ax.set_yscale('log')
t0 = np.min(np.concatenate([times[k] for k in times.keys()]))
t1 = np.max(np.concatenate([times[k] for k in times.keys()]))
ax.set_xlim(-0.1,t1-t0+0.1)
for op in ['fock_T1_v1', 'fock_T2_v1', #'fock_T1_v2', 'fock_T2_v2', 
           'tmon_T1', 'tmon_T2E', # 'tmon_T2R', 
            'gkp_Y', 'gkp_Z']: #'gkp_X',
    ax.errorbar(np.sort(np.array(times[op]))-t0, np.array(lifetimes[op])[np.array(times[op]).argsort()]*1e-3, 
            yerr=errorbars[op]*1e-3, linestyle='none', color='black', capsize=1.0)
    ax.plot(np.sort(np.array(times[op]))-t0, np.array(lifetimes[op])[np.array(times[op]).argsort()]*1e-3,
            marker=marker[op], label=label[op], color=colors[op], #linewidth=1,
            linestyle='-' if op in ['gkp_X', 'gkp_Y', 'gkp_Z'] else 'none', linewidth=0.7)
# ax.legend(loc='best', fontsize=6)

ax = axes[1]
ax.grid(True, lw=0.5)
ax.set_yscale('log')
ax.set_ylim(1e1,8e3)
ax.set_ylabel('Inverse read. infideliy')
# ax.set_ylim(0,140)

ax.plot(np.sort(np.array(readout['readout_times']))-t0, 
        1/(1-np.array(readout['e'])[np.array(readout['readout_times']).argsort()]),
        marker='^', color='k', linewidth=0.7, label=r'$|e\rangle$', markersize=3)

ax.plot(np.sort(np.array(readout['readout_times']))-t0, 
        1/(1-np.array(readout['g'])[np.array(readout['readout_times']).argsort()]),
        marker='v', color='k', linewidth=0.7, label=r'$|g\rangle$', markersize=3)

# ax.legend(ncol=2,loc='lower center')


ax = axes[2]
ax.set_xlabel('Time (days)')

ax.set_ylabel(r'$\sqrt{\overline{n}}$ (DAC units)')

T1_vs_n_times, nbars, T1_vs_nbars = T1_vs_n['times'], T1_vs_n['nbars'], T1_vs_n['T1s']
ind = np.argsort(T1_vs_n_times)
T1_vs_n_times, T1_vs_nbars = T1_vs_n_times[ind], T1_vs_nbars[ind]

days = (T1_vs_n_times-T1_vs_n_times[0])
p = ax.pcolormesh(days, nbars, T1_vs_nbars.transpose(), cmap='RdYlGn', vmin=0, vmax=300)
ax.set_ylim(0,0.45)
ax.set_yticks([0,0.2,0.4])
ax.plot(days, np.ones_like(days)*0.22, linestyle='--', color='k')
# plt.colorbar(p, orientation='horizontal', label=r'$T_1^{\, q}$ (us)',
#               ticks=[0,100,200,300])

plt.tight_layout()

# Save figure
if SAVE_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\lifetimes_stability'
    savename = 'lifetimes_stability'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')
    
    

##############################################################################
##############################################################################
### PLOT HISTOGRAM OF LIFETIMES

bins={'tmon_T1' : 25, 
      'tmon_T2E' : 20, 
      'tmon_T2R' : 26,
      'fock_T1_v1' : 20, 
      'fock_T2_v1' : 20,
      'fock_T1_v2' : 26, 
      'fock_T2_v2' : 26,
      'gkp_X' : 30, 
      'gkp_Y' : 30, 
      'gkp_Z' : 30}

colors = {'tmon_T1' : '#bc3908', 
          'tmon_T2E' : '#4f772d', 
          'tmon_T2R' : plt.get_cmap('tab20b')(3),
          'fock_T1_v1' : '#219ebc', 
          'fock_T1_v2' : plt.get_cmap('Paired')(0), 
          'fock_T2_v1' : '#457b9d',
          'fock_T2_v2' : plt.get_cmap('Paired')(2),
          'gkp_X' : '#f4a261', 
          'gkp_Y' : '#e76f51', 
          'gkp_Z' : '#2a9d8f'}

fig, axes = plt.subplots(1,2, figsize=(7,2), dpi=600, gridspec_kw={'width_ratios': [1.8, 1]})
ax = axes[0]
ax.set_xlabel(r'Time constant (ms)')
ax.set_ylabel(r'Counts')
ax.set_yticks([0,5,10,15,20])
for op in ['tmon_T1', 'tmon_T2R', 'tmon_T2E',
           'gkp_Z', 'gkp_Y', 'fock_T1_v1', 'fock_T2_v1']: 
    ax.hist(lifetimes[op]*1e-3, color=colors[op], bins=bins[op], label=label[op])
# ax.legend(ncol=3)

T_phi_v1 = 1/(1/lifetimes['fock_T2_v1']-1/2/lifetimes['fock_T1_v1'])
from_tmon = lifetimes['tmon_T1']/Pe

ax = axes[1]
ax.set_xlabel(r'Oscillator dephasing time (ms)')
# ax.set_ylabel(r'Counts')
ax.set_xlim(3,9.5)
ax.set_yticks([0,5,10])
ax.hist(from_tmon*1e-3, bins=31, label=r'$(\gamma_\varphi^{\,c,t})^{-1}$')
ax.hist(T_phi_v1*1e-3, alpha=0.6, bins=31, label=r'($\gamma_2^{\,c}-\gamma_1^{\,c}/2)^{-1}$')
# ax.legend()
plt.tight_layout()


# Save figure
if SAVE_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures_working\histogram_lifetimes'
    savename = 'lifetimes_histogram'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')






##############################################################################
##############################################################################
### PLOT QEC GAIN   

# First, calculate the gain for a 6-day scan.
gain, delta_gain = [], []
for i_gkp, t in enumerate(times['gkp_X']):
    i_fock_T1 = np.argmin(np.abs(np.array(times['fock_T1_v1'])-t))
    i_fock_T2 = np.argmin(np.abs(np.array(times['fock_T2_v1'])-t))
    
    gamma_fock = 1./3.*(1./lifetimes['fock_T1_v1'][i_fock_T1] + 2./lifetimes['fock_T2_v1'][i_fock_T2])
    gamma_gkp = 1./3.*(1./lifetimes['gkp_X'][i_gkp] + 1./lifetimes['gkp_Y'][i_gkp] + 1./lifetimes['gkp_Z'][i_gkp])
 
    delta_gamma_gkp = 1/3.0 * np.sqrt((errorbars['gkp_X'][i_gkp]/lifetimes['gkp_X'][i_gkp]**2)**2+(errorbars['gkp_Y'][i_gkp]/lifetimes['gkp_Y'][i_gkp]**2)**2+(errorbars['gkp_Z'][i_gkp]/lifetimes['gkp_Z'][i_gkp]**2)**2)
    delta_gamma_fock = 1/3. * np.sqrt(4*(errorbars['fock_T2_v1'][i_fock_T2]/lifetimes['fock_T2_v1'][i_fock_T2]**2)**2+(errorbars['fock_T1_v1'][i_fock_T1]/lifetimes['fock_T1_v1'][i_fock_T1]**2)**2)
    
    this_gain = gamma_fock / gamma_gkp
    this_delta_gain = this_gain * np.sqrt((delta_gamma_gkp/gamma_gkp)**2+(delta_gamma_fock/gamma_fock)**2)
    
    gain.append(this_gain)
    delta_gain.append(this_delta_gain)
    
x_axis = np.array(times['gkp_Z'])-t0
ind = np.argsort(x_axis)
x_axis = x_axis[ind]
gain, delta_gain = np.array(gain)[ind], np.array(delta_gain)[ind]

# Plot this gain
fig, ax = plt.subplots(1,1, figsize=(2.5, 2), dpi=600)
ax.grid(True, lw=0.5)
ax.set_xlabel('Time (days)')
ax.set_ylabel('Gain')
ax.set_ylim(0.0,2.6)
ax.errorbar(x_axis, gain, yerr=delta_gain, linestyle='none', 
            capsize=2.0, color='k')
ax.plot(x_axis, gain, marker='.', linestyle='-', color=plt.get_cmap('Dark2')(0),
        label='Scan #1')

plt.tight_layout()

# Save figure
if SAVE_GAIN_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\channel_fidelity_and_gain'
    savename = 'qec_gain'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')

