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
round_time_us = 4.924
times = dict(np.load(r'Z:\shared\tmp\for Vlad\from_vlad\lifetimes_stability_2\times.npz'))
lifetimes = dict(np.load(r'Z:\shared\tmp\for Vlad\from_vlad\lifetimes_stability_2\lifetimes.npz'))
errorbars = dict(np.load(r'Z:\shared\tmp\for Vlad\from_vlad\lifetimes_stability_2\errorbars.npz'))
readout = dict(np.load(r'Z:\shared\tmp\for Vlad\from_vlad\lifetimes_stability_2\readout.npz'))

Pe = np.load(r'Z:\shared\tmp\for Vlad\from_vlad\lifetimes_stability_2\Pe.npz')['Pe']
idx = np.argmax(Pe)
Pe = np.delete(Pe, idx)

SAVE_FIGURE = False
USE_LEGEND = False
SAVE_GAIN_FIGURE = True


print('Mean Fock T1 v1: %.2f +- %.2f' %(lifetimes['fock_T1_v1'].mean(), lifetimes['fock_T1_v1'].std()))
print('Mean Fock T1 v2: %.2f +- %.2f' %(lifetimes['fock_T1_v2'].mean(), lifetimes['fock_T1_v2'].std()))
print('')
print('Mean Fock T2 v1: %.2f +- %.2f' %(lifetimes['fock_T2_v1'].mean(), lifetimes['fock_T2_v1'].std()))
print('Mean Fock T2 v2: %.2f +- %.2f' %(lifetimes['fock_T2_v2'].mean(), lifetimes['fock_T2_v2'].std()))
print('')
print('Mean Tmon T1: %.2f +- %.2f' %(lifetimes['tmon_T1'].mean(), lifetimes['tmon_T1'].std()))
print('Mean Tmon T2R: %.2f +- %.2f' %(lifetimes['tmon_T2R'].mean(), lifetimes['tmon_T2R'].std()))
print('Mean Tmon T2E: %.2f +- %.2f' %(lifetimes['tmon_T2E'].mean(), lifetimes['tmon_T2E'].std()))
print('')
print('Mean P(e): %.4f +- %.4f' %(Pe.mean(), Pe.std()))
print('')
print('Median T_phi from tmon: %.2f' %np.median(np.delete(lifetimes['tmon_T1'], np.argmax(Pe))/Pe))
print('Median T_phi v1: %.2f' %np.median(1/(1/lifetimes['fock_T2_v1']-1/2/lifetimes['fock_T1_v1'])))
print('Median T_phi v2: %.2f' %np.median(1/(1/lifetimes['fock_T2_v2']-1/2/lifetimes['fock_T1_v2'])))



### SOME PLOTTING CONVENTIONS

label={'tmon_T1' : r'$T_1^{\,q}$', 
       'tmon_T2E' : r'$T_{2E}^{\,q}$', 
       'tmon_T2R' : r'$T_{2R}^{\,q}$',
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

colors = {'tmon_T1' : plt.get_cmap('tab20b')(17), 
          'tmon_T2E' : plt.get_cmap('tab20b')(0), 
          'tmon_T2R' : plt.get_cmap('tab20b')(3),
          'fock_T1_v1' : plt.get_cmap('Paired')(1), 
          'fock_T1_v2' : plt.get_cmap('Paired')(0), 
          'fock_T2_v1' : plt.get_cmap('Paired')(3),
          'fock_T2_v2' : plt.get_cmap('Paired')(2),
          'gkp_X' : plt.get_cmap('tab20c')(4), 
          'gkp_Y' : plt.get_cmap('tab20')(6), 
          'gkp_Z' : plt.get_cmap('tab20c')(5)}


##############################################################################
##############################################################################
### PLOT LIFETIMES STABILITY
fig, axes = plt.subplots(2,1, dpi=600, gridspec_kw={'height_ratios': [2.23, 1]}, 
                          sharex=True, figsize=(3.5,2.5)) # (3.3,2.5)
ax = axes[0]
ax.grid(True)
tau_max = np.max(np.concatenate([lifetimes[k] for k in lifetimes.keys()]))
ax.set_ylim(0,(tau_max+100)*1e-3)
ax.set_yticks(np.arange(0, (tau_max+70)*1e-3, 0.2))
ax.set_ylabel(r'Lifetime (ms)')
t0 = np.min(np.concatenate([times[k] for k in times.keys()]))
t1 = np.max(np.concatenate([times[k] for k in times.keys()]))
ax.set_xlim(-0.1,t1-t0+0.1)
for op in ['fock_T1_v1', 'fock_T2_v1', #'fock_T1_v2', 'fock_T2_v2', 
           'tmon_T1', 'tmon_T2E', # 'tmon_T2R', 
            'gkp_Y', 'gkp_Z']: #'gkp_X',
    ax.errorbar(np.sort(np.array(times[op]))-t0, np.array(lifetimes[op])[np.array(times[op]).argsort()]*1e-3, 
            yerr=errorbars[op]*1e-3, linestyle='none', color='black', capsize=1.0)
    ax.plot(np.sort(np.array(times[op]))-t0, np.array(lifetimes[op])[np.array(times[op]).argsort()]*1e-3,
            markersize=3.5, marker=marker[op], label=label[op], color=colors[op], #linewidth=1,
            linestyle='-' if op in ['gkp_X', 'gkp_Y', 'gkp_Z'] else 'none')
# ax.legend(loc='best', fontsize=6)

ax = axes[1]
ax.grid(True)
ax.set_xlabel('Time (days)')
ax.set_ylabel(r'$(1-{\cal F}_{\rm read})^{-1}$')
ax.set_ylim(0,110)

ax.plot(np.sort(np.array(readout['readout_times']))-t0, 
        1/(1-np.array(readout['e'])[np.array(readout['readout_times']).argsort()]),
        marker='.', color='k')

plt.tight_layout()

# Save figure
if SAVE_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\fig3_lifetimes_and_stability'
    savename = 'lifetimes_stability_3'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')
    
    

##############################################################################
##############################################################################
### PLOT HISTOGRAM OF LIFETIMES

bins={'tmon_T1' : 26, 
      'tmon_T2E' : 21, 
      'tmon_T2R' : 25,
      'fock_T1_v1' : 26, 
      'fock_T2_v1' : 26,
      'fock_T1_v2' : 26, 
      'fock_T2_v2' : 26,
      'gkp_X' : 26, 
      'gkp_Y' : 26, 
      'gkp_Z' : 26}

fig, axes = plt.subplots(1,2, figsize=(7,2), dpi=600, gridspec_kw={'width_ratios': [1.8, 1]})
ax = axes[0]
ax.set_xlabel(r'Time constant (ms)')
ax.set_ylabel(r'Counts')
ax.set_yticks([0,5,10,15,20])
for op in ['tmon_T1', 'tmon_T2R', 'tmon_T2E',
           'gkp_Z', 'gkp_Y', #'fock_T1_v2', 'fock_T2_v2', 
           'fock_T1_v1', 'fock_T2_v1']: #'gkp_X'
    ax.hist(lifetimes[op]*1e-3, color=colors[op], bins=bins[op], label=label[op])
ax.legend(ncol=3)



T_phi_v1 = np.delete(1/(1/lifetimes['fock_T2_v1']-1/2/lifetimes['fock_T1_v1']), idx)
from_tmon = (np.delete(lifetimes['tmon_T1'], idx)/Pe)

ax = axes[1]
ax.set_xlabel(r'$T_\varphi^{\,c}$ (ms)')
# ax.set_ylabel(r'Counts')
ax.set_xlim(3,9.5)
ax.set_yticks([0,2,4,6])
ax.hist(from_tmon*1e-3, bins=31, label=r'$(\Gamma_\varphi^{\,c,t})^{-1}$')
ax.hist(T_phi_v1*1e-3, alpha=0.6, bins=61, label=r'($\gamma_2^{\,c}-\gamma_1^{\,c}/2)^{-1}$')
ax.legend()
plt.tight_layout()


# Save figure
if SAVE_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\histogram_lifetimes'
    savename = 'lifetimes_histogram'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')








##############################################################################
##############################################################################
### PLOT QEC GAIN   
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


fig, ax = plt.subplots(1,1, figsize=(2.5, 2), dpi=600)
ax.grid(True)
ax.set_xlabel('Time (days)')
ax.set_ylabel('QEC gain')
ax.set_ylim(0.0,2.2)
ax.plot(x_axis, np.ones_like(x_axis), color='black', linestyle='-')
ax.errorbar(x_axis, gain, yerr=delta_gain, linestyle='none', capsize=2.0, color='k')
ax.plot(x_axis, gain, marker='.', linestyle='-')
plt.tight_layout()

# Save figure
if SAVE_GAIN_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\channel_fidelity_and_gain'
    savename = 'qec_gain'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')



# ##############################################################################
# ##############################################################################
# fig, ax = plt.subplots(1,1, figsize=(3.375,2), dpi=600)
# ax.set_yscale('log')
# t = readout['readout_times']
# ind = np.argsort(t)
# ax.plot(t[ind], 1/(1-readout['g'][ind]), marker='.')
# ax.plot(t[ind], 1/(1-readout['e'][ind]), marker='.')
# ax.plot(t[ind], 1/(1-readout['f'][ind]), marker='.')


# fig, ax = plt.subplots(1,1, figsize=(3.375,2), dpi=600)
# ax.set_xlabel('Readout infidelity')
# ax.set_ylabel(r'$1/T_Z$ (ms)')
# e_readout_infidelity = 1-np.array(readout['e'])[ind]
# lifetime_Z = np.array(lifetimes['gkp_Z'])[np.array(times['gkp_Z']).argsort()]

# ax.plot(e_readout_infidelity, 1/lifetime_Z*1e3, marker='o', fillstyle='none', 
#         markersize=2, linestyle='none', color='r')

# # ax.set_ylim(0,110)
# # ax.set_xlim(0,(tau_max+100)*1e-3)

# ax.set_yscale('log')
# ax.set_xscale('log')
# ax.set_yticks([0.5, 1.0, 1.5])

# plt.tight_layout()





# bins={'tmon_T1' : 26, 
#         'tmon_T2E' : 21, 
#         'tmon_T2R' : 21,
#         'fock_T1_v1' : 26, 
#         'fock_T2_v1' : 26,
#         'fock_T1_v2' : 26, 
#         'fock_T2_v2' : 26,
#         'gkp_X' : 26, 
#         'gkp_Y' : 26, 
#         'gkp_Z' : 26}

# fig, axes = plt.subplots(2,1, dpi=600, gridspec_kw={'height_ratios': [2.23, 1]}, 
#                           figsize=(1, 2.5))

# ax = axes[0]
# ax.set_xticklabels([])
# ax.set_yticklabels([])
# ax.set_ylim(0,tau_max+100)

# # ax.set_xlabel(r'Lifetime ($\mu$s)')
# # ax.set_ylabel(r'Counts')
# # ax.set_yticks([0,5,10,15,20])
# for op in ['gkp_Z', 'gkp_Y', #'gkp_X', 
#             'fock_T1_v1', 'fock_T2_v1', #'fock_T1_v2', 'fock_T2_v2', 
#             'tmon_T1', 'tmon_T2R', 'tmon_T2E']:
#     ax.hist(lifetimes[op], color=colors[op], bins=bins[op], orientation='horizontal')


# ax = axes[1]
# ax.set_yticklabels([])
# # ax.set_ylabel('Readout infidelity')
# readout_infidelity = 1-np.array(readout['e'])[np.array(readout['readout_times']).argsort()]
# lifetime_Z = np.array(lifetimes['gkp_Z'])[np.array(times['gkp_Z']).argsort()]

# ax.plot(lifetime_Z*1e-3, 1/readout_infidelity, marker='o', fillstyle='none', 
#         markersize=2, linestyle='none', color='k')

# ax.set_ylim(0,110)
# ax.set_xlim(0,(tau_max+100)*1e-3)
# ax.set_xticks([0,1,2])

# ax.set_xlabel('$T_Z$ (ms)')

# plt.tight_layout()

# # Save figure
# if SAVE_FIGURE:
#     savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\fig3_lifetimes_and_stability'
#     savename = 'histogram_and_correlation'
#     fig.savefig(os.path.join(savedir, savename), fmt='pdf')