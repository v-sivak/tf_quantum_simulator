# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:42:33 2022
"""
import os
import plot_config
import numpy as np 
import matplotlib.pyplot as plt
from math import sqrt, pi
from scipy.optimize import curve_fit

def linear(x, a, b):
    return a+b*x


datadir = r'E:\data\paper_data\transmon_noise_injection'

SAVE_MAIN_FIGURE = True
SAVE_VERIFICATION_FIGURE = True


color = {'T1' : plt.get_cmap('tab20b')(17), 
          'T2E' : plt.get_cmap('tab20b')(0), 
          'T1_cav' : plt.get_cmap('Paired')(1), 
          'T2_cav' : plt.get_cmap('Paired')(3),
          'T_phi' : plt.get_cmap('tab20b')(0), 
          'T_phi_cav' : plt.get_cmap('Paired')(3)}


### NOISE VERIFICATION EXPERIMENTS
fig, axes = plt.subplots(1, 3, figsize=(7,2.5), dpi=600)


# Plot changing T1
V_rms = np.load(os.path.join(datadir, 'T1_noise_V_rms.npz'))['V_rms']
Pe = np.load(os.path.join(datadir, 'T1_noise_Pe.npz'))['Pe']
taus = dict(np.load(os.path.join(datadir, 'T1_noise_taus.npz')))
errorbars = dict(np.load(os.path.join(datadir, 'T1_noise_errorbars.npz')))
readout = dict(np.load(os.path.join(datadir, 'T1_noise_readout.npz')))

ax = axes[1]
ax.set_xlabel('Noise voltage RMS (V)')
ax.set_title(r'$\gamma_\varphi^{\,q}$ constant, $\gamma_1^{\,q}$ increasing')
ax.set_ylim(1e-1, 2.5e2)
ax.grid(True, lw=0.5)
ax.set_yscale('log')
ax.set_ylabel(r'Error rate $(1\,/\,\rmms)$')

for m in ['T1', 'T2E', 'T1_cav', 'T2_cav']:
    ax.plot(V_rms, 1/(taus[m]*1e-6), marker='.', linestyle='none', color=color[m])
    ax.errorbar(V_rms, 1/(taus[m]*1e-6), yerr=errorbars[m]/(taus[m]**2*1e-6),
                linestyle='none', color=color[m], capsize=1.5)

for m in ['T_phi', 'T_phi_cav']:
    ax.plot(V_rms, 1/(taus[m]*1e-6), linestyle='--', zorder=0, color=color[m])

    ax.errorbar(V_rms, 1/(taus[m]*1e-6), yerr=errorbars[m]/(taus[m]**2*1e-6),
                linestyle='none', color=color[m], capsize=1.5)

fock_gamma_phi_due_to_transmon = Pe * (1/taus['T1'] * (1-Pe))
ax.plot(V_rms, fock_gamma_phi_due_to_transmon*1e6, color='k', zorder=2,
        label=r'$\gamma_{\phi}^{\,c}$ due to transmon', linestyle=':')

# Plot readout fidelity
ax = axes[2]
ax.set_ylabel('Readout infidelity')
# ax.set_xlabel('Noise voltage RMS (V)')
ax.set_yscale('log')
ax.plot(V_rms, 1-readout['g'], color=plt.get_cmap('Dark2')(0), linestyle='-', label='g')
ax.plot(V_rms, 1-readout['e'], color=plt.get_cmap('Dark2')(5), linestyle='-', label='e')


# Plot changing T2
V_rms = np.load(os.path.join(datadir, 'T2_noise_V_rms.npz'))['V_rms']
ind = np.argsort(V_rms); V_rms = V_rms[ind]
Pe = np.load(os.path.join(datadir, 'T2_noise_Pe.npz'))['Pe'][ind]
taus = dict(np.load(os.path.join(datadir, 'T2_noise_taus.npz')))
errorbars = dict(np.load(os.path.join(datadir, 'T2_noise_errorbars.npz')))
readout = dict(np.load(os.path.join(datadir, 'T2_noise_readout.npz')))
taus = {k:v[ind] for (k,v) in taus.items()}
errorbars = {k:v[ind] for (k,v) in errorbars.items()}
readout = {k:v[ind] for (k,v) in readout.items()}


errorbars['T1_cav'][errorbars['T1_cav'].argmax()] = taus['T1_cav'][errorbars['T1_cav'].argmax()-1]

ax = axes[0]
ax.set_title(r'$\gamma_1^{\,q}$ constant, $\gamma_\varphi^{\,q}$ increasing')
ax.grid(True, lw=0.5)
ax.set_yscale('log')
ax.set_ylim(1e-1, 6e2)

for m in ['T1', 'T2E', 'T1_cav', 'T2_cav']:
    ax.plot(V_rms, 1/(taus[m]*1e-6), marker='.', linestyle='none', color=color[m])
    ax.errorbar(V_rms, 1/(taus[m]*1e-6), yerr=errorbars[m]/(taus[m]**2*1e-6),
                linestyle='none', color=color[m], capsize=1.5)

# for m in ['T_phi', 'T_phi_cav']:
#     ax.plot(V_rms, 1/(taus[m]*1e-6), linestyle='--', zorder=0, color=color[m])

#     ax.errorbar(V_rms, 1/(taus[m]*1e-6), yerr=errorbars[m]/(taus[m]**2*1e-6),
#                 linestyle='none', color=color[m], capsize=1.5)

# fock_gamma_phi_due_to_transmon = Pe * (1/taus['T1'] * (1-Pe))
# ax.plot(V_rms, fock_gamma_phi_due_to_transmon*1e6, color='k', zorder=2,
#         label=r'$\gamma_{\phi}^{\,c}$ due to transmon', linestyle=':')

## Readout fidelity
ax = axes[2]
ax.plot(V_rms, 1-readout['g'], color=plt.get_cmap('Dark2')(0), linestyle='--')
ax.plot(V_rms, 1-readout['e'], color=plt.get_cmap('Dark2')(5), linestyle='--')
# ax.legend()

plt.tight_layout()

# Save figure
if SAVE_VERIFICATION_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\noise_injection'
    savename = 'noise_verification'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')
    
    



### LOGICAL VS PHYSICAL RATES
fig, ax = plt.subplots(1, 1, figsize=(3.375,2), dpi=600)
# ax.set_yscale('log')
# ax.set_xscale('log')
ax.grid(True, lw=0.5)
ax.set_xlabel(r'Physical error rate (1/ms)')
ax.set_ylabel(r'Logical error rate (1/ms)')


# Changing T1 -- reload data
V_rms = np.load(os.path.join(datadir, 'T1_noise_V_rms.npz'))['V_rms']
Pe = np.load(os.path.join(datadir, 'T1_noise_Pe.npz'))['Pe']
taus = dict(np.load(os.path.join(datadir, 'T1_noise_taus.npz')))
errorbars = dict(np.load(os.path.join(datadir, 'T1_noise_errorbars.npz')))
readout = dict(np.load(os.path.join(datadir, 'T1_noise_readout.npz')))

colors = {'gkp_Y' : plt.get_cmap('Oranges')(0.8), 
          'gkp_Z' : plt.get_cmap('Greens')(0.8)}

for m in ['gkp_Y', 'gkp_Z']: 
    ax.errorbar(1/(taus['T1']*1e-6), 1/(taus[m]*1e-6), 
            yerr=errorbars[m]/(taus[m]**2*1e-6), xerr=errorbars['T1']/(taus['T1']**2*1e-6),
            linestyle='none', color='k', capsize=1.5)

    ax.plot(1/(taus['T1']*1e-6), 1/(taus[m]*1e-6), marker='.', 
            color=colors[m], linestyle='none')

    popt, pcov = curve_fit(linear, 1/(taus['T1']*1e-6), 1/(taus[m]*1e-6))
    eq_label = 'T1 noise ' + m + r'=%.2f + %.2f $\gamma_1^{\,t}$' %tuple(popt)
    print(eq_label)
    # ax.plot(1.1/(taus['T1']*1e-6), linear(1.1/(taus['T1']*1e-6), *popt), 
    #         color=colors[m], linestyle='--', zorder=0)

# Changing T2 -- reload data
V_rms = np.load(os.path.join(datadir, 'T2_noise_V_rms.npz'))['V_rms']
ind = np.argsort(V_rms); V_rms = V_rms[ind]
Pe = np.load(os.path.join(datadir, 'T2_noise_Pe.npz'))['Pe'][ind]
taus = dict(np.load(os.path.join(datadir, 'T2_noise_taus.npz')))
errorbars = dict(np.load(os.path.join(datadir, 'T2_noise_errorbars.npz')))
readout = dict(np.load(os.path.join(datadir, 'T2_noise_readout.npz')))

taus = {k:v[ind] for (k,v) in taus.items()}
errorbars = {k:v[ind] for (k,v) in errorbars.items()}
readout = {k:v[ind] for (k,v) in readout.items()}
errorbar_gamma_phi = np.sqrt((errorbars['T2E']/(taus['T2E']**2))**2 + (errorbars['T1']/(taus['T1']**2)/2.0)**2)


colors = {'gkp_Y' : plt.get_cmap('Oranges')(0.6), 
          'gkp_Z' : plt.get_cmap('Greens')(0.6)}

for m in ['gkp_Y', 'gkp_Z']:
    ax.errorbar(1/(taus['T_phi']*1e-6), 1/(taus[m]*1e-6), 
            yerr=errorbars[m]/(taus[m]**2*1e-6), xerr=errorbar_gamma_phi/1e-6,
            linestyle='none', color='k', capsize=1.5)

    ax.plot(1/(taus['T_phi']*1e-6), 1/(taus[m]*1e-6), marker='.',
            color=colors[m], linestyle='none')

    popt, pcov = curve_fit(linear, 1/(taus['T_phi'][:-10]*1e-6), 1/(taus[m][:-10]*1e-6))
    eq_label = 'T2 noise ' + m + r'=%.2f + %.4f $\gamma_\phi^{\,t}$' %tuple(popt)
    print(eq_label)
    # ax.plot(1/(taus['T_phi']*1e-6), linear(1/(taus['T_phi']*1e-6), *popt),
    #         color=colors[m], linestyle='--', zorder=0)

plt.tight_layout()

# Save figure
if SAVE_MAIN_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\noise_injection'
    savename = 'noise_injection_one_panel'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')
    
    