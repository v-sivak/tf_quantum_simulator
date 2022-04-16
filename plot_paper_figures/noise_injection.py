# -*- coding: utf-8 -*-
"""
Created on Thu Mar  3 10:42:33 2022

@author: qulab
"""
import os
import plot_config
import numpy as np 
import matplotlib.pyplot as plt
from math import sqrt, pi
from scipy.optimize import curve_fit

def linear(x, a, b):
    return a+b*x

SAVE_FIGURE = False
LEGEND = False
MARKERSIZE = 3

gkp_step_ns = 6.45 * 1e3 #in [ns] to be consistent with other experiments



V_rms = np.load(r'Z:\shared\tmp\for Vlad\from_vlad\noise_injection\V_rms_1.npz')['V_rms']
Pe = np.load(r'Z:\shared\tmp\for Vlad\from_vlad\noise_injection\Pe_1.npz')['Pe']
taus = dict(np.load(r'Z:\shared\tmp\for Vlad\from_vlad\noise_injection\taus_1.npz'))
errorbars = dict(np.load(r'Z:\shared\tmp\for Vlad\from_vlad\noise_injection\errorbars_1.npz'))



labels = {
        'T1' : r'$\gamma_1^{\,t}$',
        'T2' : r'$\gamma_{2e}^{\,t}$',
        'T1_cav' : r'$\gamma_1^{\, c}$',
        'T2_cav' : r'$\gamma_{2}^{\, c}$',
        'T_phi' : r'$\gamma_{\phi}^{\,t}=\gamma_{2e}^{\, t}-\gamma_1^{\, t}\,/\,2$',
        'T_phi_cav' : r'$\gamma_{\phi}^{\,c}=\gamma_2^{\,c}-\gamma^{\,c}_1\,/\,2$',
        'gkp_X' : r'$\gamma_X$',
        'gkp_Y' : r'$\gamma_Y$',
        'gkp_Z' : r'$\gamma_Z$'
        }


palette = plt.get_cmap('tab10')
color = {'T1' : palette(9), 
        'T2' : palette(5), 
        'T_phi' : palette(5), 
        'T1_cav' : palette(3), 
        'T2_cav' : palette(4), 
        'T_phi_cav' : palette(4), 
        'gkp_X' : palette(0), 
        'gkp_Y' : palette(1), 
        'gkp_Z' : palette(2)}


# markers = {
#         'T1' : 'o',
#         'T2' : 'o',
#         'T1_cav' : '^',
#         'T2_cav' : '^',
#         'gkp_X' : 's',
#         'gkp_Y' : 's',
#         'gkp_Z' : 's'
#         }

markers = {
        'T1' : 'o',
        'T2' : 'o',
        'T1_cav' : 'o',
        'T2_cav' : 'o',
        'gkp_X' : 'o',
        'gkp_Y' : 'o',
        'gkp_Z' : 'o'
        }

### PANEL 00

# plot qubit and cavity rates
fig, axes = plt.subplots(2,2, figsize=(4,3.5), dpi=600, sharey='row', 
                         gridspec_kw={'height_ratios': [1.5, 1]})
ax = axes[0,0]
ax.set_ylim(1e-1, 2.5e2)
ax.set_title(r'$\gamma_1^{\,t}$ changing, $\gamma_\phi^{\,t}$ constant')
ax.grid(True)
ax.set_yscale('log')
ax.set_xlabel('Noise voltage RMS (V)')
ax.set_ylabel(r'Error rate (1/ms)')

for m in ['T1', 'T2', 'T1_cav', 'T2_cav']:
    ax.plot(V_rms, 1/(taus[m]*1e-6), label=labels[m], marker=markers[m], 
            linestyle='none', color=color[m], markersize=MARKERSIZE)
    # ax.errorbar(V_rms, 1/(taus[m]*1e-6), yerr=errorbars[m]/(taus[m]**2*1e-6),
    #             linestyle='none', color='black', capsize=1.5)


for m in ['T_phi', 'T_phi_cav']:
    ax.plot(V_rms, 1/(taus[m]*1e-6), label=labels[m], linestyle='--', linewidth=1.5, 
            zorder=0, color=color[m])

    ax.errorbar(V_rms, 1/(taus[m]*1e-6), yerr=errorbars[m]/(taus[m]**2*1e-6),
                linestyle='none', color='black', capsize=1.5)

fock_gamma_phi_due_to_transmon = Pe * (1/taus['T1'] * (1-Pe))
ax.plot(V_rms, fock_gamma_phi_due_to_transmon*1e6, color='k',
        label=r'$\gamma_{\phi}^{\,c}$ due to transmon', linestyle=':', linewidth=1.5, zorder=0)

# if LEGEND: ax.legend()




### PANEL 10
# plot logical rates as a function of gamma_1
ax = axes[1,0]
ax.set_ylim(0,19)
ax.grid(True)
ax.set_xlabel(r'Relaxation + excitation, $\gamma_1^{\,t}$ (1/ms)')
ax.set_ylabel(r'Logical error rate (1/ms)')

for m in ['gkp_X', 'gkp_Y', 'gkp_Z']: 
    ax.errorbar(1/(taus['T1']*1e-6), 1/(taus[m]*1e-6), 
            yerr=errorbars[m]/(taus[m]**2*1e-6), xerr=errorbars['T1']/(taus['T1']**2*1e-6),
            linestyle='none', color='black', capsize=1.5)

    ax.plot(1/(taus['T1']*1e-6), 1/(taus[m]*1e-6), label=labels[m], marker=markers[m], 
            color=color[m], linestyle='none', markersize=MARKERSIZE)

    popt, pcov = curve_fit(linear, 1/(taus['T1']*1e-6), 1/(taus[m]*1e-6))
    eq_label = labels[m] + r'=%.2f + %.2f $\gamma_1^{\,t}$' %tuple(popt)
    ax.plot(1/(taus['T1']*1e-6), linear(1/(taus['T1']*1e-6), *popt), label=eq_label,
            color='black', linestyle='--')

#ax.legend()





V_rms = np.load(r'Z:\shared\tmp\for Vlad\from_vlad\noise_injection\V_rms_2.npz')['V_rms']
Pe = np.load(r'Z:\shared\tmp\for Vlad\from_vlad\noise_injection\Pe_2.npz')['Pe']
taus = dict(np.load(r'Z:\shared\tmp\for Vlad\from_vlad\noise_injection\taus_2.npz'))
errorbars = dict(np.load(r'Z:\shared\tmp\for Vlad\from_vlad\noise_injection\errorbars_2.npz'))


### PANEL 01
# plot qubit and cavity rates
ax = axes[0,1]
ax.set_title(r'$\gamma_1^{\,t}$ constant, $\gamma_\phi^{\,t}$ changing')
ax.grid(True)
ax.set_yscale('log')
ax.set_xlabel('Noise voltage RMS (V)')

for m in ['T1', 'T2', 'T1_cav', 'T2_cav']:
    ax.plot(V_rms, 1/(taus[m]*1e-6), marker=markers[m], linestyle='none',
            color=color[m], markersize=MARKERSIZE) # label=labels[m], 
#    ax.errorbar(V_rms, 1/(taus[m]*1e-6), yerr=errorbars[m]/(taus[m]**2*1e-6),
#                linestyle='none', color='black', capsize=1.5)

for m in ['T_phi', 'T_phi_cav']:
    ax.plot(V_rms, 1/(taus[m]*1e-6), linestyle='--', linewidth=1.5, zorder=0,
            color=color[m]) #label=labels[m], 

    ax.errorbar(V_rms, 1/(taus[m]*1e-6), yerr=errorbars[m]/(taus[m]**2*1e-6),
                linestyle='none', color='black', capsize=1.5)

fock_gamma_phi_due_to_transmon = Pe * (1/taus['T1'] * (1-Pe))
ax.plot(V_rms, fock_gamma_phi_due_to_transmon*1e6, color='k',
        linestyle=':', linewidth=1.5, zorder=0) # label=r'$\gamma_{\phi}^{\rm cav}$ due to transmon'

#ax.legend(ncol=2)


### PANEL 11
# plot logical rates as a function of gamma_phi
ax = axes[1,1]
ax.grid(True)
ax.set_xlabel(r'Pure dephasing, $\gamma_{\phi}^{\,t}$ (1/ms)')

errorbar_gamma_phi = np.sqrt((errorbars['T2']/(taus['T2']**2))**2 + (errorbars['T1']/(taus['T1']**2)/2.0)**2)

for m in ['gkp_X', 'gkp_Y', 'gkp_Z']:
    ax.errorbar(1/(taus['T_phi']*1e-6), 1/(taus[m]*1e-6), 
            yerr=errorbars[m]/(taus[m]**2*1e-6), xerr=errorbar_gamma_phi/1e-6,
            linestyle='none', color='black', capsize=1.5)

    ax.plot(1/(taus['T_phi']*1e-6), 1/(taus[m]*1e-6), marker=markers[m], color=color[m],
            linestyle='none', markersize=MARKERSIZE) # label=labels[m],

    popt, pcov = curve_fit(linear, 1/(taus['T_phi']*1e-6), 1/(taus[m]*1e-6))
    eq_label = labels[m] + r'=%.2f + %.4f $\gamma_\phi^{\,t}$' %tuple(popt)
    ax.plot(1/(taus['T_phi']*1e-6), linear(1/(taus['T_phi']*1e-6), *popt), label=eq_label,
            color='black', linestyle='--')

#ax.legend()
    
if LEGEND: fig.legend()

plt.tight_layout()

# Save figure
if SAVE_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\fig4_noise_injection_experiments'
    savename = 'noise_injection'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')