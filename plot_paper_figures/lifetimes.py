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
datadir = os.path.join(plot_config.data_root_dir, 'lifetimes')
times_us = np.load(os.path.join(datadir, 'times_us.npz'))
state = np.load(os.path.join(datadir, 'state.npz'))
fit_params = np.load(os.path.join(datadir, 'fit_params.npz'))
fit_stdev = np.load(os.path.join(datadir, 'fit_stdev.npz'))
round_time_us = 4.924

USE_LEGEND = False
SAVE_FIGURE = False
SAVE_GAIN_FIGURE = False

### DEFINE FIT FUNCTIONS
def exp_decay(t, a, b, c):
    return a * np.exp(-t/b) + c

def cos_exp_decay(t, a, b, c, omega, phi):
    return a * np.exp(-t/b) * np.cos(omega*t + phi) + c

def exp_decay_to_zero(t, a, b):
    return exp_decay(t, a, b, 0.5)


### DEFINE COLORS
palette = plt.get_cmap('tab10')

color = {'transmon_T1' : '#bc3908', #'#cc6900', #plt.get_cmap('tab20b')(17), 
          'transmon_T2E' : '#4f772d', #plt.get_cmap('tab20b')(0), 
          'fock_T1' : '#e63946', #'#8e6e53', #plt.get_cmap('Paired')(1), 
          'fock_T2' : '#457b9d', #'#bc3908', #plt.get_cmap('Paired')(3),
          'gkp_X' : '#f4a261', # plt.get_cmap('tab20c')(4)
          'gkp_Y' : '#e76f51', #plt.get_cmap('tab20')(6), 
          'gkp_Z' : '#2a9d8f', #plt.get_cmap('tab20c')(5),
          'gkp_Y_off' : '#e76f51', #plt.get_cmap('tab20')(6), 
          'gkp_Z_off' : '#2a9d8f' #plt.get_cmap('tab20c')(5)}
          }

### PLOT LIFETIMES
fig, axes = plt.subplots(3, 1, sharex=True, dpi=600, figsize=(3.375,3),
                         gridspec_kw={'height_ratios':[1,1,1.5]}) # (3.1,2.5)
# array of times to plot the fit
fit_times = {bit : np.linspace(times_us[bit][0], times_us[bit][-1], 301) 
             for bit in times_us.keys()}


### PLOT TRANSMON
ax = axes[0]
ax.set_ylim(-0.05, 1.05)
ax.set_yticks([0,0.5,1.0])
ax.grid(True, lw=0.5)
for i, bit in enumerate(['transmon_T1', 'transmon_T2E']):
    if bit == 'transmon_T1':
        ax.plot(fit_times[bit]*1e-3, exp_decay(fit_times[bit], *fit_params[bit]), zorder=0,
                marker='None', color=color[bit], linewidth=0.75)
        ax.plot(times_us[bit]*1e-3, state[bit], marker='.',
                linestyle='none', color=color[bit], markersize=2.0,
                label=r'$T_1^{\,t}$=%.0f $\pm$ %.0f' %(fit_params[bit][1], fit_stdev[bit][1]))
    if bit == 'transmon_T2E':
        ax.plot(fit_times[bit]*1e-3, cos_exp_decay(fit_times[bit], *fit_params[bit]), zorder=0,
                marker='None', color=color[bit], linewidth=0.75)
        ax.plot(times_us[bit]*1e-3, state[bit], marker='.',
                linestyle='none', color=color[bit], markersize=2.0,
                label=r'$T_{2e}^{\,t}$=%.0f $\pm$ %.0f' %(fit_params[bit][1], fit_stdev[bit][1]))
    print(bit + r' = %.0f +- %.0f' %(fit_params[bit][1], fit_stdev[bit][1]))
if USE_LEGEND: ax.legend(loc='right', fontsize=6.5, title='Transmon')


### PLOT FOCK
ax = axes[1]
ax.set_ylim(-0.05, 1.05)
ax.set_yticks([0,0.5,1.0])
ax.grid(True, lw=0.5)
for i, bit in enumerate(['fock_T1', 'fock_T2']):
    if bit == 'fock_T1':
        ax.plot(fit_times[bit]*1e-3, exp_decay(fit_times[bit], *fit_params[bit]), zorder=0, 
                marker='None', color=color[bit], linewidth=0.75)
        ax.plot(times_us[bit]*1e-3, state[bit], marker='.', 
                linestyle='none', color=color[bit], markersize=3.5,
                label=r'$T_1^{\,c}$=%.0f $\pm$ %.0f' %(fit_params[bit][1], fit_stdev[bit][1]))
    if bit == 'fock_T2':
        ax.plot(fit_times[bit]*1e-3, cos_exp_decay(fit_times[bit], *fit_params[bit]), zorder=0, 
                marker='None', color=color[bit], linewidth=0.75)                
        ax.plot(times_us[bit]*1e-3, state[bit], marker='.', 
                linestyle='none', color=color[bit], markersize=3.5,
                label=r'$T_2^{\,c}$=%.0f $\pm$ %.0f' %(fit_params[bit][1], fit_stdev[bit][1]))
    print(bit + r' = %.0f +- %.0f' %(fit_params[bit][1], fit_stdev[bit][1]))
if USE_LEGEND: ax.legend(loc='right', fontsize=6.5, title='Fock')


### PLOT GKP (with QEC)
gkp_rounds = np.round(times_us['gkp_X']/round_time_us)
ax = axes[2]
ax.set_ylim(-0.05, 1.05)
ax.set_yticks([0,0.5,1.0])
ax.grid(True, lw=0.5)
# ax.set_ylabel(r'Prob. of $|+Z\rangle$, $|+Y\rangle$')

for i, bit in enumerate(['gkp_Z', 'gkp_Y', 'gkp_X']):
    s = bit[-1]
    print(bit + r' = %.0f +- %.0f' %(fit_params[bit][1], fit_stdev[bit][1]))
    
    if bit in ['gkp_X']:
        # plot -Pauli state points
        ax.plot(times_us[bit]*1e-3, 1-state[bit], marker='.', 
                linestyle='none', color=color[bit], markersize=3.5,
                label=r'$T_'+s+r'$=%.0f $\pm$ %.0f' %(fit_params[bit][1], fit_stdev[bit][1]))

        # plot fit for +Pauli state
        ax.plot(fit_times[bit]*1e-3, 1-exp_decay_to_zero(fit_times[bit], *fit_params[bit]), zorder=0, 
                marker='None', color=color[bit], linewidth=0.75)
        
    if bit in ['gkp_Z', 'gkp_Y']:
        # plot all points, Y eigenstates don't flip after even number of rounds
        ax.plot(times_us[bit]*1e-3, state[bit], marker='.', 
                linestyle='none', color=color[bit], markersize=3.5,
                label=r'$T_'+s+r'$=%.0f $\pm$ %.0f' %(fit_params[bit][1], fit_stdev[bit][1]))

        # plot fit for +Pauli state
        ax.plot(fit_times[bit]*1e-3, exp_decay_to_zero(fit_times[bit], *fit_params[bit]), zorder=0, 
                marker='None', color=color[bit], linewidth=0.75)

        
bit = 'gkp_X'; print(bit + r' = %.0f +- %.0f' %(fit_params[bit][1], fit_stdev[bit][1]))


### EXTRACT LOGICAL PAULI ERROR PER ROUND
p_XY = 0.5 * round_time_us/fit_params['gkp_Z'][1]
p_YZ = 0.5 * round_time_us/fit_params['gkp_X'][1]
p_ZX = 0.5 * round_time_us/fit_params['gkp_Y'][1]

p_X = (p_ZX + p_XY - p_YZ) / 2
p_Y = (p_XY + p_YZ - p_ZX) / 2
p_Z = (p_ZX + p_YZ - p_XY) / 2

# quadrature error sum
delta2 = 0
for bit in ['gkp_Z', 'gkp_Y', 'gkp_X']:
    delta2 += (round_time_us*fit_stdev[bit][1]/4/fit_params[bit][1]**2)**2
delta_p = np.sqrt(delta2)


### PLOT GKP (without QEC)
for i, bit in enumerate(['gkp_Z_off', 'gkp_Y_off']):
    ax.plot(times_us[bit]*1e-3, 1-state[bit], marker='o', fillstyle='none',
            linestyle='none', markersize=1.5, color=color[bit])

if USE_LEGEND: ax.legend(loc='right', fontsize=6, title='GKP')

ax.set_xlim(-0.100,6.000)
ax.set_xlabel(r'Time (ms)')
plt.tight_layout()

# def rounds_to_time(x):
#     return x * round_time_us * 1e-3
# def time_to_rounds(x):
#     return x * 1e3 / round_time_us
# secax = ax.secondary_xaxis('top', functions=(time_to_rounds, rounds_to_time))
# secax.set_xlabel('angle [rad]')

# Save figure
if SAVE_FIGURE:
    savedir = os.path.join(plot_config.save_root_dir, 'fig3_lifetimes')
    savename = 'Lifetimes'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')


#------------------------------------------------------------------------------


### CALCULATE QEC GAIN
gamma_gkp  = 1/3 * sum([1/fit_params[s][1] for s in ['gkp_X', 'gkp_Y', 'gkp_Z']])
delta_gamma_gkp = 1/3.0 * np.sqrt(sum([(fit_stdev[s][1]/fit_params[s][1]**2)**2 for s in ['gkp_X', 'gkp_Y', 'gkp_Z']]))

gamma_fock = 1/3 * sum([1/fit_params[s][1] for s in ['fock_T1', 'fock_T2', 'fock_T2']])
delta_gamma_fock = 1/3.0 * np.sqrt(sum([(fit_stdev[s][1]/fit_params[s][1]**2)**2 for s in ['fock_T1', 'fock_T1', 'fock_T2', 'fock_T2', 'fock_T2']]))

gamma_tmon = 1/3 * sum([1/fit_params[s][1] for s in ['transmon_T1', 'transmon_T2E', 'transmon_T2E']])
delta_gamma_tmon = 1/3 * np.sqrt(sum([(fit_stdev[s][1]/fit_params[s][1]**2)**2 for s in ['transmon_T1', 'transmon_T1', 'transmon_T2E', 'transmon_T2E', 'transmon_T2E']]))


gain = gamma_fock / gamma_gkp
delta_gain = gain * np.sqrt((delta_gamma_gkp/gamma_gkp)**2+(delta_gamma_fock/gamma_fock)**2)

print('QEC Gain: %.2f +- %.2f' %(gain, delta_gain))


### BAR PLOT OF CHANNEL FIDELITY LIFETIME
fig, ax = plt.subplots(1,1, dpi=600, figsize=(1.38,0.915))
ax.bar([0,1,2], [1e-3/gamma_tmon, 1e-3/gamma_fock, 1e-3/gamma_gkp],
       color='#8ecae6')

delta_T_gkp = delta_gamma_gkp / gamma_gkp**2
delta_T_fock = delta_gamma_fock / gamma_fock**2
delta_T_tmon = delta_gamma_tmon / gamma_tmon**2
# ax.errorbar([0,1,2], [1e-3/gamma_tmon, 1e-3/gamma_fock, 1e-3/gamma_gkp], 
#             yerr=[1e-3*delta_T_tmon, 1e-3*delta_T_fock, 1e-3*delta_T_gkp], 
#             linestyle='none', color='black', capsize=10.0)
ax.set_yticks([0,1,2])
ax.set_xticks([])
ax.set_ylim(0,2)
plt.tight_layout()

if SAVE_FIGURE:
    savedir = os.path.join(plot_config.save_root_dir, 'fig3_lifetimes')
    savename = 'bar_plot'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')


### PLOT EXPECTED PROCESS FIDELITY
fig, ax = plt.subplots(1,1, dpi=600, figsize=(2.5,2))
ax.set_xlabel('Time (ms)')
ax.set_ylabel('Average channel fidelity')
ax.set_yticks([0.5, 0.75,1])

def F_transmon(t):
    return 1/2 + 1/3*np.exp(-t/fit_params['transmon_T2E'][1]) + 1/6*np.exp(-t/fit_params['transmon_T1'][1])

def F_fock(t):
    return 1/2 + 1/3*np.exp(-t/fit_params['fock_T2'][1]) + 1/6*np.exp(-t/fit_params['fock_T1'][1])

def F_gkp(t):
    return 1/2 + 1/6*np.exp(-t/fit_params['gkp_X'][1]) + 1/6*np.exp(-t/fit_params['gkp_Y'][1]) + 1/6*np.exp(-t/fit_params['gkp_Z'][1])


times = np.linspace(0,5000,201)
ax.plot(times*1e-3, F_transmon(times), label=r'$\{|g\rangle, |e\rangle\}$')
ax.plot(times*1e-3, F_fock(times), label=r'$\{|0\rangle, |1\rangle\}$')
ax.plot(times*1e-3, F_gkp(times), label='grid code')
ax.legend()
plt.tight_layout()

# Save figure
if SAVE_GAIN_FIGURE:
    savedir = os.path.join(plot_config.save_root_dir, 'channel_fidelity_and_gain')
    savename = 'channel_fidelity_vs_time'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')
    

    
    
    
    
    

def scale(a):
    return (a-np.mean(a[-5:]))/(a-np.mean(a[-5:]))[0]

fig, ax = plt.subplots(1,1, dpi=600)
ax.set_yscale('log')
ax.set_ylim(1e-2,1.2)

line = '--'
ax.plot(times_us['fock_T1'], scale(state['fock_T1']), linestyle=line, marker='.')
ax.plot(times_us['transmon_T1'], scale(state['transmon_T1']), linestyle=line, marker='.')
ax.plot(times_us['transmon_T2E'], scale(state['transmon_T2E']), linestyle=line, marker='.')
ax.plot(times_us['gkp_X'], scale(state['gkp_X']), linestyle=line, marker='.')
ax.plot(times_us['gkp_Y'], scale(state['gkp_Y']), linestyle=line, marker='.')
ax.plot(times_us['gkp_Z'], scale(state['gkp_Z']), linestyle=line, marker='.')
ax.plot(times_us['fock_T2'], scale(state['fock_T2']), linestyle=line, marker='.')

