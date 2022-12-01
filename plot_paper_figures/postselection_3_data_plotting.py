# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 10:58:24 2021
"""
import os
import plot_config
import numpy as np 
import matplotlib.pyplot as plt
from math import sqrt, pi
from scipy.optimize import curve_fit


def exp_decay(t, A, tau):
    return A * np.exp(-t/tau)

def survival_prob(n, p):
    return p ** n

def log_survival_prob(n, log_p, n0):
    return (n-n0) * log_p

SAVE_FIGURE = False

datadir = os.path.join(plot_config.data_root_dir, 'error_postselection_dataset\postselected_lifetimes')

messy_dict = {}
for f in os.listdir(datadir):
    label, s, _ = f.split('__')
    data = np.load(os.path.join(datadir, f))
    for name in list(data):
        if name not in messy_dict.keys(): messy_dict[name] = {}
        if label not in messy_dict[name].keys(): messy_dict[name][label] = {}
        messy_dict[name][label][s] = data[name]

rounds = messy_dict['rounds']
pauli_L = messy_dict['pauli_L']
fit_params = messy_dict['fit_params']
fit_std = messy_dict['fit_std']
N_shots = messy_dict['N_shots']

labels = ['include_leakage', 'remove_5x_errors', 'remove_4x_errors', 
          'remove_3x_errors', 'remove_2x_errors', 'remove_1x_errors']

# Define plot options and make a plot
colors = {'+Z': plt.get_cmap('Blues'), '+Y': plt.get_cmap('Oranges')}


# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
### Plot main text figure
# Post-selected decay of the logical Pauli eigenstates
SAVE_FIGURE = False

# Plot logical lifetimes
fig, ax = plt.subplots(1,1, dpi=600, figsize=(2.2,2))
ax.set_yscale('log')
# ax.grid(True, lw=0.5, axis='y')
ax.set_yticks([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
ax.set_yticklabels([0.2,0.3, 0.4, 0.5,0.6, 0.7, 0.8])
ax.set_ylabel(r'$\langle Z_L\rangle$, $\langle Y_L\rangle$, '+\
              'including SPAM error')
ax.set_xlabel('Time (cycles)')

ax.set_xlim(-5,170)

for s in ['+Y', '+Z']:
    for i, label in enumerate(np.flip(labels)):    
        color = colors[s](1-(len(labels)-i)/10)
        # plot error bars
        sigma = np.sqrt((1-pauli_L[label][s]**2)/N_shots[label][s])
        ax.errorbar(rounds[label][s], pauli_L[label][s],
                    yerr=sigma, linestyle='none', color=color, capsize=2)

        # plot data points
        ax.plot(rounds[label][s], pauli_L[label][s],
                linestyle='none', marker='.', color=color,
                label=label+', '+s+', T=%.0f' % fit_params[label][s][1])
        # plot the fit lines
        ax.plot(np.arange(165), 
                exp_decay(np.arange(165), *fit_params[label][s]),
                linestyle='-', color=color)
        
plt.tight_layout()

if SAVE_FIGURE:
    savename = os.path.join(plot_config.save_root_dir, 
                            r'fig4_characterization\postselection')
    fig.savefig(savename, fmt='pdf')
    
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------




# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
### Plot survival probability
fig, ax = plt.subplots(1,1, dpi=600, figsize=(2.2,2))
ax.set_ylabel('Post-selection survival probability')
ax.set_xlabel('Time (cycles)')
# ax.set_yscale('log')
# ax.set_ylim(9e-3,1.3)

survival_probs = []
for s in ['+Y']:
    for i, label in enumerate(labels):   
        
        # plot survival probability
        ind = np.nonzero(np.in1d(rounds['include_leakage'][s], rounds[label][s]))[0]
        survived_frac = N_shots[label][s]/N_shots['include_leakage'][s].astype(float)[ind]
        ax.plot(rounds[label][s], survived_frac,
                linestyle='none', marker='.', color=colors[s](1-i/10))

        popt, pcov = curve_fit(log_survival_prob, rounds[label][s][1:], 
                                np.log(survived_frac)[1:], p0=(0.95,0))
        survival_probs.append(np.exp(popt[0]))
        
        t_range = np.linspace(0,150,151)
        ax.plot(t_range, np.exp(log_survival_prob(t_range, *popt)),
                color=colors[s](1-i/10), linestyle='--')
        
plt.tight_layout()
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------



# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
# ----------------------------------------------------------------------------
### Plot supplement figure
fig, axes = plt.subplots(1,3, dpi=600, figsize=(7,2))

# Plot lifetimes of Z and Y eigenstates
ax = axes[2]
ax.set_xticks([1,2,3,4,5])
ax2 = ax.twinx()
ax.grid(lw=0.5)
ax.set_yscale('log')
ax2.set_yscale('log')
ax.set_ylabel('Lifetime (cycles)')
ax2.set_ylabel('Lifetime (ms)')

round_time_us = 4.924
ax.set_ylim(150, 4000)
ax2.set_ylim(150*round_time_us*1e-3, 4000*round_time_us*1e-3)
ax.set_yticks([200, 400, 800, 1600, 3200])
ax.set_yticklabels([200, 400, 800, 1600, 3200])
ax2.set_yticks([1, 2, 4, 8, 16])
ax2.set_yticklabels([1, 2, 4, 8, 16])

ax.minorticks_off()
ax2.minorticks_off()

colors = {'+Z' : '#2a9d8f', '+Y' : '#e76f51'}

lifetimes, lifetimes_err = {}, {}
for s in ['+Z', '+Y']:
    lifetimes[s], lifetimes_err[s] = [], []
    for label in labels:
        lifetimes[s].append(fit_params[label][s][1])
        lifetimes_err[s].append(fit_std[label][s][1])
    lifetimes[s], lifetimes_err[s] = np.array(lifetimes[s]), np.array(lifetimes_err[s])

ax.grid(lw=0.5)
for s in ['+Z', '+Y']:
    ax.plot(np.arange(1,5+1), np.ones(5)*lifetimes[s][0], 
        linestyle='--', color=colors[s])

    for i, label in enumerate(labels[1:]):
        ax.scatter(len(labels)-(1+i), lifetimes[s][i+1], 
                   color=colors[s], marker='.')
            
        ax.errorbar([len(labels)-(1+i)], [lifetimes[s][i+1]],
                    yerr=[lifetimes_err[s][i+1]], linestyle='none', 
                    color=colors[s], capsize=2)

# Plot rejection probability per cycle
ax = axes[0]
ax.set_ylim(5e-4,1e-1)
ax.set_yscale('log')
ax.set_ylabel('Rejection probability per cycle')
ax.plot(np.flip(np.arange(1,5+1,1)), 1-np.array(survival_probs[1:]), marker='.',
        color='#118ab2')


# Plot improvement factor of average channel fidelity lifetime
ax = axes[1]
ax.set_ylabel(r'$\Gamma_{\rm GKP}$ improvement factor')
ax.set_xlabel(r'Length of discarded $e$-strings, $d$')
ax.set_xticks([1,2,3,4,5])

# fidelity decay rate with no post-selection
rate_np = 2/lifetimes['+Z'][0] + 1/lifetimes['+Y'][0]
rates = []
for i, label in enumerate(labels[1:]):
    rate = 2/lifetimes['+Z'][i+1] + 1/lifetimes['+Y'][i+1]
    rates.append(rate_np/rate)
    print(label + ' improvement factor %.2f, survival prob. %.5f' %(rate_np/rate, survival_probs[i+1]))

ax.plot(np.flip(np.arange(1,5+1)), rates, color='#ef476f', marker='.')
ax.plot(np.arange(1,5+1), np.ones(5), linestyle='--', color='k')

plt.tight_layout()

if SAVE_FIGURE:
    savename = os.path.join(plot_config.save_root_dir, 
                            r'postselection_of_errors\plot')
    fig.savefig(savename, fmt='pdf')



### Analyze effect of leakage in this dataset
rate_without_leakage = 2/fit_params['remove_leakage']['+Z'][1] + 1/fit_params['remove_leakage']['+Y'][1]
print('Fidelity lifetime improvement after leakage rejection: %.3f' %(rate_np/rate_without_leakage))

### Plot survival probability
fig, ax = plt.subplots(1,1, dpi=600, figsize=(2.2,2))
ax.set_ylabel('Post-selection survival probability')
ax.set_xlabel('Time (cycles)')

        
# plot survival probability
label = 'remove_leakage'
ind = np.nonzero(np.in1d(rounds['include_leakage'][s], rounds[label][s]))[0]
survived_frac = N_shots[label][s]/N_shots['include_leakage'][s].astype(float)[ind]
ax.plot(rounds[label][s], survived_frac, linestyle='none', marker='.')

popt, pcov = curve_fit(log_survival_prob, rounds[label][s], np.log(survived_frac), p0=(0.9,0))
print('Leakage probability per cycle %.4f' %np.exp(popt[0]))

t_range = np.linspace(0,150,151)
ax.plot(t_range, np.exp(log_survival_prob(t_range, *popt)), linestyle='--')
        
plt.tight_layout()

