# -*- coding: utf-8 -*-
"""
Created on Sun Sep 11 10:54:41 2022
"""
import os
import plot_config
import numpy as np 
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def exp_decay(t, A, tau):
    return A * np.exp(-t/tau)

def filter_leakage(s0, s1):
    return np.where(s1, 1, 0)

def filter_error_strings(s0, s1, d):
    E = s0
    for i in range(1, d):
        E = np.logical_and(E, np.roll(s0, -2*i, axis=1) == E)
    return np.where(E, 1, 0)

def do_nothing(s0, s1):
    return np.ones(s0.shape).astype(int)


filter_function = {
        'include_leakage' : do_nothing,
        'remove_leakage' : filter_leakage,
        'remove_1x_errors' : lambda s0, s1 : filter_error_strings(s0, s1, 1),
        'remove_2x_errors' : lambda s0, s1 : filter_error_strings(s0, s1, 2),
        'remove_3x_errors' : lambda s0, s1 : filter_error_strings(s0, s1, 3),
        'remove_4x_errors' : lambda s0, s1 : filter_error_strings(s0, s1, 4),
        'remove_5x_errors' : lambda s0, s1 : filter_error_strings(s0, s1, 5)
        }

pdatadir = os.path.join(plot_config.data_root_dir, 
                        'error_postselection_dataset\pre_processed_data')
all_files = os.listdir(pdatadir)
all_rounds, all_states = [], []
for fname in all_files:
    all_rounds.append(int(fname[:-4].split('_')[1]))

all_rounds = np.array(all_rounds)
states = '+Z,+Y'

SAVE_DATA = False

# initialize some dictionaries that will be populated later
pauli_L = {label:{} for label in filter_function.keys()}
rounds = {label:{} for label in filter_function.keys()}
fit_params = {label:{} for label in filter_function.keys()}
fit_std = {label:{} for label in filter_function.keys()}
N_shots = {label:{} for label in filter_function.keys()}

for s in states.split(','):
    for label in filter_function.keys():
        pauli_L[label][s] = []
        N_shots[label][s] = []
        rounds[label][s] = []


for reps in np.sort(all_rounds):
    print('Loading %d rounds.' %reps)
    data = np.load(os.path.join(pdatadir, 'cycles_%d.npz'%reps))
    
    for s in states.split(','):
        for label in filter_function.keys():
            if reps != 0 and label != 'include_leakage':
                s0 = data['mi_'+s+'_s0']
                s1 = data['mi_'+s+'_s1']
                # create a mask == 1 for the full trajectory if there
                # is a desired error signature in this trajectory.
                mask = filter_function[label](s0, s1)
                mask = np.any(mask, axis=1).astype(int)
                ind_mask = np.where(1-mask)[0]
                postselected = data['m2_'+s][ind_mask]
            else:
                postselected = data['m2_'+s]
            
            P_L = 1-2*postselected.mean()
            N = len(postselected)
            if N>200:
                pauli_L[label][s].append(P_L)
                N_shots[label][s].append(N)
                rounds[label][s].append(reps)

for s in states.split(','):
    for label in filter_function.keys():
        pauli_L[label][s] = np.array(pauli_L[label][s])
        N_shots[label][s] = np.array(N_shots[label][s])
        rounds[label][s] = np.array(rounds[label][s])


# Define fit options and do the fit
for s in states.split(','):
    for label in filter_function.keys():
        var = 1 - pauli_L[label][s][1:]**2
        sigma = np.sqrt(var/N_shots[label][s][1:])
        popt, pcov = curve_fit(exp_decay, 
                rounds[label][s][1:], pauli_L[label][s][1:], 
                p0=(0.7, 600.), sigma=sigma)
        fit_params[label][s] = popt
        fit_std[label][s] = np.sqrt(np.diag(pcov))


# Define plot options and make a plot
colors = {'+X': 'blue', '+Y': 'red', '+Z': 'green'}

# Plot logical lifetimes
fig, axes = plt.subplots(1,2, dpi=100)
ax = axes[0]
ax.grid(True)
ax.set_ylabel(r'$\langle X_L\rangle$, $\langle Y_L\rangle$, '+\
              'including SPAM error')
ax.set_xlabel('Round')

ax = axes[1]
ax.grid(True)
ax.set_ylabel('Survival probability')
ax.set_xlabel('Round')
ax.set_yscale('log')

for label in filter_function.keys():
    for s in states.split(','):
        ax = axes[0]
        # plot data points
        ax.plot(rounds[label][s], pauli_L[label][s],
                linestyle='none', color=colors[s], marker='.',
                label=label+', '+s+', T=%.0f' % fit_params[label][s][1])
        # plot the fit lines
        ax.plot(rounds[label][s], 
                exp_decay(rounds[label][s], *fit_params[label][s]),
                linestyle='-', color=colors[s])

        print(s+' '+ label+' T=%.1f' %fit_params[label][s][1])

        # plot error bars
        sigma = np.sqrt((1-pauli_L[label][s]**2)/N_shots[label][s])
        ax.errorbar(rounds[label][s], pauli_L[label][s],
                    yerr=sigma, linestyle='none', color=colors[s], capsize=2)

        # plot survival probability
        ax = axes[1]
        ind = np.nonzero(np.in1d(rounds['include_leakage'][s], rounds[label][s]))[0]
        ax.plot(rounds[label][s], N_shots[label][s]/N_shots['include_leakage'][s].astype(float)[ind],
                linestyle='none', color=colors[s], marker='.')

plt.tight_layout()


if SAVE_DATA:
    savedir = os.path.join(plot_config.data_root_dir, 
                     'error_postselection_dataset\postselected_lifetimes')

    for label in filter_function.keys():
        for s in states.split(','):
            np.savez(os.path.join(savedir, label+'__'+s+'__'+'.npz'), 
                     rounds=rounds[label][s],
                     pauli_L=pauli_L[label][s],
                     fit_params=fit_params[label][s],
                     fit_std=fit_std[label][s],
                     N_shots=N_shots[label][s])