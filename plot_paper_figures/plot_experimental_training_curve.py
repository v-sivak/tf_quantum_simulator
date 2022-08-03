# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 13:14:42 2021

@author: qulab
"""
import pandas as pd
import tensorboard as tb
import matplotlib.pyplot as plt
import h5py
import numpy as np
import plot_config
import os
import matplotlib.gridspec as gridspec

SAVE_FIGURE = False
logdir = r'Z:\shared\tmp\for Vlad\from_vlad\24.06.2022-17.48.55'
use_epoch = 300
evaluation_interval = 20


# Load data
all_epochs = {'training' : [], 'evaluation' : []}
all_reward_data = {'training' : [], 'evaluation' : []}
all_T1_diagnostic_data = {'training' : [], 'evaluation' : []}
all_cavity_phases = {'training' : [], 'evaluation' : []}
all_betas = {'training' : [], 'evaluation' : []}
all_detunings = {'training' : [], 'evaluation' : []}
all_alphas = {'training' : [], 'evaluation' : []}

# shape = (1247, 3, 2, 100, 10) : (N_epochs, [m1,m2,m0], [Sx, Sz], N_msmts, N_candidates)

for name in os.listdir(logdir):
    for epoch_type in ['training', 'evaluation']:
        if epoch_type in name:
            data = np.load(os.path.join(logdir, name), allow_pickle=True, encoding='latin1')
            
            all_epochs[epoch_type].append(data['epoch'])
            all_reward_data[epoch_type].append(data['reward_data'].mean(axis=2))
            all_T1_diagnostic_data[epoch_type].append(data['m_diagnostic'])
            all_cavity_phases[epoch_type].append(data['action_batch'][()]['cavity_phase'])
            all_betas[epoch_type].append(data['action_batch'][()]['CD_amplitude'])
            all_detunings[epoch_type].append(data['action_batch'][()]['ge_detune'])
            all_alphas[epoch_type].append(data['action_batch'][()]['alpha'])

all_epochs = {k : np.array(v) for k,v in all_epochs.items()}
all_reward_data = {k : np.array(v) for k,v in all_reward_data.items()}
all_T1_diagnostic_data = {k : np.array(v) for k,v in all_T1_diagnostic_data.items()}
all_cavity_phases = {k : np.array(v) for k,v in all_cavity_phases.items()}
all_betas = {k : np.array(v) for k,v in all_betas.items()}
all_alphas = {k : np.array(v) for k,v in all_alphas.items()}
all_detunings = {k : np.array(v) for k,v in all_detunings.items()}
all_epochs['evaluation'] *= evaluation_interval




### Plot the results
palette = plt.get_cmap('tab10')

fig, axes = plt.subplots(1, 3, figsize = (3.375, 1.5), dpi=600)


ax = axes[0]
# ax.set_ylabel(r'Avg. reward after $T=160$ steps')
ax.set_ylim(0, 0.5)
ax.set_xticks([0,250,500])
ax.set_yticks([0.0, 0.15, 0.30, 0.45])

ind = np.arange(0, np.max(all_epochs['training']), 4) # to avoid plotting every epoch
returns_stochastic = np.mean(all_reward_data['training'][:,1], axis=(1,2))

returns_deterministic = np.mean(all_reward_data['evaluation'][:,1], axis=(1,2))
ax.plot(all_epochs['evaluation'], returns_deterministic, 
        label='deterministic', color=palette(3))


### Plot the results
ax = axes[1]

ax.set_xlabel('Epoch')
ax.set_xticks([0,250,500])
# ax.set_ylabel('ECD amplitude')

eps1 = all_betas['evaluation'].squeeze()[:,0,0] + 1j*all_betas['evaluation'].squeeze()[:,0,1]
ax.plot(all_epochs['evaluation'], eps1.real, linestyle='--',  color=palette(0))
ax.plot(all_epochs['evaluation'], eps1.imag, linestyle='-', color=palette(0))

eps2 = all_betas['evaluation'].squeeze()[:,2,0] + 1j*all_betas['evaluation'].squeeze()[:,2,1]
ax.plot(all_epochs['evaluation'], eps2.real, linestyle='--',  color=palette(3))
ax.plot(all_epochs['evaluation'], eps2.imag, linestyle='-', color=palette(3))


### Plot the results
ax = axes[2]

# ax.set_ylabel('Large displacement\n amplitude')
ax.set_xticks([0,250,500])

ax.plot(all_epochs['evaluation'], all_alphas['evaluation'].squeeze()[:,0], 
        linestyle='-',  color=palette(0))
ax.plot(all_epochs['evaluation'], all_alphas['evaluation'].squeeze()[:,1], 
        linestyle='-',  color=palette(2))
ax.plot(all_epochs['evaluation'], all_alphas['evaluation'].squeeze()[:,2], 
        linestyle='-',  color=palette(3))

plt.tight_layout()

if SAVE_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\fig2_qec_circuit_and_rl_training'
    savename = 'learning_progress'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')
