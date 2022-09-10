# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 13:14:42 2021
"""
import matplotlib.pyplot as plt
import h5py
import numpy as np
import plot_config
import os
import matplotlib.gridspec as gridspec


all_runs = {
    0 : r'E:\data\paper_data\rl_training\30.08.2022-19.14.39_main',
    1 : r'Z:\shared\tmp\for Vlad\from_vlad\22.08.2022-11.33.40',
    2 : r'E:\data\paper_data\rl_training\04.08.2022-16.52.24_different_calibrations',
    3 : r'E:\data\paper_data\rl_training\24.06.2022-17.48.55_different calibrations_1'}

SAVE_REWARD_FIGURE = False
SAVE_ACTION_FIGURE = True
use_epoch = 660
evaluation_interval = 20


# Load data
epochs = {}
rewards = {}
cavity_phases = {}
betas = {}
detunings = {}
alphas = {}



# shape = (662, 3, 2, 150, 10) : (N_epochs, [m1,m2,m0], [Sx, Sz], N_msmts, N_candidates)
for run, logdir in all_runs.items():

    epochs[run] = {'training' : [], 'evaluation' : []}
    rewards[run] = {'training' : [], 'evaluation' : []}
    cavity_phases[run] = {'training' : [], 'evaluation' : []}
    betas[run] = {'training' : [], 'evaluation' : []}
    detunings[run] = {'training' : [], 'evaluation' : []}
    alphas[run] = {'training' : [], 'evaluation' : []}    

    for name in os.listdir(logdir):
        for epoch_type in ['training', 'evaluation']:        
            if epoch_type in name:
                data = np.load(os.path.join(logdir, name), allow_pickle=True, encoding='latin1')
                
                epochs[run][epoch_type].append(data['epoch'])
                rewards[run][epoch_type].append(data['reward_data'].mean(axis=2))
                cavity_phases[run][epoch_type].append(data['action_batch'][()]['cavity_phase'])
                betas[run][epoch_type].append(data['action_batch'][()]['CD_amplitude'])
                detunings[run][epoch_type].append(data['action_batch'][()]['ge_detune'])
                alphas[run][epoch_type].append(data['action_batch'][()]['alpha'])

    epochs[run] = {k : np.array(v) for k,v in epochs[run].items()}
    rewards[run] = {k : np.array(v) for k,v in rewards[run].items()}
    cavity_phases[run] = {k : np.array(v) for k,v in cavity_phases[run].items()}
    betas[run] = {k : np.array(v) for k,v in betas[run].items()}
    alphas[run] = {k : np.array(v) for k,v in alphas[run].items()}
    detunings[run] = {k : np.array(v) for k,v in detunings[run].items()}
    epochs[run]['evaluation'] *= evaluation_interval



### PLOT REWARDS
palette = plt.get_cmap('tab10')
colors = {0: palette(3), 1: palette(2)}

fig, ax = plt.subplots(1, 1, figsize = (1.7, 1.25), dpi=600)
ax.set_ylabel('Reward')
ax.set_xlabel('Epoch')

for run in [0]: #all_runs.keys():

    ind = np.arange(0, np.max(epochs[run]['training']), 4) # to avoid plotting every epoch
    returns_stochastic = np.mean(rewards[run]['training'][:,1], axis=(1,2))
    
    returns_deterministic = np.mean(rewards[run]['evaluation'][:,1], axis=(1,2))
    returns_stochastic = np.mean(rewards[run]['training'][:,1], axis=(1,2))
    
    ax.plot(epochs[run]['training'], returns_stochastic, 
            label='stochastic', color=colors[run], linewidth=0.5)
    
    # ax.plot(all_epochs['evaluation'], returns_deterministic, 
    #         label='deterministic', color=palette(0), linewidth=0.5)

plt.tight_layout()

if SAVE_REWARD_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\fig2_qec_circuit_and_rl_training'
    savename = 'learning_progress'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')






def plot_action(action, axis, color):
    action_max = np.max(action, axis=1)
    action_min = np.min(action, axis=1)
    axis.fill_between(epochs[run]['training'], action_min, action_max, alpha=0.3, color=color)
    axis.plot(epochs[run]['training'], np.mean(action, axis=1), linestyle='-', color=color)

### PLOT ACTIONS
palette = plt.get_cmap('tab10')

fig, axes = plt.subplots(2, 2, figsize = (3.8, 2.6), dpi=600, sharex=True)

for run in [0]: 
    ax = axes[0,1]
    ax.set_ylabel(r'${\rm Re}[\beta_B]$')
    ax.set_ylim(2.49, 2.65)
    ax.set_yticks([2.5,2.6])
    
    beta_real = betas[run]['training'].squeeze()[:,:,1,0]
    plot_action(beta_real, ax, palette(2))
    
    ax.plot(epochs[run]['training'], np.ones_like(epochs[run]['training'])*np.sqrt(2*np.pi), 
            linestyle=':',  color='k')
    
    ax = axes[0,0]
    ax.set_ylabel(r'${\rm Im}[\beta_{S1}]$, ${\rm Im}[\beta_{S2}]$')
    
    eps1_imag = betas[run]['training'].squeeze()[:,:,0,1]
    plot_action(eps1_imag, ax, palette(0))

    eps2_imag = betas[run]['training'].squeeze()[:,:,2,1]
    plot_action(eps2_imag, ax, palette(3))

    ax = axes[1,0]
    ax.set_ylabel(r'$|\alpha_{S1}|$, $|\alpha_{B}|$, $|\alpha_{S3}|$')    

    alpha = alphas[run]['training'].squeeze()[:,:,0]
    plot_action(alpha, ax, palette(0))

    alpha = alphas[run]['training'].squeeze()[:,:,1]
    plot_action(alpha, ax, palette(2))

    alpha = alphas[run]['training'].squeeze()[:,:,2]
    plot_action(alpha, ax, palette(3))

    ax = axes[1,1]
    ax.set_ylabel(r'$\Delta \varphi_g$, $\Delta \varphi_e$ (deg)')
    ax.set_xlabel('Epoch')
    ax.set_xticks([0,250,500])
    
    cav_phase = cavity_phases[run]['training'].squeeze()[:,:,0]/np.pi*180
    cav_phase = cav_phase - cav_phase[0,:]
    plot_action(cav_phase, ax, palette(4))

    cav_phase = cavity_phases[run]['training'].squeeze()[:,:,1]/np.pi*180
    cav_phase = cav_phase - cav_phase[0,:]
    plot_action(cav_phase, ax, palette(1))

plt.tight_layout()

# ### PLOT ALL TRAININGS
# for run in [1,2,3]:
#     line = '-' if run == 0 else '--'

#     ax = axes[0,1]
#     beta = betas[run]['evaluation'].squeeze()[:,1,0]
#     ax.plot(epochs[run]['evaluation'], beta.real, linestyle=line,  color=palette(2))
    
#     ax = axes[0,0]
#     eps1 = betas[run]['evaluation'].squeeze()[:,0,1]
#     ax.plot(epochs[run]['evaluation'], eps1, linestyle=line, color=palette(0))
#     eps2 = betas[run]['evaluation'].squeeze()[:,2,1]
#     ax.plot(epochs[run]['evaluation'], eps2, linestyle=line, color=palette(3))
    
#     ax = axes[1,0]
#     ax.plot(epochs[run]['evaluation'], alphas[run]['evaluation'].squeeze()[:,0], 
#             linestyle=line,  color=palette(0))
#     ax.plot(epochs[run]['evaluation'], alphas[run]['evaluation'].squeeze()[:,1], 
#             linestyle=line,  color=palette(2))
#     ax.plot(epochs[run]['evaluation'], alphas[run]['evaluation'].squeeze()[:,2], 
#             linestyle=line,  color=palette(3))
    
#     ax = axes[1,1]
#     init_phases = cavity_phases[run]['evaluation'].squeeze()[0,:]
#     ax.plot(epochs[run]['evaluation'], 
#             (cavity_phases[run]['evaluation'].squeeze()[:,0]-init_phases[0])*180/np.pi,
#             linestyle=line,  color=palette(4))
#     ax.plot(epochs[run]['evaluation'], 
#             (cavity_phases[run]['evaluation'].squeeze()[:,1]-init_phases[1])*180/np.pi,
#             linestyle=line,  color=palette(1))

# plt.tight_layout()


if SAVE_ACTION_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\reinforcement_learning'
    savename = 'action_evolution'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')