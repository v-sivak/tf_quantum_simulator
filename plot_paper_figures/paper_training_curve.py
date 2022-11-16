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

def plot_action(action, axis, color, run):
    action_max = np.max(action, axis=1)
    action_min = np.min(action, axis=1)
    axis.fill_between(epochs[run]['training'], action_min, action_max, alpha=0.3, color=color)
    axis.plot(epochs[run]['training'], np.mean(action, axis=1), linestyle='-', color=color)

all_runs = {
    0 : r'E:\data\paper_data\rl_training\run_165_24.06.2022-17.48.55_paper_example',
    1 : r'E:\data\paper_data\rl_training\run_185_12.09.2022-14.58.06_example_1',
    2 : r'E:\data\paper_data\rl_training\run_186_12.09.2022-18.36.42_example_2',
    3 : r'E:\data\paper_data\rl_training\run_189_13.09.2022-14.44.50_example_3',
    4 : r'E:\data\paper_data\rl_training\run_190_13.09.2022-19.11.47_example_4',
    5 : r'E:\data\paper_data\rl_training\run_191_14.09.2022-08.43.13_example_5'
    }

SAVE_REWARD_FIGURE = False
SAVE_ACTION_FIGURE = False
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
                detunings[run][epoch_type].append(data['action_batch'][()]['ef_detune'])
                alphas[run][epoch_type].append(data['action_batch'][()]['alpha'])

    epochs[run] = {k : np.array(v) for k,v in epochs[run].items()}
    rewards[run] = {k : np.array(v) for k,v in rewards[run].items()}
    cavity_phases[run] = {k : np.array(v) for k,v in cavity_phases[run].items()}
    betas[run] = {k : np.array(v) for k,v in betas[run].items()}
    alphas[run] = {k : np.array(v) for k,v in alphas[run].items()}
    detunings[run] = {k : np.array(v) for k,v in detunings[run].items()}
    epochs[run]['evaluation'] *= evaluation_interval



### PLOT REWARDS
fig, axes = plt.subplots(1, 2, figsize = (3.375, 1.25), dpi=600,
                       gridspec_kw={'width_ratios':[1.2,1]})

palette = plt.get_cmap('tab10')

ax = axes[0]
ax.set_ylabel(r'Reward, $R$')
ax.set_xlabel('Epoch')
ax.set_xticks([0,250,500])
ax.set_yticks([0,0.2,0.4])
ax.set_ylim(0, 0.5)

ax = axes[1]
ax.set_xlabel('Epoch')
ax.set_ylabel('Intermediate\nphoton number')
ax.set_xticks([0,250,500])

run = 0

ind = np.arange(0, np.max(epochs[run]['training']), 4) # to avoid plotting every epoch
returns_stochastic = np.mean(rewards[run]['training'][:,1], axis=(1,2))

returns_deterministic = np.mean(rewards[run]['evaluation'][:,1], axis=(1,2))
returns_stochastic = np.mean(rewards[run]['training'][:,1], axis=(1,2))

ax = axes[0]
ax.plot(epochs[run]['training'], returns_stochastic, 
        label='stochastic', color=palette(3), linewidth=0.5)

# ax.plot(epochs[run]['evaluation'], returns_deterministic, 
#         label='stochastic', color=palette(3), linewidth=0.5)

# ax.plot(epochs[run]['evaluation'], np.ones(29)*returns_deterministic[0], 
#         label='stochastic', color=palette(3), linewidth=0.5)  

ax = axes[1]
alpha = alphas[run]['training'].squeeze()[:,:,1]
plot_action(alpha**2, ax, palette(0), run)

plt.tight_layout()

if SAVE_REWARD_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures_working\fig2_qec_circuit_and_rl_training'
    savename = 'learning_progress'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')






### PLOT ACTIONS
palette = plt.get_cmap('tab10')


fig, axes = plt.subplots(2, 5, figsize = (7, 2.6), dpi=600, sharex=True,
                         sharey='col')

for run in [2]: 
    ax = axes[0,0]
    ax.set_title(r'${\rm Re}[\beta_B]$')
    # ax.set_ylim(2.49, 2.65)
    # ax.set_yticks([2.5,2.6])
    
    beta_real = betas[run]['training'].squeeze()[:,:,1,0]
    plot_action(beta_real, ax, palette(2), run)
    
    ax.plot(epochs[run]['training'], np.ones_like(epochs[run]['training'])*np.sqrt(2*np.pi), 
            linestyle=':',  color='k')
    
    ax = axes[0,1]
    ax.set_title(r'${\rm Im}[\beta_{S1}]$, ${\rm Im}[\beta_{S2}]$')
    
    eps1_imag = betas[run]['training'].squeeze()[:,:,0,1]
    plot_action(eps1_imag, ax, palette(0), run)

    eps2_imag = betas[run]['training'].squeeze()[:,:,2,1]
    plot_action(eps2_imag, ax, palette(3), run)

    ax = axes[0,2]
    ax.set_title(r'$|\alpha_{S1}|$, $|\alpha_{B}|$, $|\alpha_{S2}|$')

    alpha = alphas[run]['training'].squeeze()[:,:,0]
    plot_action(alpha, ax, palette(0), run)

    alpha = alphas[run]['training'].squeeze()[:,:,1]
    plot_action(alpha, ax, palette(2), run)

    alpha = alphas[run]['training'].squeeze()[:,:,2]
    plot_action(alpha, ax, palette(3), run)

    ax = axes[0,3]
    ax.set_title(r'$\Delta \vartheta_g$, $\Delta \vartheta_e$ (deg)')
    ax.set_ylim(-6,6)
    
    cav_phase = cavity_phases[run]['training'].squeeze()[:,:,0]/np.pi*180
    cav_phase = cav_phase - cav_phase[0,:]
    plot_action(cav_phase, ax, palette(4), run)

    cav_phase = cavity_phases[run]['training'].squeeze()[:,:,1]/np.pi*180
    cav_phase = cav_phase - cav_phase[0,:]
    plot_action(cav_phase, ax, palette(1), run)

    ax = axes[0,4]
    ax.set_title(r'e$-$f detuning (MHz)')
    
    delta = detunings[run]['training'].squeeze()*1e-6
    plot_action(delta, ax, palette(5), run)

plt.tight_layout()

### PLOT ALL TRAININGS

for run in [1,2,3,4,5]:
    line = '-' if run == 0 else '--'

    ax = axes[1,0]
    beta = betas[run]['evaluation'].squeeze()[:,1,0]
    ax.plot(epochs[run]['evaluation'], beta.real, linestyle=line,  color=palette(2))
    
    ax = axes[1,1]
    eps1 = betas[run]['evaluation'].squeeze()[:,0,1]
    ax.plot(epochs[run]['evaluation'], eps1, linestyle=line, color=palette(0))
    eps2 = betas[run]['evaluation'].squeeze()[:,2,1]
    ax.plot(epochs[run]['evaluation'], eps2, linestyle=line, color=palette(3))
    
    ax = axes[1,2]
    ax.plot(epochs[run]['evaluation'], alphas[run]['evaluation'].squeeze()[:,0], 
            linestyle=line,  color=palette(0))
    ax.plot(epochs[run]['evaluation'], alphas[run]['evaluation'].squeeze()[:,1], 
            linestyle=line,  color=palette(2))
    ax.plot(epochs[run]['evaluation'], alphas[run]['evaluation'].squeeze()[:,2], 
            linestyle=line,  color=palette(3))
    
    ax = axes[1,3]
    init_phases = cavity_phases[run]['evaluation'].squeeze()[0,:]
    ax.plot(epochs[run]['evaluation'], 
            (cavity_phases[run]['evaluation'].squeeze()[:,0]-init_phases[0])*180/np.pi,
            linestyle=line,  color=palette(4))
    ax.plot(epochs[run]['evaluation'], 
            (cavity_phases[run]['evaluation'].squeeze()[:,1]-init_phases[1])*180/np.pi,
            linestyle=line,  color=palette(1))

    ax = axes[1,4]
    delta = detunings[run]['evaluation'].squeeze()*1e-6
    ax.plot(epochs[run]['evaluation'], delta, linestyle=line,  color=palette(5))

axes[1,2].set_xlabel('Epoch')
axes[1,0].set_xticks([0,400,800])

plt.tight_layout()



if SAVE_ACTION_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\reinforcement_learning'
    savename = 'action_evolution'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')