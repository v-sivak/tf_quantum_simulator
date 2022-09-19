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


data_dir = r'E:\data\paper_data\measurement_outcomes'

SAVE_RAIN_OF_BLOOD = False
SAVE_RAIN_OF_BLOOD_SMALL = False
SAVE_AVG_OUTCOME_FIGURE = False
SAVE_STATISTICS_FIGURE = False

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
### Plot all measurement outcomes (g,e,f) aka "RAIN OF BLOOD"
fig, ax = plt.subplots(1,1, dpi=600, figsize=(7, 4))

# multiply by coefficients to set the color
data = dict(np.load(os.path.join(data_dir, 'sample_for_demo.npz')))
classified_data = (data['g']*0.65 + data['e']*0.0 + data['f']*-1.0)

ax.pcolormesh(np.transpose(classified_data), cmap='RdYlGn', 
              vmin=-1, vmax=1, rasterized=True)
ax.set_ylabel('Time (cycles)')
ax.set_xlabel('Experimental shot')
plt.tight_layout()

if SAVE_RAIN_OF_BLOOD:
    savename = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\measurements_statistics\rain_of_blood'
    fig.savefig(savename, fmt='pdf')
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------



# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
### Plot probability of each measurement outcome as a function of time

data0 = dict(np.load(os.path.join(data_dir, 'sample_for_analysis.npz')))
data1 = dict(np.load(os.path.join(data_dir, 'sample_for_analysis_after_1e5_cycles.npz')))


fig, ax = plt.subplots(1,1, dpi=600, figsize=(2.1, 2))
ax.set_xlabel('Time (cycles)')
ax.set_ylabel('Probability')
ax.set_yscale('log')

colors = {'g' : plt.get_cmap('Dark2')(0),
          'f' : plt.get_cmap('RdYlGn')(0.05),
          'e' : plt.get_cmap('Dark2')(5)}

labels = {'g':'g', 'e':'e', 'f':r'$\geq$f'}

for s in ['g','e','f']:
    ax.plot(np.arange(1000), data0[s].mean(axis=0)[:1000], label=labels[s], color=colors[s])

for s in ['g','e','f']:
    ax.plot(1250+np.arange(1000), data1[s].mean(axis=0), color=colors[s])

ax.set_xticks([0, 1000, 1250, 2250])
ax.set_xticklabels([r'$0$', r'$10^3$     ', r'     $10^5$', r'$10^5$+$10^3$'])
plt.tight_layout()

Q = 100
print('Average prob of g: %.4f +- %.4f' %(np.mean(data0['g'].mean(axis=0)[Q:]), 
                                           np.std(data0['g'].mean(axis=0)[Q:])))
print('Average prob of e: %.4f +- %.4f' %(np.mean(data0['e'].mean(axis=0)[Q:]), 
                                           np.std(data0['e'].mean(axis=0)[Q:])))
print('Average prob of f: %.4f +- %.4f' %(np.mean(data0['f'].mean(axis=0)[Q:]), 
                                           np.std(data0['f'].mean(axis=0)[Q:])))


if SAVE_AVG_OUTCOME_FIGURE:
    savename = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\fig4_characterization\average_outcome'
    fig.savefig(savename, fmt='pdf')
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------







# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
### Plot histogram of lengths of strings of identical outcomes
data = dict(np.load(os.path.join(data_dir, 'sample_for_analysis.npz')))
fig, axes = plt.subplots(1,3, dpi=600, figsize=(7, 2.5), 
                         gridspec_kw={'width_ratios': [2,1,1]})


lengths, popt, pcov, durations = {}, {}, {}, {}

for s in ['g', 'f', 'e']:
    T = data[s].shape[1]
    durations[s] = []
    for i in range(data[s].shape[0]): # loop over trajectories
        traj = data[s][i]
        l = 0 # duration of the current "s" event
        flag = False # flag if we are currently in the "s" state
        for t in range(T): # loop over time steps in the trajectory
            if flag:
                if traj[t] == 1:
                    l += 1
                else:
                    flag = False
                    durations[s].append(l)
                    l = 0
            else:
                if traj[t] == 1:
                    flag = True
                    l += 1
    
    # Fit the histogram to exponential distribution. 
    def linear_fn(t, tau, A):
        return A - t/tau
    
    lengths_ = np.arange(1, np.max(durations[s])+1, 1)
    length_counts =  np.zeros_like(lengths_)
    for i, l in enumerate(lengths_):
        length_counts[i] = np.where(np.array(durations[s])==l, 1, 0).sum()
    
    ind = np.where(length_counts>0)[0]
    lengths[s], length_counts = lengths_[ind], length_counts[ind]
    
    p0 = (5, np.log(length_counts[1]))
    pts = {'g':(3,-20), 'e':(1,-1), 'f':(2,-40)}
    popt[s], pcov[s] = curve_fit(linear_fn, lengths[s][pts[s][0]:pts[s][1]], 
                                 np.log(length_counts[pts[s][0]:pts[s][1]]), p0=p0)
    
# Plot this stuff
ax = axes[0]
ax.set_ylim(1, 1e7)
ax.set_xlim(-5,150)
ax.set_yscale('log')
ax.set_xlabel('String duration (cycles)')
ax.set_ylabel('Counts ')
palette = plt.get_cmap('RdYlGn')
colors = {'g':palette((1+0.65)/2), 'e':palette((1-0.25)/2), 'f':palette((1-1.0)/2)}
colors_line = {'g':plt.get_cmap('RdYlGn')(0.99), 
               'e':plt.get_cmap('tab10')(1),  
               'f':plt.get_cmap('tab10')(3)}
labels = {'g':'g', 'e':'e', 'f':r'$\geq$f'}
for s in ['g', 'f', 'e']:
    alpha = 0.75 if s=='e' else 1.0
    bins=np.arange(1, np.max(durations[s])+2, 1) - 0.5
    ax.hist(durations[s], bins=bins, align='mid', alpha=alpha,
            color=colors[s])
    ax.plot(lengths[s], np.exp(linear_fn(lengths[s], *popt[s])), color=colors_line[s],
            label=labels[s]+r': $\tau=$%.1f cycles'%popt[s][0])
ax.legend(title='Exponential fit')
plt.tight_layout()
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------





# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
### Plot fraction of shots that experienced an error string of given duration
def exp_saturation(t, tau):
    return 1-np.exp(-t/tau)

def filter_error_strings(a, d):
    E = a
    for i in range(1, d):
        E = np.logical_and(E, np.roll(a, -i, axis=1) == E)
    return np.where(E, 1, 0)


data = dict(np.load(os.path.join(data_dir, 'sample_for_analysis.npz')))

D = [1,2,3,4,5,6,7,8]

B, T = data['e'].shape
mask_e = {d : np.zeros(T) for d in D}
for d in D:
    filtered = filter_error_strings(data['e'], d)    
    for t in range(T):
        mask_e[d][t] = np.any(filtered[:,:t+1], axis=1).astype(int).mean()

mask_f = {d : np.zeros(T) for d in D}
for d in D:
    filtered = filter_error_strings(data['f'], d)    
    for t in range(T):
        mask_f[d][t] = np.any(filtered[:,:t+1], axis=1).astype(int).mean()


axes[1].grid(True)
axes[2].grid(True)
axes[1].set_yscale('log')
axes[1].set_xscale('log')
axes[2].set_yscale('log')
axes[2].set_xscale('log')
axes[1].set_ylim(1e-3, 1.1)
axes[2].set_ylim(1e-3, 1.1)

axes[1].set_ylabel('Fraction of shots')
axes[1].set_xlabel('Time (cycles)')
axes[2].set_xlabel('Time (cycles)')
axes[1].set_title('Errors')
axes[2].set_title('Leakage')

error_rate, leakage_rate = [], []

for d in D:
    axes[1].plot(np.arange(d,T+d,1), mask_e[d], label=d, color=plt.get_cmap('tab10')(1))
    axes[2].plot(np.arange(d,T+d,1), mask_f[d], label=d, color=plt.get_cmap('tab10')(3))
    
    popt, pcov = curve_fit(exp_saturation, np.arange(d,T+d,1), mask_e[d], p0=(1000))
    print('Rate of d=%d error events' %d)
    print('tau=%.f +- %.f' %(popt[0], np.sqrt(np.diag(pcov))[0]))
    error_rate.append(1/popt[0])
    axes[1].plot(np.arange(d,T+d,1), exp_saturation(np.arange(d,T+d,1), *popt), 
            color='k', linestyle=':')

    popt, pcov = curve_fit(exp_saturation, np.arange(d,T+d,1), mask_f[d], p0=(1000))
    print('Rate of d=%d leakage events' %d)
    print('tau=%.f +- %.f' %(popt[0], np.sqrt(np.diag(pcov))[0]))
    leakage_rate.append(1/popt[0])
    axes[2].plot(np.arange(d,T+d,1), exp_saturation(np.arange(d,T+d,1), *popt), 
            color='k', linestyle=':')

axes[1].set_xticks([1,1e1,1e2,1e3])
axes[2].set_xticks([1,1e1,1e2,1e3])

plt.tight_layout()


if SAVE_STATISTICS_FIGURE:
    savename = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\measurements_statistics\fig_stat'
    fig.savefig(savename, fmt='pdf')




# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
### Plot all measurement outcomes, aka rain of blood (small main text version)
data = dict(np.load(os.path.join(data_dir, 'sample_for_demo_small.npz')))
fig, ax = plt.subplots(1,1, dpi=600, figsize=(1.3, 2))

# multiply by coefficients to set the color
classified_data = data['g']*0.65 + data['e']*-0.2 + data['f']*-1.0

ax.pcolormesh(np.transpose(classified_data)[100:190,:], cmap='RdYlGn', 
              vmin=-1, vmax=1, rasterized=True)
ax.set_ylabel('Time (cycles)')
ax.set_xlabel('Shot')
plt.tight_layout()

if SAVE_RAIN_OF_BLOOD_SMALL:
    savename = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\fig4_characterization\msmt_sample'
    fig.savefig(savename, fmt='pdf')










# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
### Plot conditional probabilities of measurement outcomes
if 0:
    cond_probs = np.zeros([3,3])
    for i, s1 in zip([0,1,2], ['g','e','f']):
        print('First msmt %d'%i)
        for j, s2 in zip([0,1,2], ['g','e','f']):
            M1 = data[s1]
            M2 = data[s2]
            x = np.ma.array(np.roll(M2, -1, axis=1), mask=1-M1)
            print(x.mean())
            cond_probs[i,j] = x.mean()
    
    
    
    fig, ax = plt.subplots(1,1, dpi=600)
    ax.set_yscale('log')
    for j, s in zip([0,1,2], ['g','e','f']):
        x = np.array([1,2,3]) + 0.25*np.array([-1,0,1])[j]
        ax.bar(x, cond_probs[:,j], width=0.25, color=colors[s])
    ax.set_xticks([1,2,3])
    ax.set_xticklabels(['g', 'e', 'f'])
        
    ax.set_ylabel(r'Prob. of outcome in cycle $t$')
    ax.set_xlabel(r'Outcome in cycle $t-1$')
    plt.tight_layout()
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
### Plot histogram of times until the first leakage
if 0:
    T = data['f'].shape[1]
    all_times_until_first_leakage = []
    all_durations_of_first_leakage = []
    for i in range(data['f'].shape[0]): # loop over trajectories
        traj = data['f'][i]
        leakage_locs = np.nonzero(traj)[0]
        if len(leakage_locs)>0:
            all_times_until_first_leakage.append(leakage_locs[0])
            
            l = 0
            if len(leakage_locs)>1:
                while leakage_locs[l+1] - leakage_locs[l] == 1:
                    l = l+1
                    if l == len(leakage_locs)-1:
                        break
            l += 1
            all_durations_of_first_leakage.append(l)
            
    
    # Fit the histogram to exponential distribution
    def linear_fn(t, tau, A):
        return A - t/tau
    
    lengths = np.arange(1, np.max(all_times_until_first_leakage)+1, 1)
    length_counts =  np.zeros_like(lengths)
    for i, l in enumerate(lengths):
        length_counts[i] = np.where(np.array(all_times_until_first_leakage)==l, 1, 0).sum()
    
    ind = np.where(length_counts>0)[0]
    lengths, length_counts = lengths[ind], length_counts[ind]
    
    p0 = (20, np.log(length_counts[1]))
    popt, pcov = curve_fit(linear_fn, lengths[:120], np.log(length_counts)[:120], p0=p0)
    
    
    fig, ax = plt.subplots(1,1, dpi=200)
    ax.set_yscale('log')
    ax.set_xlabel('Time until first leakage (steps)')
    ax.set_ylabel('Counts')
    ax.hist(all_times_until_first_leakage, align='mid',
            bins=np.arange(1, np.max(all_times_until_first_leakage)+2, 1)-0.5)
    ax.plot(lengths, np.exp(linear_fn(lengths, *popt)), 
            label='exp. fit, tau=%.1f'%popt[0])
    ax.legend()
    
    
    fig, ax = plt.subplots(1,1, dpi=200)
    ax.set_yscale('log')
    ax.set_xlabel('Duration of first leakage (steps)')
    ax.set_ylabel('Counts')
    ax.hist(all_durations_of_first_leakage, align='mid',
            bins=np.arange(1, np.max(all_durations_of_first_leakage)+2, 1)-0.5)
    ax.legend()
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------