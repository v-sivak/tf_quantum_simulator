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

def exp_decay(t, A, tau, ofs):
    return A * np.exp(-t/tau) + ofs

def linear(t, tau):
    return t/tau



### LOAD DATA
data = dict(np.load(r'Z:\shared\tmp\for Vlad\from_vlad\measurement_statistics\measurements_2.npz'))

SAVE_RAIN_OF_BLOOD = False
SAVE_STATISTICS_FIGURE = False
SAVE_CORRELATION_FIGURE = True

# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
### Plot all measurement outcomes (g,e,f) aka "RAIN OF BLOOD"
fig, ax = plt.subplots(1,1, dpi=600, figsize=(7, 4))

# multiply by coefficients to set the color
# classified_data = (data['g']*0.6 + data['e']*0.1 + data['f']*-1.0)[0:800]
classified_data = (data['g']*0.7 + data['e']*0.0 + data['f']*-1.0)[100:700]

ax.pcolormesh(np.transpose(classified_data), cmap='RdYlGn', 
              vmin=-1, vmax=1, rasterized=True)
ax.set_ylabel('QEC cycle')
ax.set_xlabel('Experimental shot')
plt.tight_layout()

if SAVE_RAIN_OF_BLOOD:
    savename = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\measurements_statistics\rain_of_blood'
    fig.savefig(savename, fmt='pdf')
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------




# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
### Analyse correlation between measurements 

fig, ax = plt.subplots(1,1, dpi=600, figsize=(1.5,1.5))
r = np.corrcoef(data['g'].transpose())
ax.set_aspect('equal')
p = ax.pcolormesh(r[:-1,:-1], cmap='seismic', vmin=-0.1, vmax=0.1)
# plt.colorbar(p)
# ax.axis('off')
# plt.tight_layout()
if SAVE_CORRELATION_FIGURE:
    savename = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\measurements_statistics\correlation'
    fig.savefig(savename, fmt='pdf')


# assume stationary process and averge the correlator; exclue first 8 rounds
# when the code corrects errors in state preparation
r = np.corrcoef(data['g'].transpose())
fig, ax = plt.subplots(1,1, dpi=600)
r_avg = np.mean([r[i,i:i+1000] for i in range(r.shape[0]-1000)], axis=0)
# ax.plot(np.arange(1, len(r_avg), 1), r_avg[1:], marker='.')
ax.plot(np.zeros(800))
ax.set_xlabel('Separation (cycles)')
ax.set_ylabel('Correlation coefficient')

# ax.plot(np.arange(800), np.flip(r[0:800,800]), marker='.')
# ax.plot(np.arange(200), np.flip(r[0:200,200]), marker='.')
# ax.plot(np.arange(40), np.flip(r[0:40,40]), marker='.')
ax.set_xscale('log')


r_avg = np.mean([r[i-800:i,i] for i in range(800,820)], axis=0)
ax.plot(1+np.arange(800), np.flip(r_avg), marker='.')

r_avg = np.mean([r[i-200:i,i] for i in range(200,220)], axis=0)
ax.plot(1+np.arange(200), np.flip(r_avg), marker='.')

r_avg = np.mean([r[i-40:i,i] for i in range(40,60)], axis=0)
ax.plot(1+np.arange(40), np.flip(r_avg), marker='.')
# ax.plot(np.arange(1, len(r_avg), 1), r_avg[1:], marker='.')


def exp(t, T,a,b):
    return a*np.exp(-t/T)+b
from scipy.optimize import curve_fit
popt, pcov = curve_fit(exp, np.arange(30, len(r_avg), 1), r_avg[30:], p0=(300,0.02,0))
print(popt[0])

ax.plot(np.arange(1, len(r_avg), 1), exp(np.arange(1, len(r_avg), 1), *popt))



Pi = (data['g']*np.roll(data['g'],1,axis=1)).mean(axis=0)
Pi_prod = data['g'].mean(axis=0)**2
print('Expectation of code projector: %.3f +- %.3f' %(Pi.mean(), Pi.std()))
print('Projector from product: %.3f +- %.3f' %(Pi_prod.mean(), Pi_prod.std()))




# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
### (0)
### Plot probability of each measurement outcome as a function of time
fig, axes = plt.subplots(1,3, dpi=200, figsize=(7, 2))
ax = axes[0]
ax.set_xlabel('QEC cycle')
ax.set_ylabel('Probability')
ax.set_yscale('log')

colors = {'g' : plt.get_cmap('Dark2')(0),
          'f' : plt.get_cmap('Set1')(0),
          'e' : plt.get_cmap('Dark2')(5)}

for s in ['g','e','f']:
    ax.plot(data[s].mean(axis=0), label=s, color=colors[s])
ax.legend(ncol=3)
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------



# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
### Plot histogram of durations of leakage events (includes true leakage + outliers)
T = data['f'].shape[1]
all_durations_of_leakage = []
for i in range(data['f'].shape[0]): # loop over trajectories
    traj = data['f'][i]
    l = 0 # duration of the current leakage event
    flag = False # flag if we are currently in the leakage state
    for t in range(T): # loop over time steps in the trajectory
        if flag:
            if traj[t] == 1:
                l += 1
            else:
                flag = False
                all_durations_of_leakage.append(l)
                l = 0
        else:
            if traj[t] == 1:
                flag = True
                l += 1

# Fit the histogram to exponential distribution. Ignore length=1 events, 
# because some unknown part of them are outliers.
def linear_fn(t, tau, A):
    return A - t/tau

lengths = np.arange(1, np.max(all_durations_of_leakage)+1, 1)
length_counts =  np.zeros_like(lengths)
for i, l in enumerate(lengths):
    length_counts[i] = np.where(np.array(all_durations_of_leakage)==l, 1, 0).sum()

ind = np.where(length_counts>0)[0]
lengths, length_counts = lengths[ind], length_counts[ind]

p0 = (10, np.log(length_counts[1]))
popt, pcov = curve_fit(linear_fn, lengths[2:40], np.log(length_counts[2:40]), p0=p0)



### (1)
ax = axes[1]
ax.set_ylim(1, 3e4)
ax.set_xlim(-5,100)
ax.set_yscale('log')
ax.set_xlabel('Leakage duration (cycles)')
ax.set_ylabel('Counts ')
bins=np.arange(1, np.max(all_durations_of_leakage)+2, 1)-0.5
ax.hist(all_durations_of_leakage, bins=bins, align='mid', 
        color=plt.get_cmap('Set3')(4))
ax.plot(lengths, np.exp(linear_fn(lengths, *popt)), color='k',
        label=r'exponential fit'+'\n'+r'$\tau=$%.1f cycles'%popt[0])
ax.legend()
plt.tight_layout()
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------




# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
def filter_length_2_leakage_events(a):
    """
    Filter of leakage events of length=2. Will create a mask == 1 if the
    trajectory contains length=2 leakage event.
    """
    cond1 = np.where(a==1, True, False)
    cond2 = np.where(np.roll(a,1,axis=1)==1, True, False)
    cond3 = np.where(np.roll(a,-1,axis=1)==1, True, False)
    
    cond = cond1 * cond2 + cond1 * cond3
    return cond.astype(int)

### Plot fraction of trajectories that experienced leakage
filtered = filter_length_2_leakage_events(data['f'])
B, T = data['f'].shape
mask_postselected, mask_filtered_v1 = np.zeros(T), np.zeros(T)
for t in range(T):
    mask_postselected[t] = np.any(data['f'][:,:t+1], axis=1).astype(int).mean()
    mask_filtered_v1[t] = np.any(filtered[:,:t+1], axis=1).astype(int).mean()



### (2)
# fig, ax = plt.subplots(1,1,dpi=600)
ax = axes[2]
ax.grid(True)
ax.set_xlabel('QEC cycle')
ax.set_ylabel('Fraction of shots with leakage')
ax.set_yscale('log')
ax.set_xscale('log')

def log_fit(t, a, tau):
    return a * (np.log(t) - np.log(tau))

popt, pcov = curve_fit(log_fit, np.arange(1,T+1,1), np.log(mask_postselected), 
                       p0=(1,2000))
print('Rate of all leakage events, f=(t/tau)^a')
print('tau=%.f +- %.f' %(popt[1], np.sqrt(np.diag(pcov))[1]))
print('a=%.4f +- %.4f' %(popt[0], np.sqrt(np.diag(pcov))[0]))
ax.plot(np.arange(1,T+1,1), np.exp(log_fit(np.arange(1,T+1,1), *popt)), 
        color='k', linestyle=':')
ax.plot(np.arange(1,T+1,1), mask_postselected, label='any leakage event')


popt, pcov = curve_fit(log_fit, np.arange(1,T+1,1), np.log(mask_filtered_v1), 
                       p0=(1,2000))
print('Rate of length>1 events, f=(t/tau)^a')
print('tau=%.f +- %.f' %(popt[1], np.sqrt(np.diag(pcov))[1]))
print('a=%.4f +- %.4f' %(popt[0], np.sqrt(np.diag(pcov))[0]))
ax.plot(np.arange(1,T+1,1), np.exp(log_fit(np.arange(1,T+1,1), *popt)), 
        color='k', linestyle=':')
ax.plot(np.arange(1,T+1,1), mask_filtered_v1, label=r'length$\geq$2 events')

ax.legend()
plt.tight_layout()
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

if SAVE_STATISTICS_FIGURE:
    savename = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\measurements_statistics\fig'
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