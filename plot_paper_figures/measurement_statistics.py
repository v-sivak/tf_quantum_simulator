# -*- coding: utf-8 -*-
"""
Created on Sun Oct  2 16:10:48 2022
"""
import os
import plot_config
import numpy as np 
import matplotlib.pyplot as plt
from math import sqrt, pi
from scipy.optimize import curve_fit

data_dir = os.path.join(plot_config.data_root_dir, 'measurement_outcomes')
SAVE_FIGURE = False


### LOAD DATA
data = dict(np.load(os.path.join(data_dir, 'sample_for_analysis.npz')))

mask = np.any(data['f'], axis=1)
strings = data['g'][np.where(mask==False)[0]] # g=1, e=0

even = np.arange(0, strings.shape[1], 2) # even indices
odd = np.arange(1, strings.shape[1], 2) # odd indices

# split into sub-strings of length 2 and make selective masks:
g1 = np.where(strings[:,even]==1, 1, 0) # 1 where 'g?'
g2 = np.where(strings[:,odd]==1, 1, 0) # 1 where '?g'
e1 = np.where(strings[:,even]==0, 1, 0) # 1 where 'e?'
e2 = np.where(strings[:,odd]==0, 1, 0) # 1 where '?e'

pdata = g1*g2 * 0 + g1*e2 * 1 + e1*g2 * 2 + e1*e2 * 3



### PLOT PROBABILITY OF A STRING gg/gg/...
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
fig, axes = plt.subplots(1,3, dpi=600, figsize=(7, 1.5))

t_slide = 40
T = pdata.shape[1]
t0 = 0
times = np.arange(1,t_slide+1,1)
probs = []

for i in range((T-t0)//t_slide):
    p = np.zeros(t_slide)
    for t in range(t_slide):
        p[t] = 1-np.any(pdata[:,t0+i*t_slide:t0+i*t_slide+t+1], axis=1).mean()
    probs.append(p)

probs = np.mean(probs, axis=0)

def linear(t, lambda0, A0):
    return np.log(A0) + t * np.log(lambda0)

popt, pcov = curve_fit(linear, times, np.log(probs))

ax = axes[0]
ax.set_xlabel(r'String length, $n$ (QEC cycles)')
ax.set_ylabel('Prob. of string'+'\n'+r'$gg$ / $gg$ / ...')
ax.set_yscale('log')
ax.plot(times, probs, color='#2a9d8f', marker='.')
ax.plot(times, np.exp(linear(times, *popt)), linestyle='-', color='k')


lambda_0, a0 = popt[0], popt[1]
print('rate %.3f' %lambda_0)
print('intercept %.3f' %a0)
print('projector %.3f' %(a0*lambda_0))



### PLOT HISTOGRAM OF LEAKAGE DURATIONS
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------

lengths, popt, pcov, durations = {}, {}, {}, {}

for s in ['f']:
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
    def linear(t, tau, A):
        return A - t/tau
    
    lengths_ = np.arange(1, np.max(durations[s])+1, 1)
    length_counts =  np.zeros_like(lengths_)
    for i, l in enumerate(lengths_):
        length_counts[i] = np.where(np.array(durations[s])==l, 1, 0).sum()
    
    ind = np.where(length_counts>0)[0]
    lengths[s], length_counts = lengths_[ind], length_counts[ind]
    
    p0 = (5, np.log(length_counts[1]))
    pts = {'g':(3,-20), 'e':(1,-1), 'f':(2,-40)}
    popt[s], pcov[s] = curve_fit(linear, lengths[s][pts[s][0]:pts[s][1]], 
                                 np.log(length_counts[pts[s][0]:pts[s][1]]), p0=p0)
    
# Plot this stuff
ax = axes[1]
ax.set_ylim(1, 1e5)
ax.set_xlim(-5,120)
ax.set_yscale('log')
ax.set_xlabel('Leakage duration (cycles)')
ax.set_ylabel('Counts ')
palette = plt.get_cmap('RdYlGn')
colors = {'g':palette((1+0.65)/2), 
          'e':palette((1-0.25)/2), 
          'f':'#8ecae6'}#palette((1-1.0)/2)}
colors_line = {'g':plt.get_cmap('RdYlGn')(0.99), 
               'e':plt.get_cmap('tab10')(1),  
               'f':plt.get_cmap('tab10')(3)}
labels = {'g':'g', 'e':'e', 'f':r'$\geq$f'}
for s in ['f']:
    alpha = 0.75 if s=='e' else 1.0
    bins=np.arange(1, np.max(durations[s])+2, 1) - 0.5
    ax.hist(durations[s], bins=bins, align='mid', alpha=alpha,
            color=colors[s])
    ax.plot(lengths[s], np.exp(linear(lengths[s], *popt[s])), color='k',
            label=r'$\tau=$%.1f cycles'%popt[s][0])
ax.legend(title='Exponential fit')
plt.tight_layout()
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------


### PLOT PROBABILITY OF LEAKAGE EVENTS 
# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
def exp_saturation(t, tau):
    return 1-np.exp(-t/tau)

def filter_strings(a, d):
    E = a
    for i in range(1, d):
        E = np.logical_and(E, np.roll(a, -i, axis=1) == E)
    return np.where(E, 1, 0)


data = dict(np.load(os.path.join(data_dir, 'sample_for_analysis.npz')))

D = [1,2,3,4,5]

mask_f = {d : np.zeros(T) for d in D}
for d in D:
    filtered = filter_strings(data['f'], d)    
    for t in range(T):
        mask_f[d][t] = np.any(filtered[:,:t+1], axis=1).astype(int).mean()



ax = axes[2]
# ax.grid(True)
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_ylim(1e-3, 1.1)
ax.set_xlabel('Time (cycles)')
ax.set_ylabel('Prob. of leakage'+'\n'+ r'of length $d$ cycles')

error_rate, leakage_rate = [], []

for d in D:
    ax.plot(np.arange(d,T+d,1), mask_f[d], label=d, color=plt.get_cmap('tab10')(3))
    popt, pcov = curve_fit(exp_saturation, np.arange(d,T+d,1), mask_f[d], p0=(1000))
    print('Rate of d=%d leakage events' %d)
    print('tau=%.f +- %.f' %(popt[0], np.sqrt(np.diag(pcov))[0]))
    leakage_rate.append(1/popt[0])
    ax.plot(np.arange(d,T+d,1), exp_saturation(np.arange(d,T+d,1), *popt), 
            color='k', linestyle=':')

ax.set_xticks([1,1e1,1e2,1e3])

plt.tight_layout()
savename = os.path.join(plot_config.save_root_dir, 
                        r'measurements_statistics\fig_stat_v2.pdf')
if SAVE_FIGURE: fig.savefig(savename)

