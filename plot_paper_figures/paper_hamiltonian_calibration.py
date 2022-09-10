# -*- coding: utf-8 -*-
"""
Created on Tue Apr 19 09:14:02 2022

@author: qulab
"""
import os
import plot_config
import numpy as np 
import matplotlib.pyplot as plt
from math import sqrt, pi
from scipy.optimize import curve_fit
import matplotlib as mpl
import matplotlib.gridspec as gridspec

datadir = r'E:\data\paper_data\hamiltonian_calibration'


SAVE_FIGURE = True


### 1st PANEL: Qubit spectroscopy with oscillator in vacuum
data = np.load(os.path.join(datadir, 'qubit_spec_0.npz'))
freq, data = data['freq'], data['data']

# fig, axes = plt.subplots(1,3, figsize=(7,2.5), dpi=600)

fig = plt.figure(1, figsize=(7,2.5), dpi=600)
gridspec.GridSpec(3,2)


ax = plt.subplot2grid((2,3), (0,0), colspan=1, rowspan=2)

ax.set_xlabel(r'Detuning (kHz)')
ax.set_ylabel(r'Prob. of qubit flip (unnormalized)')

color = plt.get_cmap('Paired')(0)
ax.plot(freq, data, linestyle='-', color=color, zorder=10)
color = plt.get_cmap('Paired')(1)
ax.plot(freq, data, marker='.', linestyle='none', color=color, zorder=10, label=r'$\alpha=0$')




### 1st PANEL: Number-resolved spectroscopy with oscillator in coherent state
data = np.load(os.path.join(datadir, 'qubit_spec_1.npz'))
freq, data = data['freq'], data['data']

color = plt.get_cmap('Paired')(4)
ax.plot(freq, data, linestyle='-', color=color, zorder=10)
color = plt.get_cmap('Paired')(5)
ax.plot(freq, data, marker='.', linestyle='none', color=color, zorder=10, label=r'$\alpha\approx0.6$')

ax.legend(title='Coherent state', loc='upper left')



### 2nd PANEL: Out-and-back phase dispersion
ax = plt.subplot2grid((2,3), (0,1), colspan=1, rowspan=2)


ax.set_xlabel(r'Mean photon number, $\overline{n}$')
ax.set_ylabel(r'Rotation frequency (kHz)')


data = np.load(os.path.join(datadir, 'out_and_back_2022_07_28.npz'))

qubit_states = ['g', 'e']
phase = {'e' : data['phase_e'], 'g' : data['phase_g']}
time, nbar = data['time'], data['nbar']

# fit rotation frequency
def linear(t, freq, offset):
    return 360*freq*t*1e-9 + offset

freq_fits = {s:[] for s in qubit_states}
offset_fits = {s:[] for s in qubit_states}
for s in qubit_states:
    for i in range(len(nbar)):
        popt, pcov = curve_fit(linear, time, phase[s][:,i])
        freq_fits[s].append(popt[0])
        offset_fits[s].append(popt[1])
freq_g, freq_e = np.array(freq_fits['g']), np.array(freq_fits['e'])

# Plot phase vs time for each alpha
ax.set_xlabel('Wait time (ns)')
ax.set_ylabel('Optimal return phase (deg)')
cmap = plt.get_cmap('Spectral_r')
norm = mpl.colors.Normalize(vmin = min(nbar), vmax = max(nbar))

for s in qubit_states:
    for i in range(len(nbar)):
        mean_phase = phase[s][:,i]
        color = cmap(norm(nbar[i]))
        ax.plot(time, mean_phase, color=color, linestyle='none', marker='.')
        ax.plot(time, linear(time, freq_fits[s][i], offset_fits[s][i]), 
                linestyle='-', marker=None, color=color)




### 3rd PANEL: Out-and-back average rotation frequency
ax = plt.subplot2grid((2,3), (0,2), colspan=1, rowspan=1)


# Plot average rotation frequency and fit Kerr and detuning
avg_freq = (freq_g + freq_e)/2.0
def avg_freq_fit_func(n, Kerr, Delta):
    return  Delta + Kerr * n
fit_pts = 45
popt, pcov = curve_fit(avg_freq_fit_func, nbar[:fit_pts], avg_freq[:fit_pts])
Kerr, Delta = popt


ax.set_ylabel(r'Average freq. (kHz)')
ax.plot(nbar, avg_freq*1e-3, marker='.',linestyle='none', color=plt.get_cmap('Paired')(5))
ax.plot(nbar[:fit_pts], avg_freq_fit_func(nbar[:fit_pts], *popt)*1e-3,
        # label=r'K=%.0f Hz, $\Delta$= %.1f kHz' %(Kerr,Delta*1e-3))
        label=' ', color='black')





### 4th PANEL: Out-and-back difference rotation frequency
ax = plt.subplot2grid((2,3), (1,2), colspan=1, rowspan=1, sharex=ax)

# Plot difference rotation frequency and fit chi and chi_prime
diff_freq = freq_g - freq_e
def diff_freq_fit_func(n, chi, chi_prime):
    return  chi + chi_prime * n
popt, pcov = curve_fit(diff_freq_fit_func, nbar[2:fit_pts], diff_freq[2:fit_pts])
chi, chi_prime = popt


ax.set_xlabel(r'Average photon number, $\overline{n}$')
ax.set_ylabel(r'Relative freq. (kHz)')
ax.plot(nbar, diff_freq*1e-3, marker='.',linestyle='none', color=plt.get_cmap('Paired')(5))
ax.plot(nbar[:fit_pts], diff_freq_fit_func(nbar[:fit_pts], *popt)*1e-3,
        # label='chi=%.0f kHz, chi_prime= %.0f Hz' %(chi*1e-3,chi_prime))
        label=' ', color='black')

ax.scatter([0], [46.6], marker='*')





plt.tight_layout()

if SAVE_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\hamiltonian_parameters'
    savename = 'hamiltonian'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')
    
    
    




