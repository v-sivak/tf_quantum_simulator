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

SAVE_FIGURE = False

data = np.load(r'Z:\shared\tmp\for Vlad\from_vlad\Hamiltonian_calibration\qubit_spec.npz')

freq = data['freq']
data = data['data']

def func(f, f0, chi, sigma, ofs, a0, a1, a2, a3):
    f1 = f0 - 1*chi
    f2 = f0 - 2*chi
    f3 = f0 - 3*chi
    return a0*np.exp(-0.5*(f-f0)**2/sigma**2) + a1*np.exp(-0.5*(f-f1)**2/sigma**2)+ a2*np.exp(-0.5*(f-f2)**2/sigma**2)+ a3*np.exp(-0.5*(f-f3)**2/sigma**2)  + ofs

popt, pcov = curve_fit(func, freq, data, p0=(0,80,30,0,1,1,1,1))

print('f0 = %.1f kHz' %popt[0])
print('sigma = %.1f kHz' %popt[2])
print('chi = %.1f' %popt[1])

fig, axes = plt.subplots(1,2, figsize=(7,2), dpi=600)
ax = axes[0]
ax.set_xlabel(r'Detuning (kHz)')
ax.set_ylabel(r'Prob. of $|0\rangle$ (unnormalized)')
color = plt.get_cmap('tab10')(3)
ax.plot(freq, data, marker='.', linestyle='none', color=color, zorder=10)
ax.plot(freq, func(freq, *popt), color='black')
    


ax = axes[1]
ax.set_xlabel(r'Mean photon number, $\overline{n}$')
ax.set_ylabel(r'Rotation frequency (kHz)')


if SAVE_FIGURE:
    savedir = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\hamiltonian_parameters'
    savename = 'hamiltonian'
    fig.savefig(os.path.join(savedir, savename), fmt='pdf')
    
    
    




