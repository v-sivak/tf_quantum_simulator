# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:43:04 2022

@author: qulab
"""
import numpy as np
import matplotlib.pyplot as plt
import plot_config
from scipy.optimize import curve_fit

SAVE_FIGURE = True
LEGEND = False

fig, axes = plt.subplots(1, 3, dpi=600, figsize=(7,2.05))

color = plt.get_cmap('tab10')(3)

# RABI
cal_data = np.load(r'Z:\shared\tmp\for Vlad\from_vlad\primitive_pulses\Rabi.npz')
amps, data = cal_data['amps'], cal_data['data']

ax = axes[0]
ax.set_ylabel(r'Prob. of $|e\rangle$')
ax.plot(amps, data, marker='.', linestyle='none', color=color, zorder=10)

def func(amp, a, b, c):
    return a * np.sin(np.pi/2 * amp / b)**2 + c

popt, pcov = curve_fit(func, amps, data, p0=(1,1,0))
ax.plot(amps, func(amps, *popt), color='black')



# DISPLACEMENT
cal_data = np.load(r'Z:\shared\tmp\for Vlad\from_vlad\primitive_pulses\displacement.npz')
amps, data = cal_data['amps'], cal_data['data']

ax = axes[1]
ax.set_xlabel(r'Pulse amplitude (relative to calibration)')
ax.set_ylabel(r'Prob. of $|0\rangle$')
ax.plot(amps, data, marker='.', linestyle='none', color=color, zorder=10)

def func(amp, a, b, c):
    return a * np.exp(-(amp/b)**2) + c

popt, pcov = curve_fit(func, amps, data, p0=(1,1,0))
ax.plot(amps, func(amps, *popt), color='black')



# WIGNER
cal_data = np.load(r'Z:\shared\tmp\for Vlad\from_vlad\primitive_pulses\wigner.npz')
amps, data = cal_data['amps'], cal_data['data']

ax = axes[2]
ax.set_ylabel(r'$W(\alpha)$')
ax.plot(amps, 1-2*data, marker='.', linestyle='none', color=color, zorder=10)

def func(amp, a, b, c):
    return a * np.exp(-2*(amp/b)**2) + c

popt, pcov = curve_fit(func, amps, 1-2*data, p0=(1,0.5,0))
ax.plot(amps, func(amps, *popt), color='black')


plt.tight_layout()
savename = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\primitive_pulses\primitive_pulses.pdf'
if SAVE_FIGURE: fig.savefig(savename)
