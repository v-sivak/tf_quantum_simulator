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

cal_data = np.load(r'Z:\shared\tmp\for Vlad\from_vlad\CD_calibration\CD_amp_cal.npz')
alphas, betas, data = cal_data['alphas'], cal_data['betas'], cal_data['data']
optimal_amps = cal_data['optimal_amps']

fig, axes = plt.subplots(1, 2, dpi=600, figsize=(4.7,2.05))
ax = axes[0]
ax.pcolormesh(betas, alphas, data, cmap='coolwarm', vmax=1, vmin=0)
ax.set_xlabel(r'Conditional disp. amp., $\beta$')
ax.set_ylabel(r'Large disp. amp., $\alpha$')
ax.plot(betas, optimal_amps, color='black', marker='.', linestyle='none')

def quadratic(beta, a, b, c):
    return a + b * beta + c * beta**2

# first clean up NaN from the array (where fit failed)
mask = np.isnan(optimal_amps)
optimal_amps = optimal_amps[np.where(mask==False)]
betas = betas[np.where(mask==False)]

# now clean up ouliers
if 0:
    popt, pcov = curve_fit(quadratic, betas, optimal_amps)
    a, b, c = popt
    
    sigma = np.abs(optimal_amps - quadratic(betas, *popt))
    mask = sigma>np.mean(sigma)
    optimal_amps = optimal_amps[np.where(mask==False)]
    betas = betas[np.where(mask==False)]

# now fit this to a linear function
popt, pcov = curve_fit(quadratic, betas, optimal_amps)
a, b, c = popt

betas_new = np.linspace(0, np.max(betas), 10)
ax.plot(betas_new, quadratic(betas_new, a, b, c), color='black')

plt.tight_layout()






cal_data = np.load(r'Z:\shared\tmp\for Vlad\from_vlad\CD_calibration\CD_phase_cal.npz')
betas, data = cal_data['betas'], cal_data['data']

ax = axes[1]
ax.set_xlabel(r'Conditional disp. amp., $\beta$')
ax.set_ylabel(r'$\langle\sigma_y\rangle=\sin(2\xi|\beta|^2)$')
ax.plot(betas, data, color='black', marker='.', linestyle='none')

def func(beta, xi):
    return np.sin(2*xi*beta**2)

popt, pcov = curve_fit(func, betas, data, p0=-0.2)
xi = popt[0]

ax.plot(betas, func(betas, xi), color='black', label=r'$\xi=$%.3f' %xi)
ax.legend()

plt.tight_layout()

savename = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\ECD_control\CD_cal.pdf'
if SAVE_FIGURE: fig.savefig(savename)
