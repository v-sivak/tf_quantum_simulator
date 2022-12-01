# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 14:43:04 2022
"""
import numpy as np
import matplotlib.pyplot as plt
import plot_config
from scipy.optimize import curve_fit
import os

datadir = os.path.join(plot_config.data_root_dir, 'CD_calibration')

SAVE_FIGURE = False
LEGEND = False


### FIGURE 1: AMPLITUDE CALIBRATION
cal_data = np.load(os.path.join(datadir, 'CD_amp_cal.npz'))
alphas, betas, data = cal_data['alphas'], cal_data['betas'], cal_data['data']
optimal_amps = cal_data['optimal_amps']

fig, axes = plt.subplots(1, 2, dpi=600, figsize=(4.7,2.05),
                         gridspec_kw={'width_ratios': [0.8, 1]})
ax = axes[0]
ax.pcolormesh(betas, alphas, data, cmap='coolwarm', vmax=1, vmin=0)
ax.set_xlabel(r'Conditional displacement amp., $\beta$')
ax.set_ylabel(r'Large displacement amp., $\alpha$')
ax.plot(betas, optimal_amps, color='black', marker='.', linestyle='none')
ax.set_ylim(alphas[0], alphas[-1])
ax.set_xticks([0,3,6])

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


cal_data = np.load(os.path.join(datadir, 'CD_cal_empirical_fit.npz'))
xdata, ydata, popt, alpha_max = cal_data['xdata'], cal_data['ydata'], cal_data['popt'], cal_data['alpha_max']


def time_func(amps, chi, t0, c):
    alpha, beta = amps
    return beta*(c+0.5/alpha/chi)-t0

# Plot the data and fit 
ax = axes[1]
ax.set_ylabel(r'Wait time, $\tau$ (ns)')
ax.set_xlabel(r'Large displacement amp., $\alpha$')

pts = int(len(ydata) / len(np.unique(xdata[1])))

for i in range(len(np.unique(xdata[1]))):
    # first plot data points from experimental calibration
    amps = xdata[:,i*pts:(i+1)*pts]
    taus = ydata[i*pts:(i+1)*pts]

    ind = np.argsort(taus)
    taus = np.sort(taus)
    amps = amps[:,ind]
    alphas, betas = amps
    
    ax.plot(alphas, taus, marker='o', linestyle='none', label='%.1f'%betas[0])
    
    # next plot the fit lines for a denser set of alphas
    alphas = np.linspace(min(alphas), max(alphas), 201)
    betas = np.ones_like(alphas)*betas[0]
    amps = [alphas, betas]
    
    ax.plot(alphas, time_func(amps, *popt), linestyle='--', color='black')

# ax.plot(np.ones_like(taus)*alpha_max, taus, linestyle=':', color='brown')
# ax.plot(np.ones_like(taus)*0.0, taus, linestyle=':', color='brown')

# included a shaded region of prohibited parameter values
ax.fill_between([0,alpha_max], [-100,-100], [0,0], color='grey', alpha=0.25)
ax.fill_between([-10,0], [-100,-100], [1200,1200], color='grey', alpha=0.25)
ax.fill_between([alpha_max,40], [-100,-100], [1200,1200], color='grey', alpha=0.25)

ax.set_xlim(-2, alpha_max+10)
ax.set_ylim(-50, 1100)

if LEGEND: ax.legend(title=r'$\beta$', markerscale=1)
plt.tight_layout()


savename = os.path.join(plot_config.save_root_dir, 
                        r'CD_calibration\CD_cal.pdf')
if SAVE_FIGURE: fig.savefig(savename)


### FIGURE 2: PHASE CALIBRATION
SAVE_FIGURE = False
LEGEND = False

data = np.load(os.path.join(datadir, '+Z_purity_cal.npz'))


sigma_x = data['sigma_x']
sigma_y = data['sigma_y']
sigma_z = data['sigma_z']
betas = data['xs']
purity = data['purity']

# Plot qubit phase and purity

fig, ax = plt.subplots(1, 1, dpi=600, figsize=(2.5,2.05))
ax.set_ylabel(r'$\langle \sigma_x \rangle, \, \langle \sigma_y \rangle, \, \langle \sigma_z \rangle$')
ax.set_xlabel(r'Conditional displacement amp., $\beta$')
ax.set_xticks([-8,-4,0,4,8])
ax.set_ylim(-1, 1)

ax.plot(betas, sigma_z, marker='.', linestyle='none')
ax.plot(betas, sigma_y, marker='.', linestyle='none')
ax.plot(betas, sigma_x, marker='.', linestyle='none')

ax.plot(betas, purity, marker='x', linestyle='none')


def purity_fit(beta, p0, a, b):
    return (p0 - a * beta**2 - b * beta**4)**2


popt_purity, pcov_purity = curve_fit(purity_fit, betas, purity, p0=(0.9, 0, 0))

ax.plot(betas, purity_fit(betas, *popt_purity), color='k')
# axes[0].plot(betas, purity_fit(betas, *popt_purity), color='k')

def phase_sin_fit(beta, xi):
    return np.sin(2*xi*beta**2) * np.sqrt(purity_fit(beta, *popt_purity))

def phase_cos_fit(beta, xi):
    return np.cos(2*xi*beta**2) * np.sqrt(purity_fit(beta, *popt_purity))


popt_phase, pcov_phase = curve_fit(phase_sin_fit, betas, sigma_y, p0=(-0.04))

ax.plot(betas, phase_sin_fit(betas, *popt_phase), color='k')
ax.plot(betas, phase_cos_fit(betas, *popt_phase), color='k')

plt.tight_layout()

savename = os.path.join(plot_config.save_root_dir, 
                        r'CD_calibration\CD_cal_2.pdf')
if SAVE_FIGURE: fig.savefig(savename)