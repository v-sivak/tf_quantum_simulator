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



fig, axes = plt.subplots(1, 3, dpi=600, figsize=(7,2.05), sharey=True, 
                         gridspec_kw={'width_ratios': [1, 1, 1.5]})
colors = plt.get_cmap('tab10')


chi = 104 # kHz
h = 6.626e-34
f = 4.48e9
k = 1.38e-23


# 1st panel: passive cooling qubit spec
cal_data = np.load(r'Z:\shared\tmp\for Vlad\from_vlad\cavity_cooling\spec_passive.npz')
freq, data = cal_data['freq'], cal_data['data']
ax = axes[0]
ax.set_title('Passive cooling')
ax.set_ylabel(r'Prob. of $|0\rangle$ (unnormalized)')
ax.set_xlabel(r'Detuning (kHz)')

def func(f, a0, a1, sigma, f0, ofs):
    f1 = f0 - chi
    return a0*np.exp(-0.5*(f-f0)**2/sigma**2) + a1*np.exp(-0.5*(f-f1)**2/sigma**2) + ofs

popt, pcov = curve_fit(func, freq, data.mean(axis=0), p0=(1,0,50,0,0))
a0, a1, sigma, f0, ofs = popt
pop_ratio = (popt[1]-popt[-1])/(popt[0]-popt[-1])
gauss0 = (a0, 0, sigma, f0, ofs)
gauss1 = (0, a1, sigma, f0, ofs)

print('Passive cooling:')
print('f0 = %.1f kHz' %f0)
print('sigma = %.1f kHz' %sigma)
print('p1/p0 = %.3f' % pop_ratio)

ax.plot(freq, data.mean(axis=0), marker='.', linestyle='none', 
        color=colors(3))

ax.plot(freq, func(freq, *popt), linestyle='-', color='black')
ax.plot(freq, func(freq, *gauss0), linestyle='--', color=colors(-1))
ax.plot(freq, func(freq, *gauss1), linestyle='--', color=colors(-1))

T = h*f/k/np.log(1/pop_ratio)
print('T = %.3f K' % T)

max_pop_passive = np.max(data.mean(axis=0))

# 2nd panel: active cooling
cal_data = np.load(r'Z:\shared\tmp\for Vlad\from_vlad\cavity_cooling\spec_active.npz')
freq, data = cal_data['freq'], cal_data['data']
ax = axes[1]
ax.set_title('Active cooling')

def func(f, a0, sigma, f0, ofs):
    return a0*np.exp(-0.5*(f-f0)**2/sigma**2) + ofs

popt, pcov = curve_fit(func, freq, data.mean(axis=0), p0=(1,50,0,0))
a0, sigma, f0, ofs = popt

print('\nActive cooling:')
print('f0 = %.1f kHz' %f0)
print('sigma = %.1f kHz' %sigma)

ax.plot(freq, data.mean(axis=0), marker='.', linestyle='none', 
        color=colors(3))

ax.plot(freq, func(freq, *popt), linestyle='-', color='black')



# 3rd panel: cooling from GKP state
cal_data = np.load(r'Z:\shared\tmp\for Vlad\from_vlad\cavity_cooling\gkp_ECDC_cooling.npz')
rounds = cal_data['rounds']
data = {}
for e in list(cal_data):
    if '0' in e:
        data[float(e)] = cal_data[e]

ax = axes[2]
ax.set_title(r'ECD cooling from GKP $|+Z\rangle$')
ax.set_xlabel(r'Rounds')

for e in np.sort(list(data.keys())):
    ax.plot(rounds, data[e], marker='.', linestyle='none', label=e)
ax.legend(title=r'$\varepsilon$', ncol=1)


plt.tight_layout()
savename = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\cavity_cooling\cavity_cooling.pdf'
if SAVE_FIGURE: fig.savefig(savename)




# Plot the same things, but in different figures
##########################################################################
##########################################################################
##########################################################################
##########################################################################

fig, axes = plt.subplots(1, 2, dpi=600, figsize=(5,2.05), sharey=True)
colors = plt.get_cmap('tab10')


chi = 104 # kHz
h = 6.626e-34
f = 4.48e9
k = 1.38e-23


# 1st panel: passive cooling qubit spec
cal_data = np.load(r'Z:\shared\tmp\for Vlad\from_vlad\cavity_cooling\spec_passive.npz')
freq, data = cal_data['freq'], cal_data['data']
ax = axes[0]
ax.set_ylim(0,1)
# ax.set_title('Passive cooling')
ax.set_ylabel(r'Prob. of $|0\rangle$ (unnormalized)')
ax.set_xlabel(r'Detuning (kHz)')

def func(f, a0, a1, sigma, f0, ofs):
    f1 = f0 - chi
    return a0*np.exp(-0.5*(f-f0)**2/sigma**2) + a1*np.exp(-0.5*(f-f1)**2/sigma**2) + ofs

popt, pcov = curve_fit(func, freq, data.mean(axis=0), p0=(1,0,50,0,0))
a0, a1, sigma, f0, ofs = popt
pop_ratio = (popt[1]-popt[-1])/(popt[0]-popt[-1])
gauss0 = (a0, 0, sigma, f0, ofs)
gauss1 = (0, a1, sigma, f0, ofs)

# print('Passive cooling:')
print('f0 = %.1f kHz' %f0)
print('sigma = %.1f kHz' %sigma)
print('p1/p0 = %.3f' % pop_ratio)


ax.plot(freq, data.mean(axis=0), marker='.', linestyle='none', 
        color=colors(3))

ax.plot(freq, func(freq, *popt), linestyle='-', color='black')
ax.plot(freq, func(freq, *gauss0), linestyle='--', color=colors(-1))
ax.plot(freq, func(freq, *gauss1), linestyle='--', color=colors(-1))

T = h*f/k/np.log(1/pop_ratio)
print('T = %.3f K' % T)


# 2nd panel: active cooling
cal_data = np.load(r'Z:\shared\tmp\for Vlad\from_vlad\cavity_cooling\spec_active.npz')
freq, data = cal_data['freq'], cal_data['data']
ax = axes[1]
# ax.set_title('Active cooling')

def func(f, a0, sigma, f0, ofs):
    return a0*np.exp(-0.5*(f-f0)**2/sigma**2) + ofs

popt, pcov = curve_fit(func, freq, data.mean(axis=0), p0=(1,50,0,0))
a0, sigma, f0, ofs = popt

print('\nActive cooling:')
print('f0 = %.1f kHz' %f0)
print('sigma = %.1f kHz' %sigma)

ax.plot(freq, data.mean(axis=0), marker='.', linestyle='none', 
        color=colors(3))

ax.plot(freq, func(freq, *popt), linestyle='-', color='black')


plt.tight_layout()
savename = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\cavity_cooling\qubit_spec.pdf'
if SAVE_FIGURE: fig.savefig(savename)





# cooling from GKP state
fig, ax = plt.subplots(1, 1, dpi=600, figsize=(3, 1.9))
ax.set_ylabel(r'Prob. of $|0\rangle$ (unnormalized)')
ax.set_ylim(0,1)


cal_data = np.load(r'Z:\shared\tmp\for Vlad\from_vlad\cavity_cooling\gkp_ECDC_cooling.npz')
rounds = cal_data['rounds']
data = {}
for e in list(cal_data):
    if '0' in e and '4' not in e:
        data[float(e)] = cal_data[e]

# ax.set_title(r'ECD cooling from GKP $|+Z\rangle$')
ax.set_xlabel(r'Rounds')

for e in np.sort(list(data.keys())):
    ax.plot(rounds, data[e], marker='.', linestyle='none', label=e)

ax.plot(rounds, np.ones_like(rounds)*max_pop_passive, linestyle='--', color=colors(0))
ax.legend(title=r'$\varepsilon$', ncol=1)


plt.tight_layout()
savename = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\cavity_cooling\gkp_ecdc_cooling.pdf'
if SAVE_FIGURE: fig.savefig(savename)