# -*- coding: utf-8 -*-
"""
Created on Thu Sep 23 14:50:42 2021
"""
import numpy as np
import matplotlib.pyplot as plt
import plot_config
import os

SAVE_FIGURE = False
LEGEND = False

datadir = os.path.join(plot_config.data_root_dir, 'pulse_waveform')

filename = os.path.join(datadir, 'plus_Z.npz')
data = np.load(filename, allow_pickle=True)
c_pulse, q_pulse = data['c_pulse'], data['q_pulse']
time = np.arange(len(c_pulse))


fig, ax = plt.subplots(1, 1, figsize=(4.85, 1.5), dpi=600)
colors = plt.get_cmap('Paired')
# ax.grid(True)
ax.set_ylabel('DAC amplitude')
ax.plot(time*1e-3, c_pulse.real, label='I', color='#219ebc', linestyle='-')
ax.plot(time*1e-3, c_pulse.imag, label='Q', color='#219ebc', linestyle=':')
ax.set_xlabel(r'Time ($\mu$s)')
ax.plot(time*1e-3, q_pulse.real, label='I', color='#fb8500', linestyle='-')
ax.plot(time*1e-3, q_pulse.imag, label='Q', color='#fb8500', linestyle=':')

if LEGEND: ax.legend()
plt.tight_layout()

savename = os.path.join(plot_config.save_root_dir, 
                        r'ECD_control\gkp_plusZ_pulse.pdf')
if SAVE_FIGURE: fig.savefig(savename)




# fancy 3D version of the plot in complex plane and time
from mpl_toolkits import mplot3d
fig = plt.figure(dpi=600, figsize=(7,3))

ax = mplot3d.axes3d.Axes3D(fig, xlabel='Time (ns)', ylabel='I', zlabel='Q')
ax.set_zticks([-0.5, 0, 0.5])
ax.set_yticks([-0.5, 0, 0.5])
ax.view_init(20, -80)

ax.set_ylim(-0.5,0.5)
ax.set_zlim(-0.5,0.5)
ax.set_xlim(-100,5600)

ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False

ax.plot3D(time, c_pulse.real, c_pulse.imag, color=colors(1))
ax.plot3D(time, q_pulse.real, q_pulse.imag, color=colors(5))

plt.tight_layout()

savename = os.path.join(plot_config.save_root_dir, 
                        r'ECD_control\gkp_plusZ_pulse_3D.pdf')
if SAVE_FIGURE: fig.savefig(savename)