# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 14:31:33 2022

@author: qulab
"""
import numpy as np
import matplotlib.pyplot as plt
import plot_config

SAVE_FIGURE = True


fig, ax = plt.subplots(1,1, dpi=600, figsize=(3.375, 4))
palette = plt.get_cmap('tab10')

# Example data
time_budget_components = {}

time_budget_components['step_1'] = {
    'NN inference' : 0.1}

time_budget_components['step_2'] = {
    'Sampling of actions' : 0.2,
    'TCP/IP communication' : 0.1}

time_budget_components['step_3'] = {
    'Compilation of FPGA \n instruction sequence' : 6.0,
    'Upload instructions and \n waveforms to FPGA' : 3.0,
    'Reset FPGA' : 1.0}

time_budget_components['step_4'] = {
    r'$T=60$ steps of QEC' : 3.0,
    'System reset between \n experimental runs' : 2.0}

time_budget_components['step_5'] = {
    'Collect and process \n measurement data' : 0.4}

time_budget_components['step_6'] = {
    'Update the NN' : 0.2}


N_bars = 0
labels = []
times = []
colors = []
for i in range(1,7,1):
    color = palette(i)
    for k,v in time_budget_components['step_'+str(i)].items():
        N_bars += 1
        labels.append(k)
        times.append(v)
        colors.append(color)
y_pos = np.arange(N_bars)

ax.barh(y_pos, times, color=colors, align='center')

ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Time (seconds)')



savename = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\reinforcement_learning\time_budget.pdf'
if SAVE_FIGURE: fig.savefig(savename)