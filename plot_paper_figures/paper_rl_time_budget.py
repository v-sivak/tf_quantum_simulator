# -*- coding: utf-8 -*-
"""
Created on Sun Mar 27 14:31:33 2022
"""
import numpy as np
import matplotlib.pyplot as plt
import plot_config

SAVE_FIGURE = True

fig, ax = plt.subplots(1,1, dpi=600, figsize=(3.1, 3.13))
ax.grid()

palette = plt.get_cmap('tab10')

# All time measured in [s]
components = [
    ('Compilation of FPGA \n instruction sequence', 6.91),
    ('Upload instructions and\n waveforms to FPGA', 0.2+3.84),
    # 3000 total shots per epoch, 160 QEC cycles per shot, 4.924 us per cycle, 
    # 10 us for initialization and logical readout
    (r'$T=160$ QEC cycles,'+'\n'+r'$N_{\rm tot}=3000$ shots ', 3000 * (160*4.924e-6 + 10e-6)),
    # 3000 total shots per epoch, 270 us active reset
    ('Active system reset\n between shots', 3000*270*1e-6),
    ('Collect and process\n measurement data', 0.44),
    ('Logging, NN update,\n inference, sampling,\n TCP/IP communication', 1.03)
    ]

y_pos = np.arange(len(components))
times = [t for (c,t) in components]
labels = [c for (c,t) in components]

ax.barh(y_pos, times, color='#219ebc', align='center', zorder=2)

ax.set_yticks(y_pos)
ax.set_yticklabels(labels)
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('Time (seconds)')



plt.tight_layout()

savename = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures\reinforcement_learning\time_budget.pdf'
if SAVE_FIGURE: fig.savefig(savename)