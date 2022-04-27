# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:24:27 2022

@author: qulab
"""
import numpy as np
import matplotlib.pyplot as plt
import plot_config

fig, ax = plt.subplots(1,1, dpi=600, figsize=(3.375/1.2, 4/1.2))
colors = plt.get_cmap('tab10')

ax.set_xlabel('Year')
ax.set_ylabel(r'Time constant, $1/\gamma_{\cal E}$ (us)')
ax.set_xlim(2015, 2023.5)

experiments = {
    'cat' : (2016, 287, 318),
    'bin' : (2019, 216, 200),
    r'GKP$_1$' : (2020, 370, 220),
    'T4C' : (2021, 440, 288),
    r'GKP$_2$' : (2022, 740, 1070),
    'd3 surf.' : (2021, 42.2, 17.5)
    }


# years = [2016, 2019, 2019, 2021, 2022]
# T_L = [318, 220, 200, 288, 1070]
# T_P = [287, 370, 216, 440, 740]

# ax.plot(years, T_L, marker='.', linestyle='none')
# ax.plot(years, T_P, marker='.', linestyle='none')


for exp_name, e in experiments.items():
    Y, T_P, T_L = e
    color = 'g' if T_L > T_P else colors(3)
    ax.plot([Y,Y], [T_P, T_L],  marker='.', linestyle='none', color='k', markersize=3)
    ax.arrow(Y, T_P, 0, T_L-T_P, width=0.001, head_width=0.1, head_length=15, 
             length_includes_head=True, color=color)
    ax.text(Y+0.2, min(T_P, T_L), exp_name)
    
# ax.grid()
    
# ax.set_ylim(0,1300)
# ax.set_yscale('log')