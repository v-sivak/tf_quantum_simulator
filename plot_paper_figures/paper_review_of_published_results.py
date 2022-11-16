# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 14:24:27 2022
"""
import numpy as np
import matplotlib.pyplot as plt
import plot_config_thesis

SAVE_FIGURE = False

# This experiment best results
tc = 4.924 * 2
Tx = 2200
Ty = 1360
Tz = 2210
T1 = 610
T2 = 950

T_fock = 3/(2/T2+1/T1)
T_gkp = 3/(1/Tx+1/Ty+1/Tz)
pc = ((1-np.exp(-tc/Tx))/2 + (1-np.exp(-tc/Ty))/2 + (1-np.exp(-tc/Tz))/2)/3


# Format: (Year, Best physical lifetime, Logical lifetime, Error probability)
# error prob. per round is calculated for experiments where it is not provided
experiments = {
    '4C' : (2016, 287, 318, 20/2/318), # https://www.nature.com/articles/nature18949
    'bin' : (2019, 216, 200, 17.9/2/200), # https://www.nature.com/articles/s41567-018-0414-3#Sec5
    r'GKP$_1$' : (2020, 370, 220, 4*2.53/2/370), # https://www.nature.com/articles/s41586-020-2603-3
    'T4C' : (2021, 440, 288, None), # https://www.nature.com/articles/s41586-021-03257-0 
    r'GKP$_2$' : (2022.7, T_fock, T_gkp, pc),
    r'd3$_1$' : (2021.5, 76.9, 17.8, 0.030),  # ETH https://www.nature.com/articles/s41586-022-04566-8
    r'd3$_2$' : (2022, 12.9, 6.3, 0.223), # China https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.129.030501
    r'd3$_3$' : (2022.3, 346.6, 25.2, 0.095),  # IBM https://arxiv.org/abs/2203.07205
    r'd5' : (2022.6, 35.6, 15.3, 0.029), # Google http://arxiv.org/abs/2207.06431
    }



# Make a figure
fig, axes = plt.subplots(1,2, dpi=600, figsize=(4.95, 3))

colors = plt.get_cmap('tab10')

ax = axes[0]
ax.set_xlabel('Year')
ax.set_title(r'Logical and best physical lifetime ($\rm \mu s$)')
ax.set_xlim(2015.5, 2023.7)
ax.set_ylim(-50, 1900)
ax.set_yticks([0,300,600,900,1200,1500,1800])
# ax.grid(lw=0.5, axis='y')
# ax.set_yscale('log')

ax = axes[1]
ax.set_xlabel('Year')
ax.set_title(r'Logical error probability per QEC cycle')
ax.set_yscale('log')
ax.set_xlim(2015.5, 2023.5)
ax.set_ylim(1e-3, 3e-1)
# ax.grid(lw=0.5, which='minor', axis='y')

for exp_name, e in experiments.items():
    Y, T_P, T_L, p_L = e
    
    ax = axes[0]
    color = 'g' if T_L > T_P else colors(3)
    ax.plot([Y,Y], [T_P, T_L],  marker='.', linestyle='none', color='k', markersize=4)
    
    l = 40 # length of arrow head
    arrow_offset = (T_L-T_P)/2+l*np.sign(T_L-T_P)/2 if np.abs(T_L-T_P)>l else l*np.sign(T_L-T_P)
    ax.arrow(Y, T_P, 0, arrow_offset, width=0.01, head_width=0.2, head_length=l, 
              length_includes_head=True, color=color)
    ax.arrow(Y, T_P, 0, (T_L-T_P), width=0.01, head_width=0, head_length=0, 
              length_includes_head=True, color=color)
    # ax.text(Y+0.2, min(T_P, T_L)-50, exp_name)
    
    ax = axes[1]
    marker = '.' if exp_name != r'GKP$_2$' else '*'
    color = 'k' if exp_name != r'GKP$_2$' else 'g'
    markersize = 6 if exp_name == r'GKP$_2$' else None
    ax.plot([Y], [p_L],  marker=marker, linestyle='none', color=color, markersize=markersize)


# ax.grid(lw=0.5)

plt.tight_layout()



if SAVE_FIGURE:
    savename = r'E:\VladGoogleDrive\Qulab\GKP\paper_qec\figures_working\qec_literature_review\review_1'
    fig.savefig(savename, fmt='.pdf')
    
    
    






### Google's distance-5 surface code
T1 = [22.3, 21.4, 
      22.2, 17.3, 21.4, 12.9, 
      22.9, 26.0, 20.0, 19.8, 22.2, 19.0, 
      21.7, 23.1, 25.3, 22.1, 22.2, 19.4, 19.3, 19.2, 
      18.6, 21.2, 22.0, 26.0, 16.6, 22.0, 23.7, 19.8, 19.2, 
      17.5, 17.2, 17.9, 19.9, 24.3, 24.5, 23.0, 13.0, 
      18.0, 22.5, 23.8, 26.8, 18.3, 21.4,
      16.3, 19.3, 19.4, 15.3,
      23.0, 18.5]

T2 = [41.4, 27.6,
      28.2, 23.9, 37.6, 36.0,
      37.7, 18.7, 39.3, 16.7, 35.4, 36.8,
      52.3, 26.6, 35.9, 34.4, 28.1, 28.6, 31.6, 45.5,
      29.4, 31.7, 27.7, 14.8, 24.7, 16.5, 34.3, 29.4, 41.1,
      31.9, 26.3, 30.1, 29.1, 36.4, 18.7, 35.2, 19.4,
      27.1, 35.0, 32.7, 38.9, 28.4, 39.4,
      34.3, 21.9, 33.2, 23.7,
      26.9, 34.6]

T1 = np.array(T1); T2 = np.array(T2)

print('--Google d5--')
T = 3/(2/T2+1/T1)
ind = np.argmax(T)
print('Physical %.1f' %T[ind])

t_cycle = 0.921 # [us]
p_error = 0.02914 # average over Z_ and X_L

T_d5 = -1 / np.log(1-2*p_error) * t_cycle
print('Logical %.1f' %T_d5)
print('G=%.2f'%(T_d5/T[ind]))
print('avg error per cycle: %.3f' %p_error)


### Chinese distance-3 surface code
T1 = [26.4, 27.1, 17.3, 28.0, 23.8, 16.2, 23.3, 17.8,
      23.0, 31.5, 36.6, 31.6, 31.4, 30.7, 29.9, 33.0, 25.5]

T2 = [8.2, 6.9, 5.9, 8.3, 7.9, 3.3, 5.8, 6.6,
      6.4, 10.0, 7.7, 3.5, 4.7, 4.4, 3.7, 4.2, 4.8]

T1 = np.array(T1); T2 = np.array(T2)

print('--China d3--')
T = 3/(2/T2+1/T1)
ind = np.argmax(T)
print('Physical %.1f' %T[ind])

t_cycle = 3.753  # [us]
p_error = (2*0.23+1*0.21)/3 # 0.21 for |0> or 0.23 for |-> 

T_d3 = -1 / np.log(1-2*p_error) * t_cycle
print('Logical %.1f' %T_d3)
print('G=%.2f'%(T_d3/T[ind]))
print('avg error per cycle: %.3f' %p_error)


### ETH distance-3 surface code
T1 = [29.1, 33.0, 65.5, 59.3, 32.2, 60.2, 36.0, 33.1, 25.9,
      12.4, 17.4, 18.5, 12.6, 17.0, 42.7, 29.7, 27.5]

T2 = [46.2, 52.2, 49.3, 75.5, 56.1, 89.3, 72.5, 59.1, 36.3,
      15.8, 33.0, 16.1, 33.1, 31.5, 26.0, 53.6, 53.6]

T1 = np.array(T1); T2 = np.array(T2)

print('--ETH d3--')
T = 3/(2/T2+1/T1)
ind = np.argmax(T)
print('Physical %.1f' %T[ind])

t_cycle = 1.1  # [us]
p_error = (2*0.029+1*0.032)/3 # 0.029 for |+>, or 0.032 for |0>

T_d3 = -1 / np.log(1-2*p_error) * t_cycle
print('Logical %.1f' %T_d3)
print('G=%.2f'%(T_d3/T[ind]))
print('avg error per cycle: %.3f' %p_error)


### IBM distance-3 heavy hexagonal code
T1 = [420.3, 354.8, 331.7, 124.8, 131.7, 424.5, 249.4, 271.7, 357.0, 283.8, 280.9, 
      349.8, 399.3, 226.6, 259.8, 234.4, 195.5, 319.6, 278.1, 206.9, 278.3, 258.7, 364.1]

T2 = [118.4, 119.8, 25.8, 77.3, 215.5, 59.7, 228.8, 316.0, 72.0, 188.8, 353.0, 
      345.0, 99.7, 217.4, 209.2, 311.7, 34.7, 167.6, 308.0, 132.4, 145.0, 14.6, 327.5]

T1 = np.array(T1); T2 = np.array(T2)

print('--IBM d3--')
T = 3/(2/T2+1/T1)
ind = np.argmax(T)
print('Physical %.1f' %T[ind])

t_cycle = 5.3  # [us]
p_error = (2*0.113+1*0.059)/3 # 0.113 for |+>, or 0.059 for |0>

T_d3 = -1 / np.log(1-2*p_error) * t_cycle
print('Logical %.1f' %T_d3)
print('G=%.2f'%(T_d3/T[ind]))
print('avg error per cycle: %.3f' %p_error)