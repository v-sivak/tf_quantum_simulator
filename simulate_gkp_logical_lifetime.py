# -*- coding: utf-8 -*-
"""
Created on Wed Oct 27 11:04:12 2021
"""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf
from mc_simulator.gkp_simulator import Simulator
import numpy as np
import helper_functions as hf
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

""" This is a simple example of measureing GKP logical lifetime. """

N = 100 # Hilbert space truncation for oscillator
batch_size = 100 # number of quantum trajectories
Delta = 0.34 # code envelope parameter
steps = 600 # number of rounds to simulate
test_states = ['+X', '+Y']


params = dict(
    K = -4.8,
    chi = 46.5e3,
    chi_prime = 5.8,
    n_th = 0.05,
    T1_qb = 258e-6,
    T2_qb = 147e-6,
    T1_osc = 622e-6,
    gamma_phi_osc = 0,
    T2_osc = 923e-6,
    t_read = 2.4e-6,
    t_VR = 448e-9, # virtual rotation gate
    t_idle = 500e-9, # idle section
    t_sbs = [232e-9, 906e-9, 308e-9, (24+78+24)*1e-9],
    dt = 100e-9, # time discretization step or MC trajectories      
    p_qnd_e = 0.990,  # prob of process e -> e
    p_qnd_g = 0.9996, # prob of process g -> g
    p_dd = 0.16 # demolition detection probability
    )


# Create GKP code words and initialize the simulator
ideal_stabilizers, ideal_paulis, states, displacement_amplitudes = \
    hf.GKP_code(True, N, Delta=Delta, tf_states=True)
sim = Simulator(N, params)

pauli_m, rounds = {}, {}
for s in test_states:
    init_state = tf.concat([states[s]]*batch_size, axis=0)
    pauli_m[s], rounds[s] = [], []

    state = init_state
    for i in range(steps):
        quad = 'x' if i % 2 == 0 else 'p'
        state = sim.sbs(state, Delta, quad)
        state_copy = tf.identity(state)
        beta = displacement_amplitudes[s[-1]]
        _, m = sim.ideal_phase_estimation(state_copy, beta, sample=False)
        pauli_m[s].append(tf.reduce_mean(m))
        rounds[s].append(i)


# Fit to exponential decay 
def exp_decay(n, T, A, B):
    return A * np.exp(-n/T) + B

fig, ax = plt.subplots(1,1, dpi=200)
ax.set_xlabel('Round')
for s in test_states:
    ax.plot(np.array(rounds[s]), np.abs(pauli_m[s]))
    popt, pcov = curve_fit(exp_decay, np.array(rounds[s]), np.abs(pauli_m[s]))
    ax.plot(np.array(rounds[s]), exp_decay(np.array(rounds[s]), *popt), 
            color='black', linestyle='--', label = s+', T=%.f'%popt[0])
ax.legend()

