#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 23 12:09:16 2020

@author: Vladimir Sivak
"""
import numpy as np

# Units: seconds, Hz

### Vlad's system
if False:
    # Oscillator
    T1_osc = 530e-6
    T2_osc = None
    K_osc = 120

    # Qubit
    T1_qb = 90e-6
    T2_qb = 110e-6

    # Coupling
    chi = 28e3
    chi_prime = 200

    # Imperfections
    t_gate = 150e-9
    t_read = 0.4e-6
    t_feedback = 0.6e-6
    t_idle = 0.0

    # Simulator discretization
    discrete_step_duration = 100e-9

### Alec's system
if False:
    # Oscillator
    T1_osc = 245e-6
    T2_osc = None
    K_osc = 1

    # Qubit
    T1_qb = 50e-6
    T2_qb = 60e-6

    # Coupling
    chi = 28e3
    chi_prime = 0

    # Imperfections
    t_gate = 1.2e-6
    t_read = 0.6e-6
    t_feedback = 0.6e-6
    t_idle = 0.0

    # Simulator discretization
    discrete_step_duration = 100e-9

### Alec's units: Hamiltonian terms in GRad/s, time step is in ns
if False:
    # Oscillator
    chi = 2 * np.pi * 1e-6 * 80
    kappa = 1 / (1e6)
    kerr = 2 * np.pi * 1e-6 * 2

    # qubit
    gamma_1 = 1 / (50e3)
    gamma_phi = 0

    # Hilbert space size
    N = 25

    # Hilbert space size for intermediate calculation of displacement operators for tomography
    N_large = 100

    # Simulator discretization in ns
    discrete_step_duration = 5.0

if True:
    # Oscillator
    chi = 2 * np.pi * 1e-6 * 80
    kappa = 1 / (1e6)
    kerr = 0  # 2 * np.pi * 1e-6 * 2

    # qubit
    gamma_1 = 1 / (50e3)
    gamma_phi = 0

    # Hilbert space size
    N = 70

    # Hilbert space size for intermediate calculation of displacement operators for tomography
    N_large = 150

    # Simulator discretization in ns
    discrete_step_duration = 1.0

# Old sharpen trim experiment parameters
if False:
    # Oscillator
    chi = 2 * np.pi * 1e-6 * 29
    kappa = 1 / (250e3)
    kerr = 0  # 2 * np.pi * 1e-6 * 2

    # qubit
    gamma_1 = 1 / (50e3)
    gamma_phi = 0

    # Hilbert space size
    N = 70

    # Hilbert space size for intermediate calculation of displacement operators for tomography
    N_large = 150

    # Simulator discretization in ns
    discrete_step_duration = 10.0

