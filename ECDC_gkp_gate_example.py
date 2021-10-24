# -*- coding: utf-8 -*-
"""
Created on Wed Oct  6 13:06:05 2021
"""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"


import tensorflow as tf
import utils
from quantum_control.layers import QubitRotation, ConditionalDisplacement
# from quantum_control.metrics import MinInfidelity
from quantum_control.callbacks import PlotCallback, MinInfidelity
from math import pi

T = 5  # circuit depth
N = 120 # oscillator truncation

# use batch shape that starts with 1 to trick TF into thinking it's batch of 1
batch_shape = [1,500]

# build the circuit as a Keras model
ECDC = tf.keras.Sequential(name='ECDC')
for i in range(T):
    echo_pulse = True if i<T-1 else False
    ECDC.add(QubitRotation(N, batch_shape))
    ECDC.add(ConditionalDisplacement(N, batch_shape, echo_pulse=echo_pulse))

# create GKP code words with Delta=0.35
import helper_functions as hf
Delta = 0.30
ideal_stabilizers, ideal_paulis, states, displacement_amplitudes = \
    hf.GKP_code(True, N, Delta=Delta, tf_states=True)


### specify how the gate acts on any basis of states. This is for S gate
gate_map = {'+X' : '+Y', 
            '-X' : '-Y'}

gate_map_extended = {
    '+X' : '+Y', 
    '-X' : '-Y',
    '+Y' : '-X',
    '-Y' : '+X',
    '+Z' : '+Z',
    '-Z' : '-Z'} # this one also has a global phase of i

phase = {'+X' : 1, 
         '-X' : 1}


# ###  specify how the gate acts on any basis of states. This is for S_dag gate
# gate_map = {'+X' : '-Y', 
#             '-X' : '+Y'}

# gate_map_extended = {
#     '+X' : '-Y', 
#     '-X' : '+Y',
#     '+Y' : '+X',
#     '-Y' : '-X',
#     '+Z' : '+Z',
#     '-Z' : '-Z'} # this one also has a global phase of -i

# phase = {'+X' : 1, 
#           '-X' : 1}


####  specify how the gate acts on any basis of states. This is for X-gate
# gate_map = {'+Z' : '-Z', 
#             '-Z' : '+Z'}

# gate_map_extended = {
#     '+Z' : '-Z',
#     '-Z' : '+Z',
#     '+Y' : '-Y', 
#     '-Y' : '+Y', 
#     '+X' : '+X', 
#     '-X' : '-X'}

# phase = {'+Z' : 1, 
#          '-Z' : 1}


####  specify how the gate acts on any basis of states. This is for Y-gate
# gate_map = {'+Z' : '-Z', 
#             '-Z' : '+Z'}

# gate_map_extended = {
#     '+Z' : '-Z',
#     '-Z' : '+Z',
#     '+Y' : '+Y', 
#     '-Y' : '-Y', 
#     '+X' : '-X', 
#     '-X' : '+X'}

# phase = {'+Z' : 1j, 
#          '-Z' : -1j}


### specify how the gate acts on any basis of states. This is for Z-gate
# gate_map = {'+X' : '-X', 
#             '-X' : '+X'}

# gate_map_extended = {
#     '+X' : '-X',
#     '-X' : '+X',
#     '+Y' : '-Y', 
#     '-Y' : '+Y', 
#     '+Z' : '+Z', 
#     '-Z' : '-Z'}

# phase = {'+X' : 1, 
#          '-X' : 1}


# # specify how the gate acts on any basis of states. This is for T-gate
# gate_map = {'+Z' : '+Z', 
#             '-Z' : '-Z'}

# phase = tf.cast(tf.math.exp(1j*pi/4), tf.complex64)

# inputs = tf.stack([states['+X'], states['-X']])
# targets = tf.stack([
#     utils.normalize(states['+Z'] + phase*states['-Z'])[0],
#     utils.normalize(states['+Z'] - phase*states['-Z'])[0]])
    

# define input and output states
inputs = tf.stack([states[i] for (i,f) in gate_map.items()])
targets =  tf.stack([states[f]*phase[i] for (i,f) in gate_map.items()])

# define the loss function and optimizer
def loss(state, target):
    return -tf.math.real(tf.math.reduce_sum(utils.batch_dot(state, target)))

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# can track optimization progress with tensorboard
# logdir = r'E:\data\gkp_sims\PPO\ECD\test'
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Custom callbacks
plot_callback = PlotCallback(inputs, targets)
fidelity_callback = MinInfidelity(inputs, targets)

# LearningRateScheduler = tf.keras.callbacks.LearningRateScheduler(lambda t: 1e-3/(1+1e-3*t))

# compile the model and run the optimizer
ECDC.compile(optimizer=optimizer, loss=loss, metrics=[])
ECDC.fit(x=inputs, y=targets, epochs=10000, 
          callbacks=[fidelity_callback, plot_callback])


# Get the ECDC parameters as a dictionary for use in experiment
def get_ECDC_params(model, ind):
    params = dict(angle=[], phase=[], beta_re=[], beta_im=[])
    
    for layer in ECDC.layers:
        for var in layer.trainable_variables:
            for pname in params.keys():
                if pname in var.name:
                    params[pname].append(float(var[0,ind]))
    return params

# save the best protocol to a file 
ind = fidelity_callback.index[-1]
F = 1 - fidelity_callback.infidelities[-1]
params = get_ECDC_params(ECDC, ind)

import numpy as np
fname = r'E:\data\gkp_sims\PPO\ECD\EXP_Vlad\ECDC_sequences\S_gate_T_%d_Delta_%.2f_F_%.4f.npz' %(T,Delta,F)
np.savez(fname, **params)


# Plot the effect of this gate on different basis states
for name in states.keys():
    input_state = states[name]
    
    hf.plot_phase_space(input_state, True, phase_space_rep='CF', title='Input ' + name)
    
    title = 'Input ' + name + '; Expected output ' + gate_map_extended[name]
    hf.plot_phase_space(ECDC(input_state)[:,ind,:], True, phase_space_rep='CF', title=title)
