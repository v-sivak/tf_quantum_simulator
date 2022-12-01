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

T = 18  # circuit depth
N = 120 # oscillator truncation

# use batch shape that starts with 1 to trick TF into thinking it's batch of 1
batch_shape = [1,200]

# build the circuit as a Keras model
ECDC = tf.keras.Sequential(name='ECDC')
for i in range(T):
    echo_pulse = True if i<T-1 else False
    ECDC.add(QubitRotation(N, batch_shape))
    ECDC.add(ConditionalDisplacement(N, batch_shape, echo_pulse=echo_pulse))

if 1: # GKP encoding
    import helper_functions as hf
    Delta = 0.32
    ideal_stabilizers, ideal_paulis, gkp_states, displacement_amplitudes = \
        hf.GKP_code(True, N, Delta=Delta, tf_states=True)


transmon_states =  {
    '+Z' : utils.Kronecker_product([utils.basis(0,2), utils.basis(0,N)]),
    '-Z' : utils.Kronecker_product([utils.basis(1,2), utils.basis(0,N)]),
    '+X' : utils.Kronecker_product([utils.normalize(utils.basis(0,2) + utils.basis(1,2))[0], utils.basis(0,N)]),
    '-X' : utils.Kronecker_product([utils.normalize(utils.basis(0,2) - utils.basis(1,2))[0], utils.basis(0,N)]),
    '+Y' : utils.Kronecker_product([utils.normalize(utils.basis(0,2) + 1j*utils.basis(1,2))[0], utils.basis(0,N)]),
    '-Y' : utils.Kronecker_product([utils.normalize(utils.basis(0,2) - 1j*utils.basis(1,2))[0], utils.basis(0,N)])
    }

# # define input and output states
cardinal_points = ['+Z','-Z','+X','-X','+Y','-Y']
inputs = tf.stack([transmon_states[s] for s in cardinal_points])
targets =  tf.stack([gkp_states[s] for s in cardinal_points])

# define the loss function and optimizer
# def loss(state, target):
#     return -tf.math.real(tf.math.reduce_sum(utils.batch_dot(state, target)))

# def loss(state, target):
#     return tf.math.log(1-tf.math.real(tf.math.reduce_mean(utils.batch_dot(state, target))))

def loss(state, target):
    return tf.math.log(1-tf.math.real(tf.math.reduce_mean(tf.math.abs(utils.batch_dot(state, target))**2)))

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-2)

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
fname = r'E:\data\gkp_sims\PPO\ECD\EXP_Vlad\ECDC_sequences\gkp_encoding_unitary_T_%d_Delta_%.2f_F_%.4f.npz' %(T,Delta,F)
np.savez(fname, **params)


# # Plot the effect of this gate on different basis states
# for name in cardinal_points:
#     input_state = transmon_states[name]
    
#     hf.plot_phase_space(input_state, True, phase_space_rep='wigner', title='Input ' + name)

#     title = 'Input ' + name
#     hf.plot_phase_space(ECDC(input_state)[:,ind,:], True, phase_space_rep='wigner', title=title)
