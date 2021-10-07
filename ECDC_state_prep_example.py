# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 11:19:02 2021
"""
import tensorflow as tf
import utils
from quantum_control.layers import QubitRotation, ConditionalDisplacement
# from quantum_control.metrics import MinInfidelity
from quantum_control.callbacks import PlotCallback, MinInfidelity

T = 12  # circuit depth
N = 100 # oscillator truncation

# use batch shape that starts with 1 to trick TF into thinking it's batch of 1
batch_shape = [1,200]

# build the circuit as a Keras model
ECDC = tf.keras.Sequential(name='ECDC')
for i in range(T):
    echo_pulse = True if i<T-1 else False
    ECDC.add(QubitRotation(N, batch_shape))
    ECDC.add(ConditionalDisplacement(N, batch_shape, echo_pulse=echo_pulse))

# create initial state and target state. Here target is Fock=1
vac = utils.Kronecker_product([utils.basis(0, 2, batch_shape), utils.basis(0, N, batch_shape)])
target = utils.Kronecker_product([utils.basis(0, 2, batch_shape), utils.basis(1, N, batch_shape)])

# target state is GKP sensor state with Delta=0.35
import helper_functions as hf
Delta = 0.30
target = hf.GKP_1D_state(True, N, Delta=Delta)

# define the loss function and optimizer
def loss(state, target):
    return tf.math.reduce_sum(utils.log_infidelity(state, target))

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# can track optimization progress with tensorboard
# logdir = r'E:\data\gkp_sims\PPO\ECD\test2'
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

# Custom callbacks
plot_callback = PlotCallback(vac, target)
fidelity_callback = MinInfidelity(vac, target)

LearningRateScheduler = tf.keras.callbacks.LearningRateScheduler(lambda t: 1e-3/(1+1e-3*t))

# compile the model and run the optimizer
ECDC.compile(optimizer=optimizer, loss=loss, metrics=[])
ECDC.fit(x=vac, y=target, epochs=5000, 
         callbacks=[plot_callback, fidelity_callback])


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
fname = r'E:\data\gkp_sims\PPO\ECD\EXP_Vlad\ECDC_sequences\gkp_sensor_T_%d_Delta_%.2f_F_%.4f.npz' %(T,Delta,F)
np.savez(fname, **params)