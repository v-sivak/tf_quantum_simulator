# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 11:19:02 2021

@author: Vladimir Sivak
"""
import tensorflow as tf
from tf_quantum_simulator import utils
from tf_quantum_simulator.quantum_control.layers import QubitRotation, ConditionalDisplacement
from tf_quantum_simulator.quantum_control.metrics import MinInfidelity
from tf_quantum_simulator.quantum_control.callbacks import PlotCallback

T = 7  # circuit depth
N = 30 # oscillator truncation

# need to use batch shape that starts with 1 to trick tensorflow into thinking it's batch of 1
batch_shape = [1,100]

ECDC = tf.keras.Sequential()
for i in range(T):
    ECDC.add(QubitRotation(N, batch_shape))
    ECDC.add(ConditionalDisplacement(N, batch_shape))

# create initial state and target state. Here target is Fock=1
vac = utils.Kronecker_product([utils.basis(0, 2, batch_shape), utils.basis(0, N, batch_shape)])
target = utils.Kronecker_product([utils.basis(0, 2, batch_shape), utils.basis(1, N, batch_shape)])

def loss(state, target):
    return tf.math.reduce_sum(utils.log_infidelity(state, target))

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# track optimization progress with tensorboard
logdir = r'E:\data\gkp_sims\PPO\ECD\test2'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

LearningRateScheduler = tf.keras.callbacks.LearningRateScheduler(lambda t: 1e-3/(1+1e-3*t))

# compile the model and run the optimizer
# ECDC.compile(optimizer=optimizer, loss=loss, metrics=[MinInfidelity()])
ECDC.compile(optimizer=optimizer, loss=loss)
ECDC.fit(x=vac, y=target, epochs=10000, callbacks=[tensorboard_callback, PlotCallback(vac, target)])