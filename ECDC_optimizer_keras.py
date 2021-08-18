# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 14:42:27 2021

@author: Vlad
"""
import os
os.chdir(r'E:\VladGoogleDrive\Qulab\Python_scripts')

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]='true'
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import tensorflow as tf 
from tf_quantum_simulator import operators as ops
from tf_quantum_simulator import utils
from tensorflow import complex64 as c64
from tensorflow.keras import initializers
from matplotlib import pyplot as plt
from IPython.display import clear_output
import numpy as np


class QubitRotation(tf.keras.layers.Layer):
    def __init__(self, N, batch_shape):
        super().__init__()
        self.rotation_op = ops.QubitRotationXY(tensor_with=[None, ops.identity(N)])
        self.batch_shape = batch_shape
        
    def build(self, input_shape):
        self.angle = self.add_weight(shape=self.batch_shape, trainable=True, name='angle',
                                     initializer=initializers.RandomNormal(stddev=3))
        self.phase = self.add_weight(shape=self.batch_shape, trainable=True, name='phase',
                                     initializer=initializers.RandomNormal(stddev=3))
        
    def call(self, input_state):
        R = self.rotation_op(self.angle, self.phase)
        output_state = tf.linalg.matvec(R, input_state)
        return output_state


class ConditionalDisplacement(tf.keras.layers.Layer):
    def __init__(self, N, batch_shape):
        super().__init__()
        self.displace = ops.DisplacementOperator(N, tensor_with=[ops.identity(2), None])
        self.P = {i: utils.tensor([ops.projector(i, 2), ops.identity(N)]) for i in [0, 1]}
        self.batch_shape = batch_shape
        
    def build(self, input_shape):
        self.beta_re = self.add_weight(shape=self.batch_shape, trainable=True, name='beta_re',
                                       initializer=initializers.RandomNormal(stddev=0.5))
        self.beta_im = self.add_weight(shape=self.batch_shape, trainable=True, name='beta_im',
                                       initializer=initializers.RandomNormal(stddev=0.5))
    
    def call(self, input_state):
        beta_complex = tf.cast(self.beta_re, c64) + 1j * tf.cast(self.beta_im, c64)
        D = self.displace(beta_complex/2)
        CD = self.P[0] @ D  + self.P[1] @ tf.linalg.adjoint(D)
        output_state = tf.linalg.matvec(CD, input_state)
        return output_state


class MinInfidelity(tf.keras.metrics.Metric):
    def __init__(self, name='min_infidelity', **kwargs):
        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        min_log_infidelity = tf.math.reduce_min(log_infidelity(y_true, y_pred))
        self.min_infidelity = tf.math.exp(min_log_infidelity)
        
    def result(self):
        return self.min_infidelity



class PlotCallback(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.log = []
        self.fig, self.ax =  plt.subplots(1,1)
            
    def on_epoch_end(self, epoch, logs={}):
        vac = utils.Kronecker_product([utils.basis(0, 2, batch_shape), utils.basis(0, N, batch_shape)])
        target = utils.Kronecker_product([utils.basis(0, 2, batch_shape), utils.basis(1, N, batch_shape)])
        state = self.model.predict(vac)
        self.log.append(tf.squeeze(log_infidelity(state, target)))
        
        if epoch % 100 == 0:
            plt.cla()
            all_trajectories = np.array(self.log)
            self.ax.plot(np.arange(all_trajectories.shape[0]), all_trajectories)
            self.ax.set_ylabel('Log infidelity')
            self.ax.set_xlabel('Epoch')
            plt.tight_layout()
            plt.pause(0.05)


T = 7 # circuit depth
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


def log_infidelity(state, target):
    return tf.math.log(1-tf.math.abs(utils.batch_dot(state, target))**2)

def loss(state, target):
    return tf.math.reduce_sum(log_infidelity(state, target))

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

# track optimization progress with tensorboard
logdir = r'E:\data\gkp_sims\PPO\ECD\test2'
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)

LearningRateScheduler = tf.keras.callbacks.LearningRateScheduler(lambda t: 1e-3/(1+1e-3*t))

# compile the model and run the optimizer
ECDC.compile(optimizer=optimizer, loss=loss, metrics=[MinInfidelity()])
ECDC.fit(x=vac, y=target, epochs=10000, callbacks=[tensorboard_callback, PlotCallback()])