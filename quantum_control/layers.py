# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 11:22:21 2021

@author: Vladimir Sivak
"""
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow import complex64 as c64
import operators as ops
import utils


class QubitRotation(tf.keras.layers.Layer):
    def __init__(self, N, batch_shape, name='qubit_rotation'):
        super().__init__(name=name)
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


class OscillatorDisplacement(tf.keras.layers.Layer):
    def __init__(self, N, batch_shape, name='oscillator_displacement'):
        super().__init__(name=name)
        self.displace = ops.DisplacementOperator(N, tensor_with=[ops.identity(2), None])
        self.batch_shape = batch_shape
        
    def build(self, input_shape):
        self.alpha_re = self.add_weight(shape=self.batch_shape, trainable=True, 
            name='alpha_re', initializer=initializers.RandomNormal(stddev=0.5))
        self.alpha_im = self.add_weight(shape=self.batch_shape, trainable=True, 
            name='alpha_im', initializer=initializers.RandomNormal(stddev=0.5))
        
    def call(self, input_state):
        alpha_complex = tf.cast(self.alpha_re, c64) + 1j * tf.cast(self.alpha_im, c64)
        D = self.displace(alpha_complex)
        output_state = tf.linalg.matvec(D, input_state)
        return output_state


class ConditionalDisplacement(tf.keras.layers.Layer):
    def __init__(self, N, batch_shape, echo_pulse=True, name='conditional_displacement'):
        super().__init__(name=name)
        self.displace = ops.DisplacementOperator(N, tensor_with=[ops.identity(2), None])
        self.P = {i: utils.tensor([ops.projector(i, 2), ops.identity(N)]) for i in [0, 1]}
        self.batch_shape = batch_shape
        self.qubit_op = utils.tensor([ops.sigma_x(), ops.identity(N)]) if echo_pulse else ops.identity(2*N)
        
    def build(self, input_shape):
        self.beta_re = self.add_weight(shape=self.batch_shape, trainable=True, name='beta_re',
                                       initializer=initializers.RandomNormal(stddev=0.5))
        self.beta_im = self.add_weight(shape=self.batch_shape, trainable=True, name='beta_im',
                                       initializer=initializers.RandomNormal(stddev=0.5))
    
    def call(self, input_state):
        beta_complex = tf.cast(self.beta_re, c64) + 1j * tf.cast(self.beta_im, c64)
        D = self.displace(beta_complex/2)
        CD = self.P[0] @ D  + self.P[1] @ tf.linalg.adjoint(D)
        state = tf.linalg.matvec(CD, input_state)
        output_state = tf.linalg.matvec(self.qubit_op, state)
        return output_state