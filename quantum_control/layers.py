# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 11:22:21 2021

@author: Vladimir Sivak
"""
import tensorflow as tf
from tensorflow.keras import initializers
from tensorflow import complex64 as c64
from tf_quantum_simulator import operators as ops
from tf_quantum_simulator import utils


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