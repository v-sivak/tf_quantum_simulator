# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 11:23:47 2021

@author: Vladimir Sivak
"""
import tensorflow as tf
from tf_quantum_simulator import utils


class MinInfidelity(tf.keras.metrics.Metric):
    def __init__(self, name='min_infidelity', **kwargs):
        super().__init__(name=name, **kwargs)

    def update_state(self, y_true, y_pred, sample_weight=None):
        min_log_infidelity = tf.math.reduce_min(utils.log_infidelity(y_true, y_pred))
        self.min_infidelity = tf.math.exp(min_log_infidelity)
        
    def result(self):
        return self.min_infidelity