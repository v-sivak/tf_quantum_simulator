# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 11:24:15 2021

@author: Vladimir Sivak
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import utils


class PlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        super().__init__()
    
    def on_train_begin(self, logs={}):
        self.log = []
        self.fig, self.ax =  plt.subplots(1,1)
            
    def on_epoch_end(self, epoch, logs={}):
        states = self.model.predict(self.inputs)
        
        infidelity = tf.math.exp(utils.log_infidelity(self.targets, states))
        # average the infidelity over all input states.
        avg_state_infidelity = tf.reduce_mean(infidelity, axis=0)
        avg_state_log_infidelity = tf.math.log(avg_state_infidelity)
        self.log.append(tf.squeeze(avg_state_log_infidelity))
        
        if epoch % 100 == 0:
            plt.cla()
            all_trajectories = np.array(self.log)
            self.ax.plot(np.arange(all_trajectories.shape[0]), all_trajectories)
            self.ax.set_ylabel('Log infidelity')
            self.ax.set_xlabel('Epoch')
            plt.tight_layout()
            plt.pause(0.05)
            

class MinInfidelity(tf.keras.callbacks.Callback):
    def __init__(self, inputs, targets):
        self.inputs = inputs
        self.targets = targets
        super().__init__()
    
    def on_train_begin(self, logs={}):
        self.infidelities = []
        self.epochs = []
        self.index = []
            
    def on_epoch_end(self, epoch, logs={}):
        
        if epoch % 10 == 0:
            states = self.model.predict(self.inputs)
            
            infidelity = tf.math.exp(utils.log_infidelity(self.targets, states))
            # average the infidelity over all input states.
            avg_state_infidelity = tf.reduce_mean(infidelity, axis=0)
            min_infidelity = float(tf.math.reduce_min(avg_state_infidelity))
            ind = int(tf.argmin(avg_state_infidelity, axis=0))
        
            self.infidelities.append(min_infidelity)
            self.epochs.append(epoch)
            self.index.append(ind)
            print('Min infidelity: %.4f' % min_infidelity)