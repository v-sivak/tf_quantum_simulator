# -*- coding: utf-8 -*-
"""
Created on Wed Sep 29 11:24:15 2021

@author: Vladimir Sivak
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tf_quantum_simulator import utils


class PlotCallback(tf.keras.callbacks.Callback):
    def __init__(self, vac, target):
        self.vac = vac
        self.target = target
        super().__init__()
    
    def on_train_begin(self, logs={}):
        self.log = []
        self.fig, self.ax =  plt.subplots(1,1)
            
    def on_epoch_end(self, epoch, logs={}):
        state = self.model.predict(self.vac)
        self.log.append(tf.squeeze(utils.log_infidelity(state, self.target)))
        
        if epoch % 100 == 0:
            plt.cla()
            all_trajectories = np.array(self.log)
            self.ax.plot(np.arange(all_trajectories.shape[0]), all_trajectories)
            self.ax.set_ylabel('Log infidelity')
            self.ax.set_xlabel('Epoch')
            plt.tight_layout()
            plt.pause(0.05)