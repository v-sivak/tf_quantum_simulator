# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:58:30 2020

@author: Vladimir Sivak
"""
import tensorflow as tf
import tensorflow_probability as tfp
from utils import normalize


class QuantumTrajectorySim:
    """
    Tensorflow implementation of the Monte Carlo quantum trajectory simulator.
    """

    def __init__(self, Kraus_operators):
        """
        Args:
            Kraus_operators (dict: Tensor(c64)): dictionary of Kraus operators. 
                By convention, K[0] is no-jump operator, K[i>0] are jump operators.
        """
        self.Kraus_operators = Kraus_operators

    def _step(self, j, psi, steps, save_frequency):
        """ 
        A single trajectory step: calculate probabilities associated to
        different Kraus ops, and sample Kraus ops from this distribution.
        
        """
        batch_shape = psi.shape[:-1]
        cumulant = tf.zeros(batch_shape+[1])
        prob = tf.random.uniform(batch_shape+[1])
        check = tf.zeros(batch_shape+[1], dtype=bool) # check if all probs sum to 1
        state = psi
        for i, Kraus in self.Kraus_operators.items():
            # Compute a trajectory for this Kraus operator
            traj = tf.linalg.matvec(Kraus, psi)  # shape = [B1,...Bb,N]
            traj, norm = normalize(traj)
            # Probability that this Kraus applies. These should sum to 1.
            p = tf.math.real(norm) ** 2
            # Select this trajectory depending on sampled 'prob'
            mask = tf.math.logical_and(prob > cumulant, prob < cumulant + p)
            check = tf.math.logical_or(mask, check)
            state = tf.where(mask, traj, state)
            cumulant += p
        if tf.math.reduce_any(tf.math.logical_not(check)):
            raise Exception('Probabilities not summing to 1.')
        if save_frequency > 0 and tf.math.floormod(j, save_frequency) == 0:
            self.psi_history.append(state)
        return [j + 1, state, steps, save_frequency]

    def _cond(self, j, psi, steps, save_frequency):
        return tf.less(j, steps)

    def run(self, psi, steps, save_frequency=0):
        """
        Simulate a batch of trajectories for a number of steps.
        
        Args:
            psi (Tensor([B1,...Bb,N], c64)): batch of quantum states.
            steps (int): number of steps to run the trajectory
        """
        if save_frequency > 0: # FIXME: pass save_frequency on initialization
            self.psi_history = []
        
        psi, _ = normalize(psi)
        j = tf.constant(0)
        _, psi_new, _, _ = tf.while_loop(
            self._cond, self._step, loop_vars=[j, psi, steps, save_frequency]
        )

        if save_frequency > 0:
            return tf.stack(self.psi_history) # FIXME: tf.stack is super slow
        
        # Check for NaN
        mask = tf.math.is_nan(tf.math.real(psi_new))
        psi_new = tf.where(mask, psi, psi_new)
        return psi_new
