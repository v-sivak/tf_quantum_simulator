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

    def _step(self, j, psi, steps):
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

        # Check if all probs sum to 1
        all_traj_check = tf.math.reduce_all(check)
        accept = lambda: [j + 1, state, steps]
        repeat = lambda: [j, psi, steps]
        return tf.cond(all_traj_check, accept, repeat)

    def _cond(self, j, psi, steps):
        return tf.less(j, steps)

    def run(self, psi, steps):
        """
        Simulate a batch of trajectories for a number of steps.
        
        Args:
            psi (Tensor([B1,...Bb,N], c64)): batch of quantum states.
            steps (int): number of steps to run the trajectory
        """
        
        psi, _ = normalize(psi)
        j = tf.constant(0)
        _, psi_new, _= tf.while_loop(
            self._cond, self._step, loop_vars=[j, psi, steps]
        )
        
        # Check for NaN
        mask = tf.math.is_nan(tf.math.real(psi_new))
        psi_new = tf.where(mask, psi, psi_new)
        return psi_new
