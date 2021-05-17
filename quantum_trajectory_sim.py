# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 14:58:30 2020

@author: Vladimir Sivak
"""
import tensorflow as tf
import tensorflow_probability as tfp
from tf_quantum_simulator.utils import normalize


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
        One step in the Markov chain.
        """
        traj, p, norm = {}, {}, {}
        cumulant = tf.zeros([psi.shape[0], 1])
        prob = tf.random.uniform([psi.shape[0], 1])
        state = psi
        masks = []
        for i, Kraus in self.Kraus_operators.items():
            # Compute a trajectory for this Kraus operator
            traj[i] = tf.linalg.matvec(Kraus, psi)  # shape = [b,N]
            traj[i], norm[i] = normalize(traj[i])
            p[i] = tf.math.real(norm[i]) ** 2
            # Select this trajectory depending on sampled 'prob'
            # need to understand why this is failing sometimes....
            mask = tf.math.logical_and(prob > cumulant, prob < cumulant + p[i])
            masks.append(mask)
            state = tf.where(mask, traj[i], state)
            # Update cumulant
            cumulant += p[i]
        traj_not_taken = tf.logical_not(tf.reduce_any(tf.stack(masks), axis=0))
        """
        num_not_taken = tf.reduce_sum(tf.cast(traj_not_taken, tf.float32)).numpy()
        if num_not_taken > 0:
            print(
                "step %d num trajectories where krauss not taken: %d"
                % (j, num_not_taken)
            )
        """
        # take K0 anywhere where, for some reason, a trajectory was not taken.
        state = tf.where(traj_not_taken, traj[0], state)
        if save_frequency > 0 and j % save_frequency == 0:
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
        if save_frequency > 0:
            self.psi_history = []
        psi, _ = normalize(psi)
        j = tf.constant(0)
        _, psi_new, _, _ = tf.while_loop(
            self._cond, self._step, loop_vars=[j, psi, steps, save_frequency]
        )

        if save_frequency > 0:
            return tf.stack(self.psi_history)
        # Check for NaN
        mask = tf.math.is_nan(tf.math.real(psi_new))
        psi_new = tf.where(mask, psi, psi_new)
        return psi_new
