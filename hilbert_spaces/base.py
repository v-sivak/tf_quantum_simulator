"""
Abstract base class to define a specific simulator Hilbert space

Created on Sun Jul 26 19:39:18 2020

@author: Henry Liu
"""
from abc import ABC, abstractmethod

import tensorflow as tf
import tensorflow_probability as tfp
from tf_quantum_simulator.quantum_trajectory_sim import QuantumTrajectorySim
import numpy as np


class HilbertSpace(ABC):
    """
    Abstract base class which intializes a Monte Carlo simulator for a particular
    Hilbert space. The space is defined by the subclass, which implements a set of
    operators on the space, a Hamiltonian, and a set of jump operators.
    """

    def __init__(self, *args, discrete_step_duration, **kwargs):
        """
        Args:
            discrete_step_duration (float): Simulator time discretization in seconds.
        """
        self.discrete_step_duration = discrete_step_duration
        # Tensor ops acting on oscillator Hilbert space
        self._define_fixed_operators()

        # Initialize quantum trajectories simulatorP
        self.mcsim = lambda *H_args: QuantumTrajectorySim(
            self._kraus_ops(discrete_step_duration, *H_args)
        )

        # returns time setps, and batch result
        # todo: could "append" to times, or have a tf variable be a running clock
        # todo: get rid of all numpy...
        def simulate(psi, time, *H_args, save_frequency=0, t_init=0):
            # TODO: fix the rounding issue
            steps = tf.cast(time / discrete_step_duration, dtype=tf.int32)
            result = self.mcsim(*H_args).run(psi, steps, save_frequency)
            if save_frequency == 0:
                t = steps.numpy() * discrete_step_duration
                return np.array([t_init + t]), result
            else:
                times = (
                    np.arange(result.shape[0]) * save_frequency * discrete_step_duration
                )
                return times, result

        # todo: make this work:
        # self.simulate = tf.function(simulate)
        self.simulate = simulate

        super().__init__()

    @abstractmethod
    def _define_fixed_operators(self):
        """
        Create operators on this Hilbert space. To be defined by the subclass.
        Example:
            self.I = operators.identity(self.N)
            self.p = operators.momentum(self.N)
            self.displace = operators.DisplacementOperator(self.N)
        """
        pass

    @abstractmethod
    def _hamiltonian(self, *H_args):
        """
        System Hamiltonian (Tensor(c64)). To be defined by the subclass.
        """
        pass

    @property
    @abstractmethod
    def _collapse_operators(self):
        """
        List of collapse operators (Tensor(c64)). To be defined by the subclass.
        """
        pass

    def _kraus_ops(self, dt, *H_args):
        """
        Create kraus ops for free evolution simulator

        Args:
            dt (float): Discretized time step of simulator
            H_args: arguments passed to Hamiltonian function
        """
        Kraus = {}
        # Kraus[0] = self.I - 1j * self._hamiltonian(*H_args) * dt
        # note: could save this to avoid re-calculation!
        Kraus[0] = tf.linalg.expm(-1j * self._hamiltonian(*H_args) * dt)
        for i, c in enumerate(self._collapse_operators):
            Kraus[i + 1] = tf.cast(tf.sqrt(dt), dtype=tf.complex64) * c
            Kraus[0] -= 1 / 2 * tf.linalg.matmul(c, c, adjoint_a=True) * dt
        return Kraus
