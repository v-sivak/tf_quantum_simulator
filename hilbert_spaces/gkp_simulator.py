# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:33:51 2021
"""
import tensorflow as tf
from math import pi, sqrt

import operators as ops
from utils import tensor, measurement, normalize
from quantum_trajectory_sim import QuantumTrajectorySim

class Simulator():
    """ Simulator of oscillator-qubit Hilbert space. """
    def __init__(self, N):
        """
        Args:
            N (int): oscillator Hilbert space truncation in the number basis
            
        """
        self.N = N
        
        # units: seconds, Hz
        self.K = -58
        self.chi_prime = 96
        self.T1_qb = 110e-6
        self.T2_qb = 100e-6
        self.T1_osc = 590e-6
        self.dt = 100e-9
        
        self.create_operators()
 
        # Initialize quantum trajectories simulator
        self.mcsim = QuantumTrajectorySim(self._kraus_ops(self.dt))
        def simulate_quantum_jumps(state, time):
            steps = tf.cast(time / self.dt, dtype=tf.int32) # FIXME: rounding
            return self.mcsim.run(state, steps)
        # wrapping in tf.function allows to speed this up by ~ 2-3
        self.simulate_quantum_jumps = tf.function(simulate_quantum_jumps)
    
    def create_operators(self):
        N = self.N
        
        # oscillator fixed operators
        self.I = tensor([ops.identity(2), ops.identity(N)])
        self.a = tensor([ops.identity(2), ops.destroy(N)])
        self.a_dag = tensor([ops.identity(2), ops.create(N)])
        self.q = tensor([ops.identity(2), ops.position(N)])
        self.p = tensor([ops.identity(2), ops.momentum(N)])
        self.n = tensor([ops.identity(2), ops.num(N)])
        self.parity = tensor([ops.identity(2), ops.parity(N)])

        # qubit fixed operators
        self.sx = tensor([ops.sigma_x(), ops.identity(N)])
        self.sy = tensor([ops.sigma_y(), ops.identity(N)])
        self.sz = tensor([ops.sigma_z(), ops.identity(N)])
        self.sm = tensor([ops.sigma_m(), ops.identity(N)])
        self.H = tensor([ops.hadamard(), ops.identity(N)])

        # oscillator parameterized operators
        tensor_with = [ops.identity(2), None]
        self.phase = ops.Phase()
        self.displace = ops.DisplacementOperator(N, tensor_with=tensor_with)
        self.rotate = ops.RotationOperator(N, tensor_with=tensor_with)
        
        # qubit parameterized operators
        tensor_with = [None, ops.identity(N)]
        self.rotate_qb_xy = ops.QubitRotationXY(tensor_with=tensor_with)
        self.rotate_qb_z = ops.QubitRotationZ(tensor_with=tensor_with)

        # qubit sigma_z measurement projector
        self.P = {i: tensor([ops.projector(i, 2), ops.identity(N)]) for i in [0, 1]}
        

    @tf.function
    def ctrl(self, U0, U1):
        """
        Controlled-U gate.  Apply 'U0' if qubit is '0', and 'U1' if qubit is '1'.

        Args:
            U0, U1 (Tensor([B1, ..., Bb, 2N, 2N], c64)): unitaries that only
                act on the oscillator subspace, but defined on the full Hilbert
                space. 

        """
        return self.P[0] @ U0 + self.P[1] @ U1

    @property
    def _hamiltonian(self):
        chi_prime = 1/4 * (2*pi) * self.chi_prime * self.ctrl(self.n**2, -self.n**2)
        kerr = 1/2 * (2*pi) * self.K * self.n**2
        
        return kerr + chi_prime

    @property
    def _collapse_operators(self):
        photon_loss = sqrt(1/self.T1_osc) * self.a
        qubit_decay = sqrt(1/self.T1_qb) * self.sm
        
        T_phi = 1/(1/self.T2_qb - 1/(2*self.T1_qb))
        qubit_pure_dephasing = sqrt(1/(2*T_phi)) * self.sz

        return [photon_loss, qubit_decay, qubit_pure_dephasing]
    
    def _kraus_ops(self, dt):
        """
        Create Kraus ops for the free evolution simulator.

        Args:
            dt (float): discretized time step of the simulator
        """
        Kraus = {}
        Kraus[0] = self.I - 1j * self._hamiltonian * dt
        for i, c in enumerate(self._collapse_operators):
            Kraus[i + 1] = sqrt(dt) * c
            Kraus[0] -= 1/2 * tf.linalg.matmul(c, c, adjoint_a=True) * dt
        return Kraus


    def ideal_ecdc_sequence(self, state, beta, angle, phase):
        """
            state (Tensor([B1, ..., Bb, 2N], c64)): batched quantum state 
            beta, angle, phase (Tensor([T,1], c64)): params of ECDC sequence
        """
        T = len(angle)
        
        for t in range(T):
            # Construct gates
            D = self.displace(beta[t]/2.0)
            CD = self.ctrl(D, tf.linalg.adjoint(D))
            R = self.rotate_qb_xy(angle[t], phase[t])
            
            # Apply gates
            state = tf.linalg.matvec(R, state)
            state = tf.linalg.matvec(CD, state)
            
            if t < T-1:
                state = tf.linalg.matvec(self.sx, state)
        
        state, _ = normalize(state)
        
        return state


    def ecdc_sequence(self, state, beta, angle, phase):
        """
            state (Tensor([B1, ..., Bb, 2N], c64)): batched quantum state 
            beta, angle, phase (Tensor([T,1], c64)): params of ECDC sequence
        """
        T = len(angle)
        
        for t in range(T):
            # qubit rotation
            R = self.rotate_qb_xy(angle[t], phase[t])
            state = tf.linalg.matvec(R, state)
            
            # conditional displacement
            D = self.displace(beta[t]/4.0)
            CD = self.ctrl(D, tf.linalg.adjoint(D))
            
            state = tf.linalg.matvec(CD, state)
            state = self.simulate_quantum_jumps(state, 100e-9)
            state = tf.linalg.matvec(CD, state)
            
            if t < T-1:
                state = tf.linalg.matvec(self.sx, state)
        
        state, _ = normalize(state)
    
        return state
    

    def ideal_phase_estimation(self, state, beta, sample=False):
        """
        One round of ideal phase estimation.

        Args:
            state (Tensor([B1, ..., Bb, 2N], c64)): batched quantum state 
            beta (float): displacement amplitude
            sample (bool): flag to sample or take expectation value

        Returns:
            state (Tensor([B1, ..., Bb, 2N], c64)): batch of collapsed states 
                if sample==True, otherwise same as input state
            z (Tensor([B1, ..., Bb, 1], f32)): batch of measurement outcomes 
                if sample==True, otherwise batch of sigma_z expectation values
        """
        D = self.displace(beta/2.0)
        CD = self.ctrl(D, tf.linalg.adjoint(D))
        
        Ry_plus_pi2 = self.rotate_qb_xy(tf.constant(pi/2), tf.constant(pi/2))
        Ry_minus_pi2 = self.rotate_qb_xy(-tf.constant(pi/2), tf.constant(pi/2))
        
        state = tf.linalg.matvec(Ry_plus_pi2, state)
        state = tf.linalg.matvec(CD, state)
        state = tf.linalg.matvec(Ry_minus_pi2, state)

        return measurement(state, self.P, sample)