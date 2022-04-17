# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 13:33:51 2021
"""
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import complex64 as c64
from math import pi, sqrt

import operators as ops
from utils import tensor, measurement, normalize
from mc_simulator.quantum_trajectory_sim import QuantumTrajectorySim

# TODO: implement the following things
    # More realistic simulation of readout with phase updates in SBS
    # Keep track of duration of different operations (running time variable?)
    # Add option to change troterization order in ecdc 
    # Why div 2 in dephasing jump operator?
    # add also Kerr back-action

class Simulator():
    """ Monte Carlo simulator of the GKP error correction experiment. 
        The quantum states are represented in photon number basis in the 
        combined Hilbert space of oscillator and qubit. 
    
    """
    def __init__(self, N):
        """
        Args:
            N (int): oscillator Hilbert space truncation in photon number basis
            
        """
        self.N = N
        self.create_operators()
        
        # units: seconds, Hz
        self.K = -58
        self.chi = 100e3
        self.chi_prime = 96
        self.T1_qb = 110e-6
        self.T2_qb = 100e-6
        self.T1_osc = 550e-6
        self.T2_osc = 920e-6
        self.t_read = 1.5e-6
        self.t_phase_update = 700e-9
        self.t_sbs_b = 540e-9
        self.t_sbs_s = 200e-9
        self.t_idle = 9e-6
        self.dt = 100e-9
        
        self.p_qnd_e = 0.954  # prob of process e -> e
        self.p_qnd_g = 0.9996 # prob of process g -> g
        self.p_snr_g = 0.998 # prob to get outcome 'g' in process g -> g
        self.p_snr_e = 0.998 # prob to get outcome 'e' in process e -> e
        self.p_dd = 0.37 # demolition detection probability


        # Initialize quantum trajectories simulator
        self.mcsim = QuantumTrajectorySim(self._kraus_ops())
        def simulate_quantum_jumps(state, time):
            steps = tf.cast(time / self.dt, dtype=tf.int32) # FIXME: rounding
            return self.mcsim.run(state, steps)
        # wrapping in tf.function allows to speed this up by ~ x2-3
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
        self.displace = ops.DisplacementOperator(N, tensor_with=tensor_with)
        self.rotate = ops.RotationOperator(N, tensor_with=tensor_with)
        
        # qubit parameterized operators
        tensor_with = [None, ops.identity(N)]
        self.rotate_qb_xy = ops.QubitRotationXY(tensor_with=tensor_with)
        self.rotate_qb_z = ops.QubitRotationZ(tensor_with=tensor_with)

        # qubit sigma_z measurement projector
        self.P = {i: tensor([ops.projector(i, 2), ops.identity(N)]) 
                  for i in [0, 1]}
        

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
        gamma_phi_qb = 1/self.T2_qb - 1/(2*self.T1_qb)
        qubit_pure_dephasing = sqrt(gamma_phi_qb/2) * self.sz 
        gamma_phi_osc = 1/self.T2_osc - 1/(2*self.T1_osc)
        cavity_pure_dephasing = sqrt(gamma_phi_osc/2) * self.n
        return [photon_loss, qubit_decay, qubit_pure_dephasing]#, cavity_pure_dephasing]
    

    def _kraus_ops(self):
        """
        Create Kraus operators for the free evolution simulator.
        """
        Kraus = {}
        Kraus[0] = self.I - 1j * self._hamiltonian * self.dt
        for i, c in enumerate(self._collapse_operators):
            Kraus[i + 1] = sqrt(self.dt) * c
            Kraus[0] -= 1/2 * tf.linalg.matmul(c, c, adjoint_a=True) * self.dt
        return Kraus


    def ideal_ecdc_sequence(self, state, beta, angle, phase):
        """
        Args:
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
            
            # echo pulse inside ECD gate; last step doesn't have it
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
                if sample==True, otherwise replicate the input state
            z (Tensor([B1, ..., Bb, 1], f32)): batch of measurement outcomes 
                if sample==True, otherwise batch of sigma_z expectation values
        """
        D = self.displace(beta/2.0)
        CD = self.ctrl(D, tf.linalg.adjoint(D))
        
        Y90p = self.rotate_qb_xy(tf.constant(pi/2), tf.constant(pi/2))
        Y90m = self.rotate_qb_xy(-tf.constant(pi/2), tf.constant(pi/2))
        
        state = tf.linalg.matvec(Y90p, state)
        state = tf.linalg.matvec(CD, state)
        state = tf.linalg.matvec(Y90m, state)
        return measurement(state, self.P, sample)


    def ecdc_sequence(self, state, beta, angle, phase, tau=None):
        """
            state (Tensor([B1, ..., Bb, 2N], c64)): batched quantum state 
            beta, angle, phase (Tensor([T,1], c64)): params of ECDC sequence
            tau (Tensor([T,1], f32)): durations of the ECDC blocks in seconds
        """
        T = len(angle)
        
        if tau == None: 
            return self.ideal_ecdc_sequence(state, beta, angle, phase)
        
        for t in range(T):
            # qubit rotation
            R = self.rotate_qb_xy(angle[t], phase[t])
            state = tf.linalg.matvec(R, state)
            
            # conditional displacement
            D = self.displace(beta[t]/4.0)
            CD = self.ctrl(D, tf.linalg.adjoint(D))
            
            state = tf.linalg.matvec(CD, state)
            state = self.simulate_quantum_jumps(state, tau[t])
            state = tf.linalg.matvec(CD, state)
            
            # echo pulse inside ECD gate; last step doesn't have it
            if t < T-1:
                state = tf.linalg.matvec(self.sx, state)
        
        state, _ = normalize(state)
        return state
    

    def sbs(self, state, Delta, quad):
        """
        small-big-small protocol for GKP error correction. See this paper:
        https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.125.260509
        
        Args:
            state (Tensor([B1, ..., Bb, 2N], c64)): batched quantum state 
            Delta (float): effective squeezing parameter
            quad (str): quadrature for this round, either 'x' or 'p'
        """
        # Apply SBS step with trotterized error channels as in ECDC method
        eps = Delta ** 2 * sqrt(2*pi)
        beta  = tf.cast([0.5j*eps, sqrt(2*pi)+0j, 0.5j*eps, 0j], c64)
        beta *= (1 if quad == 'x' else 1j)
        angle = tf.cast([pi/2, -pi/2, pi/2, pi/2], c64)
        phase = tf.cast([pi/2, 0, 0, -pi/2], c64)
        tau = tf.cast([self.t_sbs_s, self.t_sbs_b, self.t_sbs_s, 0], tf.float32)
        state = self.ecdc_sequence(state, beta, angle, phase, tau)
        
        # Ideal qubit measurement. Non-idealities included next
        state, m_i = measurement(state, self.P, sample=True)
        
        # probability of qubit final state 'e' after the measurement 
        p_e = tf.where(m_i==-1, self.p_qnd_e, 1-self.p_qnd_g)
        # sample final qubit state after the measurement (due to non-QNDness)
        m_f = 1-2*tfp.distributions.Bernoulli(probs=p_e, dtype=tf.float32).sample()
        
        # non-QND back-action on the qubit
        state = tf.where(m_i == m_f, state, tf.linalg.matvec(self.sx, state))
        
        # rotation back-action on the oscillator
        delta_angle = (2*pi) * self.chi * self.t_read / 2
        # sample the fraction "r" of readout time when the demolition occured
        r = tf.random.uniform(m_i.shape, 0.0, 1.0) 
        angle = delta_angle * tf.where(m_i == m_f, m_i, m_i*r+m_f*(1-r))
        state = tf.linalg.matvec(self.rotate(angle), state)
        
        # sample measurement outcomes from {+1,-1} that controller will declare
        mask = tf.where(m_i == m_f, 1.0, 0.0)
        p_me = tf.where(m_i==-1, self.p_snr_e, 1-self.p_snr_g)
        s = tfp.distributions.Bernoulli(probs=p_me, dtype=tf.float32).sample()
        m = mask * (1-2*s) + (1-mask) * tf.where(r<self.p_dd, m_f, m_i)
        
        # Feedback pi-pulse based on the received measurement outcome
        state = tf.where(m==1, state, tf.linalg.matvec(self.sx, state))
        
        # Feedback phase update, equivalent to rotation
        state = tf.linalg.matvec(self.rotate(-m*delta_angle), state)
        
        free_time = self.t_idle + self.t_phase_update
        state = self.simulate_quantum_jumps(state, free_time)
        return state
        

