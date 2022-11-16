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


class Simulator():
    """ Monte Carlo simulator of the GKP error correction experiment. 
        The quantum states are represented in photon number basis in the 
        combined Hilbert space of oscillator and qubit. 
    
    """
    def __init__(self, N, params=None):
        """
        Args:
            N (int): oscillator Hilbert space truncation in photon number basis
            
        """
        self.N = N
        self.create_operators()
        
        if params is None:
            # units: [seconds], [Hz]
            self.K = -4.8
            self.chi = 46.5e3
            self.chi_prime = 5.8
            
            self.n_th = 0.05
            
            self.T1_qb = 258e-6
            self.T2_qb = 147e-6
            self.T1_osc = 622e-6
            self.T2_osc = 923e-6
            self.gamma_phi_osc = 0
            
            self.t_read = 2.4e-6
            self.t_VR = 448e-9 # virtual rotation gate
            self.t_idle = 500e-9 # idle section
            
            self.t_sbs = [232e-9, 906e-9, 308e-9, (24+78+24)*1e-9]
            
            self.dt = 100e-9 # time discretization step or MC trajectories
            
            self.p_qnd_e = 0.990  # prob of process e -> e
            self.p_qnd_g = 0.9996 # prob of process g -> g
            self.p_dd = 0.16 # demolition detection probability
        else:
            for (p,v) in params.items():
                self.__setattr__(p, v)
        
        self.H_evol = ops.HamiltonianEvolutionOperator(self.hamiltonian)

        # 1st Monte Carlo sim: includes all system error channels.
        # This will be used during trotterized CD gates
        c_ops = [op for (op_name, op) in self.collapse_ops.items()]
        kraus_ops = self.kraus_ops(0, c_ops)
        self.mcsim = QuantumTrajectorySim(kraus_ops)
        def mcsim_wrapper(state, time):
            steps = tf.cast(time / self.dt, dtype=tf.int32) # FIXME: rounding
            return self.mcsim.run(state, steps)
        # wrapping in tf.function allows to speed this up by ~ x2-3
        self.simulate_quantum_jumps = tf.function(mcsim_wrapper)
    
        # 2nd Monte Carlo sim: only photon loss (use for readout, since
        # qubit errors during readout are modeled separately)
        c_ops = [self.collapse_ops['cavity_photon_loss'], self.collapse_ops['cavity_dephasing']]
        kraus_ops = self.kraus_ops(0, c_ops)
        self.mcsim_2 = QuantumTrajectorySim(kraus_ops)
        def mcsim_wrapper_2(state, time):
            steps = tf.cast(time / self.dt, dtype=tf.int32) # FIXME: rounding
            return self.mcsim_2.run(state, steps)
        self.simulate_photon_loss = tf.function(mcsim_wrapper_2)



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
        self.sp = tensor([ops.sigma_p(), ops.identity(N)])
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
    def hamiltonian(self):
        chi = 1/2 * self.chi * self.ctrl(self.n, -self.n)
        chi_prime = 1/4 * self.chi_prime * self.ctrl(self.n**2, -self.n**2)
        kerr = 1/2 * self.K * self.n**2
        return kerr + chi_prime + chi


    @property
    def collapse_ops(self):
        gamma_phi_qb = max(1/self.T2_qb - 1/(2*self.T1_qb), 0)
        gamma_up_qb = self.n_th * 1/self.T1_qb
        gamma_down_qb = (1-self.n_th) * 1/self.T1_qb
        collapse_operators = {
            'cavity_photon_loss' : sqrt(1/self.T1_osc) * self.a,
            'cavity_dephasing' : sqrt(2*pi*self.gamma_phi_osc) * self.n,
            'qubit_decay' : sqrt(gamma_down_qb) * self.sm,
            'qubit_excitation' : sqrt(gamma_up_qb) * self.sp,
            'qubit_dephasing' : sqrt(0.5*gamma_phi_qb) * self.sz
            }
        return collapse_operators


    def kraus_ops(self, hamiltonian, collapse_operators):
        """
        Create Kraus operators for the free evolution simulator.
        """
        Kraus = {}
        Kraus[0] = self.I - 1j * hamiltonian * self.dt
        for i, c in enumerate(collapse_operators):
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


    def ecdc_sequence(self, state, beta, angle, phase, tau=None, trotter=None):
        """
            state (Tensor([B1, ..., Bb, 2N], c64)): batched quantum state 
            beta, angle, phase (Tensor([T,1], c64)): params of ECDC sequence
            tau (Tensor([T,1], f32)): durations of the ECDC blocks in seconds
        """
        T = len(angle)

        if tau is None: 
            return self.ideal_ecdc_sequence(state, beta, angle, phase)

        for t in range(T):
            # qubit rotation
            R = self.rotate_qb_xy(angle[t], phase[t])
            state = tf.linalg.matvec(R, state)
            
            K = 2 if trotter is None else trotter[t]
            
            # conditional displacement
            D = self.displace(beta[t]/2.0/K)
            CD = self.ctrl(D, tf.linalg.adjoint(D))

            state = tf.linalg.matvec(CD, state)
            for k in range(K-1):
                state = self.simulate_quantum_jumps(state, tau[t]/(K-1))
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
        # beta  = tf.cast([0.13j, sqrt(2*pi)+0j, 0.24j, 0j], c64)
        beta *= (1 if quad == 'x' else 1j)
        angle = tf.cast([pi/2, -pi/2, pi/2, pi/2], c64)
        phase = tf.cast([pi/2, 0, 0, -pi/2], c64)
        tau = tf.cast(self.t_sbs, tf.float32)
        trotter = [2, 10, 2, 2]
        state = self.ecdc_sequence(state, beta, angle, phase, tau, trotter)

        # Ideal qubit measurement. Non-idealities included next
        state, m_i = measurement(state, self.P, sample=True)

        # For each trajectory, depending on the qubit state, what is the prob.
        # that the measurement will not demolish this state? 
        P_qnd = tf.where(m_i==-1, self.p_qnd_e, self.p_qnd_g)
        # sample demolition events (qnd=1 if there was no demolition)
        qnd = tfp.distributions.Bernoulli(probs=P_qnd, dtype=tf.int32).sample()
        # sample the moment "r" during reset time when the demolition occured
        r = tf.random.uniform(m_i.shape, 0.0, 1.0)
        
        # simulate Hamiltonian time evolution until demolition
        state = tf.linalg.matvec(self.H_evol(r * self.t_read), state)
        
        # demolition event
        state = tf.where(qnd==1, state, tf.linalg.matvec(self.sx, state))
       
        # simulate Hamiltonian time evolution after demolition
        state = tf.linalg.matvec(self.H_evol((1-r) * self.t_read), state)
        
        # Sample measurement outcome from {+1,-1} that controller declares.
        # It will depend on when the demolition happened: if it was in the 
        # first p_dd fraction of reset time, then it will be detected, else
        # it will not be detected and controller will declare a state that is
        # different from the actual qubit state        
        m = tf.where(qnd==1, m_i, tf.where(r<self.p_dd, -m_i, m_i))
        # Feedback pi-pulse based on the received measurement outcome
        state = tf.where(m==1, state, tf.linalg.matvec(self.sx, state))

        # Feedback virtual rotation of the oscillator
        angle = - (2*pi) * self.chi / 2 * self.t_read * m
        state = tf.linalg.matvec(self.rotate(angle), state)

        # Now simulate dephasing due to qubit jumps during t_idle and t_VR.
        # It's exact same sequence of operations, so no detailed comments.

        # To find qubit state after reset, pretent that there was a new
        # measurement. This one will produce deterministic outcome though.
        state, m_i = measurement(state, self.P, sample=True)

        # Find probability p_e of being in |e> state after time t_idle + t_VR 
        # given the initial qubit state and up/down jump probabilities.
        p_up = (self.t_idle + self.t_VR) / self.T1_qb * self.n_th
        p_down = (self.t_idle + self.t_VR) / self.T1_qb * (1-self.n_th)
        P_qnd = tf.where(m_i==-1, 1-p_down, 1-p_up)
        qnd = tfp.distributions.Bernoulli(probs=P_qnd, dtype=tf.int32).sample()
        r = tf.random.uniform(m_i.shape, 0.0, 1.0)
        time = r * (self.t_idle + self.t_VR)
        state = tf.linalg.matvec(self.H_evol(time), state)
        state = tf.where(qnd==1, state, tf.linalg.matvec(self.sx, state))
        time = (1-r) * (self.t_idle + self.t_VR)
        state = tf.linalg.matvec(self.H_evol(time), state)

        # This is "second part" of feedback virtual rotation of the oscillator
        # Controller assumes that the qubit was perfectly reset and sits in "g"
        # In experiment VR is done in a single step, but this is just convenient
        # for a simulation. Rotations commute so this is okay.
        angle = - (2*pi) * self.chi / 2 * (self.t_idle + self.t_VR) * 1
        state = tf.linalg.matvec(self.rotate(angle), state)

        # Sample photon loss events during readout, idle, and VR.
        # Effect of qubit errors  was already included
        t_tot = self.t_read + self.t_idle + self.t_VR
        state = self.simulate_photon_loss(state, t_tot)

        return state