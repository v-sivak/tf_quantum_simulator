# -*- coding: utf-8 -*-
"""
Created on Mon Oct  4 15:47:53 2021
@author: Vladimir Sivak
"""
import tensorflow as tf
import utils
import operators as ops
import numpy as np
from tensorflow import complex64 as c64
from math import pi, sqrt
import matplotlib.pyplot as plt
import qutip as qt

def plot_phase_space(state, tensorstate, phase_space_rep='wigner', 
                     lim=4, pts=81, title=None):
    """
    Plot phase space representation of the state. Converts a batch of states
    to density matrix.
    
    Args:
        state (tf.Tensor([B,N], c64)): batched state vector
        tensorstate (bool): flag if tensored with qubit
        phase_space_rep (str): either 'wigner' or 'CF'
        lim (float): plot limit in displacement units
        pts (int): number of pixels in each direction 
        title (str): figure title (optional)
    
    """

    assert len(state.shape)>=2 and state.shape[1] > 1
    
    # create operators    
    if tensorstate:
        N = int(state.shape[1] / 2)
        parity = utils.tensor([ops.identity(2), ops.parity(N)])
        D = ops.DisplacementOperator(N, tensor_with=[ops.identity(2), None])
    else:
        N = state.shape[1]
        D = ops.DisplacementOperator(N)
        parity = ops.parity(N)
    
    # project every trajectory onto |g> subspace
    if tensorstate:
        P0 = utils.tensor([ops.projector(0, 2), ops.identity(N)])
        state, _ = utils.normalize(tf.linalg.matvec(P0, state))
    
    # make a density matrix
    dm = utils.density_matrix(state)

    # Generate a grid of phase space points
    x = np.linspace(-lim, lim, pts)
    y = np.linspace(-lim, lim, pts)
    
    xs_mesh, ys_mesh = np.meshgrid(x, y, indexing='ij')
    grid = tf.cast(xs_mesh + 1j*ys_mesh, c64)
    grid_flat = tf.reshape(grid, [-1])
    
    matmul = tf.linalg.matmul
    
    # Calculate and plot the phase space representation
    if phase_space_rep == 'wigner':
        displaced_parity = matmul(D(grid_flat), matmul(parity, D(-grid_flat)))
        W = 1/pi * tf.linalg.trace(matmul(displaced_parity, dm))
        W_grid = tf.reshape(W, grid.shape)
    
        fig, ax = plt.subplots(1,1)
        fig.suptitle(title)
        ax.pcolormesh(x, y, np.transpose(W_grid.numpy().real), 
                      cmap='RdBu_r', vmin=-1/pi, vmax=1/pi)
        ax.set_aspect('equal')
    
    if phase_space_rep == 'CF':
        
        C = tf.linalg.trace(matmul(D(grid_flat), dm))
        C_grid = tf.reshape(C, grid.shape)
        
        fig, axes = plt.subplots(1,2, sharey=True)
        fig.suptitle(title)
        axes[0].pcolormesh(x, y, np.transpose(C_grid.numpy().real), 
                           cmap='RdBu_r', vmin=-1, vmax=1)
        axes[1].pcolormesh(x, y, np.transpose(C_grid.numpy().imag), 
                           cmap='RdBu_r', vmin=-1, vmax=1)
        axes[0].set_title('Re')
        axes[1].set_title('Im')
        axes[0].set_aspect('equal')
        axes[1].set_aspect('equal')
    
    plt.tight_layout()

        
def envelope_operator(N, Delta):
    a = qt.destroy(N)
    n_op = a.dag()*a
    G = (-Delta**2 * n_op).expm()
    G_inv = (Delta**2 * n_op).expm()
    return lambda rho: G*rho*G_inv
        

def GKP_1D_state(tensorstate, N, Delta):
    """ This function creates a gkp sensor state as tf.Tensor."""    
    Sx, Sp = qt.displace(N, sqrt(pi)), qt.displace(N, 1j*sqrt(pi))
     
    # Define Hermitian stablizers and apply the envelope
    chan = envelope_operator(N, Delta)
    Sx = chan((Sx + Sx.dag())/2)
    Sp = chan((Sp + Sp.dag())/2)
    
    # find ground state of this Hamiltonian
    state = (-Sx - Sp).groundstate()[1].unit()
    if tensorstate: state = qt.tensor(qt.basis(2, 0), state)
    dim = N if not tensorstate else 2*N 
    
    tf_state = tf.cast(state.full().reshape([1,dim]), c64)
    
    return tf_state


def GKP_code(tensorstate, N, Delta, S=np.array([[1, 0], [0, 1]]), tf_states=True):
    
    # Check if the matrix is simplectic
    Omega = np.array([[0,1],[-1,0]])
    if not np.allclose(S.T @ Omega @ S ,Omega):
        raise Exception('S is not symplectic')

    a  = qt.destroy(N)
    q_op = (a + a.dag())/sqrt(2.0)
    p_op = 1j*(a.dag() - a)/sqrt(2.0)
    
    # Define stabilizers
    Sz = (2j*sqrt(pi)*(S[0,0]*q_op + S[1,0]*p_op)).expm()
    Sx = (-2j*sqrt(pi)*(S[0,1]*q_op + S[1,1]*p_op)).expm()
    Sy = (2j*sqrt(pi)*((S[0][0]-S[0][1])*q_op + (S[1][0]-S[1][1])*p_op)).expm()
    ideal_stabilizers = {'S_x' : Sx, 'S_z' : Sz, 'S_y' : Sy}    
    
    # Define Pauli operators
    z =  (1j*sqrt(pi)*(S[0,0]*q_op + S[1,0]*p_op)).expm()
    x = (-1j*sqrt(pi)*(S[0,1]*q_op + S[1,1]*p_op)).expm()
    y = (1j*sqrt(pi)*((S[0][0]-S[0][1])*q_op + (S[1][0]-S[1][1])*p_op)).expm()
    ideal_paulis = {'X' : x, 'Y' : y, 'Z' : z}

    displacement_amplitudes = {
        'S_z': sqrt(2*pi)*(-S[1,0]+1j*S[0,0]),
        'Z'  : sqrt(pi/2)*(-S[1,0]+1j*S[0,0]),
        'S_x': sqrt(2*pi)*(S[1,1]-1j*S[0,1]),
        'X'  : sqrt(pi/2)*(S[1,1]-1j*S[0,1]),
        'S_y': sqrt(2*pi)*((S[1,1]-S[1,0])+1j*(S[0,0]-S[0,1])),
        'Y'  : sqrt(pi/2)*((S[1,1]-S[1,0])+1j*(S[0,0]-S[0,1]))
        }
    
    # Define Hermitian Paulis and stablizers
    ops = [Sz, Sx, Sy, x, y, z]
    ops = [(op + op.dag())/2.0 for op in ops]
    # pass them through the channel 
    chan = envelope_operator(N, Delta)
    ops = [chan(op) for op in ops]
    Sz, Sx, Sy, x, y, z = ops
    
    # find 'Z+' as groundstate of this Hamiltonian
    d = (- Sz - Sx - Sy - z).groundstate()
    zero = (d[1]).unit()
    one  = (x*d[1]).unit()

    states = {'+Z' : zero, 
              '-Z' : one,
              '+X' : (zero + one).unit(), 
              '-X' : (zero - one).unit(),
              '+Y' : (zero + 1j*one).unit(), 
              '-Y' : (zero - 1j*one).unit()}

    # Tensordot everything with qubit
    if tensorstate:
        for key, val in ideal_stabilizers.items():
            ideal_stabilizers[key] = qt.tensor(qt.identity(2), val)
        for key, val in ideal_paulis.items():
            ideal_paulis[key] = qt.tensor(qt.identity(2), val)
        for key, val in states.items():
            states[key] = qt.tensor(qt.basis(2,0), val)
    
    if tf_states:
        for key, val in states.items():
            states[key] = tf.cast(val.full().reshape([1,val.shape[0]]), c64)

    return ideal_stabilizers, ideal_paulis, states, displacement_amplitudes

