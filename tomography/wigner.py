#%%
import tensorflow as tf
from tensorflow import complex64 as c64
from math import pi
import numpy as np

from scipy.special import genlaguerre

try:
    from scipy.misc import factorial
except:
    from scipy.special import factorial

DTYPE = np.complex64


matmul = tf.linalg.matmul
real = tf.math.real
imag = tf.math.imag
trace = tf.linalg.trace


# N_large is dimension used to compute the tomography matrix
def create_displaced_parity_tf(alphas, N_large=100, N=7):
    from simulator import operators as ops

    D = ops.DisplacementOperator(N_large)
    P = ops.parity(N_large)

    displaced_parity = matmul(matmul(D(alphas), P), D(-alphas))
    # Convert to lower-dimentional Hilbert space; shape=[N_alpha,N,N]
    displaced_parity = displaced_parity[:, :N, :N]

    displaced_parity_re = real(displaced_parity)
    displaced_parity_im = imag(displaced_parity)
    return (displaced_parity_re, displaced_parity_im)


"""
# wigner from qutip density matrix or state vector
def w_qt(rho, betas, N_large=100):
    rho_np = rho.full()
    N = rho_np.shape[0]
    rho = tf.cast(rho_np, c64)
    betas = tf.cast(betas, c64)
    betas_flat = tf.reshape(betas, [-1])
    disp_op = create_disp_op_tf(betas_flat, N_large=N_large, N=N)
    # disp_op = tf.cast(disp_op_laguerre(betas_flat, N=N), c64)

    CF = trace(matmul(rho, disp_op))
    return tf.reshape(CF, betas.shape).numpy()

"""
# wigner from tensorflow state (can be density matrix or state vector)
def wigner_tf(state, alphas, N_large=100):
    N = state.shape[-1]
    dm = state.shape[0] == N
    alphas = tf.cast(alphas, c64)
    alphas_flat = tf.reshape(alphas, [-1])
    parity_op = create_displaced_parity_tf(alphas_flat, N_large=N_large, N=N)
    return parity_op
    if dm:
        W = trace(matmul(state, parity_op))
    else:
        W = matmul(tf.linalg.adjoint(state), matmul(parity_op, state))
    return W


# %%
