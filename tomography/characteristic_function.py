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


def disp_op_laguerre(disps, N=7):
    dim = N
    alphas = np.array(disps)
    betas = np.abs(alphas) ** 2

    x_mat = np.zeros((len(disps), dim, dim), dtype=DTYPE)

    for m in range(dim):
        x_mat[:, m, m] = genlaguerre(m, 0)(betas)
        for n in range(0, m):  # scan over lower triangle, n < m
            x_mn = (
                np.sqrt(factorial(n) / factorial(m))
                * (alphas) ** (m - n)
                * genlaguerre(n, m - n)(betas)
            )
            x_mat[:, m, n] = x_mn

        for n in range(m + 1, dim):  # scan over upper triangle, m < n
            x_mn = (
                np.sqrt(factorial(m) / factorial(n))
                * (-alphas.conj()) ** (n - m)
                * genlaguerre(m, n - m)(betas)
            )
            x_mat[:, m, n] = x_mn

    x_mat = np.einsum("ijk,i->ijk", x_mat, np.exp(-betas / 2))
    # x_mat = np.matrix(x_mat.reshape(len(disps), dim ** 2))
    return x_mat  # .conj()


def create_disp_op_tf(betas, N_large=100, N=7):
    from tf_quantum_simulator import operators as ops

    D = ops.DisplacementOperator(N_large)
    # Convert to lower-dimentional Hilbert space; shape=[N_alpha,N,N]
    disp_op = D(betas)[:, :N, :N]
    return disp_op


# charasteristic function from qutip density matrix


def cf_rho_qt(rho, betas, N_large=100):
    rho_np = rho.full()
    N = rho_np.shape[0]
    rho = tf.cast(rho_np, c64)
    betas = tf.cast(betas, c64)
    betas_flat = tf.reshape(betas, [-1])
    disp_op = create_disp_op_tf(betas_flat, N_large=N_large, N=N)
    # disp_op = tf.cast(disp_op_laguerre(betas_flat, N=N), c64)

    CF = trace(matmul(rho, disp_op))
    return tf.reshape(CF, betas.shape).numpy()


# charasteristic function from tensorflow
# can take
def cf_rho_tf(rho, betas, N_large=100):
    N = rho.shape[-1]
    rho = tf.cast(rho_np, c64)
    betas = tf.cast(betas, c64)
    betas_flat = tf.reshape(betas, [-1])
    disp_op = create_disp_op_tf(betas_flat, N_large=N_large, N=N)
    CF = trace(matmul(rho, disp_op))
    return tf.reshape(CF, betas.shape).numpy()


# %%
