#%%
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 24 19:32:52 2021

@author: Vladimir Sivak
"""

import tensorflow as tf
from tensorflow import complex64 as c64
from math import pi
import numpy as np
import qutip as qt

# Use the GitHub version of TFCO
# !pip install git+https://github.com/google-research/tensorflow_constrained_optimization
import tensorflow_constrained_optimization as tfco
import matplotlib.pyplot as plt

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
    from tf_quantum_simulator import operators as ops

    D = ops.DisplacementOperator(N_large)
    P = ops.parity(N_large)

    displaced_parity = matmul(matmul(D(alphas), P), D(-alphas))
    # Convert to lower-dimentional Hilbert space; shape=[N_alpha,N,N]
    displaced_parity = displaced_parity[:, :N, :N]

    displaced_parity_re = real(displaced_parity)
    displaced_parity_im = imag(displaced_parity)
    return (displaced_parity_re, displaced_parity_im)


def create_disp_op_tf(betas, N_large=100, N=7):
    from tf_quantum_simulator import operators as ops

    D = ops.DisplacementOperator(N_large)
    # Convert to lower-dimentional Hilbert space; shape=[N_alpha,N,N]
    disp_op = D(betas)[:, :N, :N]
    return real(disp_op), imag(disp_op)


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


# N is dimension of normalized density matrix
def reconstruct_state_wigner(normalized_W_data, alphas_I, alphas_Q, N=7, N_large=100):
    wigner_flat = tf.reshape(normalized_W_data, [-1])
    wigner_flat = tf.cast(wigner_flat, dtype=tf.float32)
    # ----- create displaced parity matrix
    xs_mesh, ys_mesh = np.meshgrid(xs, ys, indexing="xy")
    grid = tf.cast(xs_mesh + 1j * ys_mesh, c64)
    grid_flat = tf.reshape(grid, [-1])
    disp_parity_re, disp_parity_im = create_displaced_parity_tf(
        alphas=grid_flat, N_large=N_large, N=N
    )
    # ----- create parameterization of the density matrix
    A = tf.Variable(tf.random.uniform([N, N]), dtype=tf.float32, name="A")
    B = tf.Variable(tf.random.uniform([N, N]), dtype=tf.float32, name="B")

    def loss_fn():
        rho_im = B - tf.transpose(B)
        rho_re = A + tf.transpose(A)
        W = scale * trace(
            matmul(rho_re, disp_parity_re) - matmul(rho_im, disp_parity_im)
        )
        loss = tf.reduce_mean((wigner_flat - W) ** 2)
        return loss

    # ----- create constrainted minimization problem
    class ReconstructionMLE(tfco.ConstrainedMinimizationProblem):
        def __init__(self, loss_fn, weights):
            self._loss_fn = loss_fn
            self._weights = weights

        @property
        def num_constraints(self):
            return 2

        def objective(self):
            return self._loss_fn()

        def constraints(self):
            A, B = self._weights
            # it works with inequality constraints
            trace_le_1 = trace(A + tf.transpose(A)) - 1
            trace_gr_1 = 1 - trace(A + tf.transpose(A))
            constraints = tf.stack([trace_le_1, trace_gr_1])
            return constraints

    problem = ReconstructionMLE(loss_fn, [A, B])

    optimizer = tfco.LagrangianOptimizer(
        optimizer=tf.optimizers.Adam(learning_rate=0.1),
        num_constraints=problem.num_constraints,
    )

    i = 0
    max_iter = 2000
    stop_loss = 1e-6
    while i < max_iter:
        optimizer.minimize(problem, var_list=[A, B])
        if i % 500 == 0:
            print(f"step = {i}")
            l = loss_fn()
            print(f"loss = {l}")
            print(f"constraints = {problem.constraints()}")
            if l < stop_loss:
                i = max_iter
        i += 1
    print("final:")
    print(f"step = {i}")
    l = loss_fn()
    print(f"loss = {l}")
    print(f"constraints = {problem.constraints()}")

    rho_im, rho_re = B - tf.transpose(B), A + tf.transpose(A)
    rho = tf.cast(rho_re, c64) + 1j * tf.cast(rho_im, c64)

    W = scale * trace(matmul(rho_re, disp_parity_re) - matmul(rho_im, disp_parity_im))
    wigner_reconstructed = tf.reshape(W, grid.shape)

    return qt.Qobj(rho.numpy()), wigner_reconstructed.numpy()


# characteristic function reconstruction
def reconstruct_state_cf(
    normalized_cf_data,
    betas_I,
    betas_Q=None,
    rho_seed=None,
    N=7,
    N_large=100,
    psd_condition=True,
):
    betas_Q = betas_I if betas_Q is None else betas_Q
    CF_flat = tf.reshape(normalized_cf_data, [-1])

    # ----- create displaced parity matrix
    xs_mesh, ys_mesh = np.meshgrid(betas_I, betas_Q, indexing="xy")
    grid = tf.cast(xs_mesh + 1j * ys_mesh, c64)
    grid_flat = tf.reshape(grid, [-1])

    disp_re, disp_im = create_disp_op_tf(betas=grid_flat, N_large=N_large, N=N)
    # disp_op = disp_op_laguerre(grid_flat, N=N)
    # disp_re, disp_im = real(disp_op), imag(disp_op)

    # ----- create parameterization of the density matrix
    if rho_seed is None:
        seed_scale = 1e-4
        A = tf.Variable(
            tf.random.uniform([N, N], minval=-1 * seed_scale, maxval=1 * seed_scale),
            dtype=tf.float32,
            name="A",
        )
        B = tf.Variable(
            tf.random.uniform([N, N], minval=-1 * seed_scale, maxval=1 * seed_scale),
            dtype=tf.float32,
            name="B",
        )
    else:
        A = tf.Variable(np.real(rho_seed.full()) / 2.0, dtype=tf.float32, name="A",)
        B = tf.Variable(np.imag(rho_seed.full()) / 2.0, dtype=tf.float32, name="B",)
    """
    A = tf.Variable(tf.zeros([N, N]), dtype=tf.float32, name="A",)
    B = tf.Variable(tf.zeros([N, N]), dtype=tf.float32, name="B",)
    print("A device:" + str(A.device))
    print("B device:" + str(B.device))
    """

    def loss_fn():
        rho_im = B - tf.transpose(B)
        rho_re = A + tf.transpose(A)
        rho = tf.cast(rho_re, dtype=c64) + tf.cast(
            tf.constant(1j), dtype=c64
        ) * tf.cast(rho_im, dtype=c64)
        if psd_condition:
            e, v = tf.linalg.eigh(rho)
            tr = tf.cast(tf.math.reduce_sum(e), dtype=tf.float32)
            tr_abs = tf.cast(tf.math.reduce_sum(tf.math.abs(e)), dtype=tf.float32)
            loss_e = (tr - tr_abs) ** 2
        else:
            loss_e = 0
        CF_re = trace(matmul(rho_re, disp_re) - matmul(rho_im, disp_im))
        CF_im = trace(matmul(rho_re, disp_im) + matmul(rho_im, disp_re))
        loss_re = tf.reduce_mean((real(CF_flat) - CF_re) ** 2)
        loss_im = tf.reduce_mean((imag(CF_flat) - CF_im) ** 2)
        return loss_re + loss_im + loss_e

    # ----- create constrainted minimization problem
    class ReconstructionMLE(tfco.ConstrainedMinimizationProblem):
        def __init__(self, loss_fn, weights):
            self._loss_fn = loss_fn
            self._weights = weights

        @property
        def num_constraints(self):
            return 2

        def objective(self):
            return loss_fn()

        def constraints(self):
            A, B = self._weights
            # it works with inequality constraints
            # adding more weight to these constraints
            trace_le_1 = 1000 * (trace(A + tf.transpose(A)) - 1)
            trace_gr_1 = 1000 * (1 - trace(A + tf.transpose(A)))
            constraints = tf.stack([trace_le_1, trace_gr_1])
            return constraints

    problem = ReconstructionMLE(loss_fn, [A, B])

    optimizer = tfco.LagrangianOptimizer(
        optimizer=tf.optimizers.Adam(learning_rate=5e-4),
        num_constraints=problem.num_constraints,
    )

    print("initial:")
    l = loss_fn()
    print(f"loss = {l}")
    print(f"constraints = {problem.constraints()}")
    i = 0
    max_iter = 30000
    # max_iter = 0
    stop_loss = 1e-8
    while i < max_iter:
        optimizer.minimize(problem, var_list=[A, B])
        if i % 500 == 0:
            print(f"step = {i}")
            l = loss_fn()
            print(f"loss = {l}")
            print(f"constraints = {problem.constraints()}")
            # print("A:")
            # print(A)
            # print("B:")
            # print(B)
            if l < stop_loss:
                i = max_iter
        i += 1
    print("final:")
    print(f"step = {i}")
    l = loss_fn()
    print(f"loss = {l}")
    print(f"constraints = {problem.constraints()}")

    # ----- get the reconstructed density matrix and CF
    rho_im, rho_re = B - tf.transpose(B), A + tf.transpose(A)
    rho = tf.cast(rho_re, c64) + 1j * tf.cast(rho_im, c64)

    CF_re = trace(matmul(rho_re, disp_re) - matmul(rho_im, disp_im))
    CF_im = trace(matmul(rho_re, disp_im) + matmul(rho_im, disp_re))

    CF_re_reconstructed = tf.reshape(CF_re, grid.shape)
    CF_im_reconstructed = tf.reshape(CF_im, grid.shape)

    return (
        qt.Qobj(rho.numpy()),
        CF_re_reconstructed.numpy() + 1j * CF_im_reconstructed.numpy(),
    )


# The smolin method:
# Will find closest rho which is positive semidefinite (will "fix" negative eigenvalues...)
def smolin(rho):
    from scipy.linalg import eigh

    """ Fast method for maximum likelihood reconstruction 
    Based on Smolin (2012) 10.1103/PhysRevLett.108.070502
    From Phil Reinhold
    """
    dim = rho.shape[0]
    e_vals, e_vecs = eigh(rho)
    lam = e_vals.copy()
    k = 0
    while np.any(lam < 0):
        d = np.argwhere(lam < 0)[-1][0]
        a = np.sum(lam[: d + 1])
        lam[: d + 1] = 0
        lam[d + 1 :] += a / (dim - (d + 1))
        k += 1
        assert k < dim
    return (lam * e_vecs).dot(e_vecs.conj().T)


# %%
