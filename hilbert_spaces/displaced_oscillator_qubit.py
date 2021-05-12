import tensorflow as tf
import numpy as np
from numpy import pi, sqrt
from tensorflow import complex64 as c64
from tf_quantum_simulator.utils import (
    measurement,
    tensor,
    expectation,
    batch_expect,
    outer_product,
    basis,
    normalize,
)
from .base import HilbertSpace
from tf_quantum_simulator import operators as ops

mult = tf.linalg.matvec
# simulator class for oscillator with dispersive coupling to a qubit
# simulated in the displaced frame of the oscillator
class DisplacedOscillatorQubit(HilbertSpace):
    """
    Define all relevant operators as tensorflow tensors of shape [2N,2N].
    We adopt the notation in which qt.basis(2,0) is a qubit ground state.
    Methods need to take care of batch dimension explicitly.

    Initialize tensorflow quantum trajectory simulator.
    """

    def __init__(
        self,
        *args,
        chi,
        kappa=0,
        kerr=0,
        gamma_1=0,
        gamma_phi=0,
        N=20,
        N_large=100,
        **kwargs
    ):
        """
        Args:
            chi (float): dispersive coupling (GRad/s).
            kappa (float): energy relaxation rate of oscillator (GRad/s).
            gamma_1 (float): energy relaxation rate of qubit (GRad/s).
            gamma_phi (float): dephasing rate of qubit (GRad/s).
            N (int, optional): Size of oscillator Hilbert space.
        """
        self._N = N
        self._N_large = N_large
        self._chi = chi
        self._kappa = kappa
        self._kerr = kerr
        self._gamma_1 = gamma_1
        self._gamma_phi = gamma_phi

        self.alphas_flat = None
        self.tomo_ops = None

        super().__init__(self, *args, **kwargs)

    def _define_fixed_operators(self):
        N = self.N
        N_large = self._N_large
        self.I = tensor([ops.identity(2), ops.identity(N)])
        self.a = tensor([ops.identity(2), ops.destroy(N)])
        self.a_dag = tensor([ops.identity(2), ops.create(N)])
        self.q = tensor([ops.identity(2), ops.position(N)])
        self.p = tensor([ops.identity(2), ops.momentum(N)])
        self.n = tensor([ops.identity(2), ops.num(N)])
        self.parity = tensor([ops.identity(2), ops.parity(N)])

        self.sx = tensor([ops.sigma_x(), ops.identity(N)])
        self.sy = tensor([ops.sigma_y(), ops.identity(N)])
        self.sz = tensor([ops.sigma_z(), ops.identity(N)])
        self.sm = tensor([ops.sigma_m(), ops.identity(N)])

        tensor_with = [ops.identity(2), None]
        self.phase = ops.Phase()
        self.translate = ops.TranslationOperator(N, tensor_with=tensor_with)
        self.displace = lambda a: self.translate(sqrt(2) * a)
        self.rotate = ops.RotationOperator(N, tensor_with=tensor_with)

        # displacement operators with larger intermediate hilbert space used for tomography
        self.translate_large = lambda a: tensor(
            [ops.identity(2), ops.TranslationOperator(N_large)(a)[:, :N, :N]]
        )
        self.displace_large = lambda a: self.translate_large(sqrt(2) * a)
        self.displaced_parity_large = lambda a: tf.linalg.matmul(
            tf.linalg.matmul(self.displace_large(a), self.parity),
            self.displace_large(-a),
        )

        tensor_with = [None, ops.identity(N)]
        self.Rxy = ops.QubitRotationXY(tensor_with=tensor_with)
        self.Rz = ops.QubitRotationZ(tensor_with=tensor_with)

        # qubit sigma_z measurement projector
        self.P = {i: tensor([ops.projector(i, 2), ops.identity(N)]) for i in [0, 1]}

        # qubit reset
        qb_reset = outer_product(basis(0, 2), basis(1, 2)) + outer_product(
            basis(0, 2), basis(0, 2)
        )
        self.qb_reset = tensor([qb_reset, ops.identity(N)])

    def _hamiltonian(self, *H_args):
        alpha = H_args[0]
        # for now, we will assume a static alpha cd hamiltonian, ignoring all other terms
        # later, will include time-dependence in alpha.
        cd = (
            self._chi * alpha * (self.a + self.a_dag) @ (self.sz / 2.0)
            - self._kerr * self.a_dag @ self.a_dag @ self.a @ self.a
        )
        return cd

    @property
    def _collapse_operators(self):
        ops = []
        if self._kappa > 0:
            photon_loss = tf.cast(tf.sqrt(self._kappa), dtype=tf.complex64) * self.a
            ops.append(photon_loss)
        if self._gamma_1 > 0:
            qubit_decay = tf.cast(tf.sqrt(self._gamma_1), dtype=tf.complex64) * self.sm
            ops.append(qubit_decay)
        if self._gamma_phi > 0:
            qubit_dephasing = (
                tf.cast(tf.sqrt(self._gamma_phi / 2.0), dtype=tf.complex64) * self.sz
            )
            ops.append(qubit_dephasing)

        return ops

    # @tf.function
    def conditional_displacement(self, psi_batch, beta, alpha):
        # note: if the discrete_step_size is too large, t will be rounded
        # and the beta will be off.
        t = np.abs(beta) / np.abs(alpha) / self._chi
        alpha_phase = np.angle(beta) + np.pi / 2.0
        alpha = np.abs(alpha) * np.exp(1j * alpha_phase)
        return self.simulate(psi_batch, t, alpha)

    def measure_and_reset(self, psi_batch):
        psi_batch, measurement_results = measurement(psi_batch, self.P, sample=True)
        psi_batch, norm = normalize(mult(self.qb_reset, psi_batch))
        return psi_batch, measurement_results

    def compute_tomo_ops(self, alphas_flat):
        use_precompute = False
        cond1 = self.tomo_ops is not None
        cond2 = cond1 and self.alphas_flat.shape == alphas_flat.shape
        cond3 = cond2 and all(tf.equal(self.alphas_flat, alphas_flat))
        if cond3:
            print("using precomputed tomo ops.")
        else:
            print("constructing tomo ops...")
            # create parity ops with N large, then truncate to N.
            self.alphas_flat = tf.constant(alphas_flat)
            self.tomo_ops = self.displaced_parity_large(alphas_flat)
        return self.tomo_ops

    @tf.function
    def wigner(self, psi, alphas):
        alphas_flat = tf.reshape(alphas, [-1])
        # create parity ops with N large, then truncate to N.
        parity_ops = self.compute_tomo_ops(alphas_flat)
        W = expectation(psi, parity_ops, reduce_batch=False)
        return tf.reshape(W, alphas.shape)

    # @tf.function
    def wigner_batch(self, psi_batch, alphas):
        alphas_flat = tf.reshape(alphas, [-1])
        # create parity ops with N large, then truncate to N.
        parity_ops = self.compute_tomo_ops(alphas_flat)
        W = batch_expect(psi_batch, parity_ops)
        W_shape = (
            [W.shape[0], alphas.shape[0], alphas.shape[1]]
            if len(W.shape) == 2
            else alphas.shape
        )
        return tf.reshape(W, W_shape)

    @property
    def N(self):
        return self._N

    @property
    def tensorstate(self):
        return True
