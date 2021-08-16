# %%
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops.gen_functional_ops import stateless_while
from tf_quantum_simulator.hilbert_spaces.displaced_oscillator_qubit import *
from tf_quantum_simulator.utils import *
from tf_quantum_simulator import config

mult = tf.linalg.matvec


class GKPSimulation:
    # todo: generalize to other grids / envelopes.
    # What other types of grids over envelopes can we have?
    # Can the envelope grid be of a different nature than
    # the inner grid
    def __init__(self, system, l, delta, alpha):
        self.system = system
        self.l = l
        self.delta = delta
        self.alpha = alpha
        self._GKP_operators = None

    def GKP_operators(self):
        if self._GKP_operators is None:
            ops = {
                "Sp": self.system.displace(self.l / np.sqrt(2.0)),
                "Sq": self.system.displace(1j * self.l / np.sqrt(2.0)),
                "X": self.system.displace(self.l / 2.0 / np.sqrt(2.0)),
                "Y": self.system.displace(self.l * (1 + 1j) / 2.0 / np.sqrt(2.0)),
                "Z": self.system.displace(1j * self.l / 2.0 / np.sqrt(2.0)),
            }
            E = tf.linalg.expm(-self.delta ** 2 * self.system.n)
            E_inv = tf.linalg.expm(self.delta ** 2 * self.system.n)
            ops["Sp_delta"] = E @ ops["Sp"] @ E_inv
            ops["Sq_delta"] = E @ ops["Sq"] @ E_inv
            ops["X_delta"] = E @ ops["X"] @ E_inv
            ops["Y_delta"] = E @ ops["Y"] @ E_inv
            ops["Z_delta"] = E @ ops["Y"] @ E_inv
            self._GKP_operators = ops
        return self._GKP_operators

    def expect(self, psi_batch, name="GKP", output=True):
        e = self.system.expect(psi_batch, name=name, output=output)
        _, norm = normalize(psi_batch)
        norm = tf.reduce_mean(norm)
        ops = self.GKP_operators()
        for name, op in ops.items():
            exp = batch_psi_expect(psi_batch, op)
            if output:  # todo: make printing pretty
                print(name + " : %.5f + i%.5f" % (np.real(exp), np.imag(exp)))
            e[name] = exp.numpy()
        return e

    # format of all GKP functions will be time, return state

    # before this step, we assume the qubit has been cooled to the ground state.
    # could change this assumption self.later.
    def sbs_step(self, psi_batch, phase=0, autonomous=True, delta=None, l=None):
        l = self.l if l is None else l
        delta = self.delta if delta is None else delta
        # todo: I would have thought this was -pi/2?
        R0 = self.system.Rxy(np.pi / 2.0, np.pi / 2.0)
        R1 = self.system.Rxy(np.pi / 2.0, 0.0)
        R2 = self.system.Rxy(-np.pi / 2.0, 0.0)

        # using CD not CT, hence the factor sqrt(2)
        epsilon = l * np.sinh(delta ** 2)
        beta1 = (epsilon / 2.0) * np.exp(1j * phase) / np.sqrt(2.0)
        beta2 = (-1j * l * np.cosh(delta ** 2)) * np.exp(1j * phase) / np.sqrt(2.0)
        beta3 = (epsilon / 2.0) * np.exp(1j * phase) / np.sqrt(2.0)

        psi1 = mult(R0, psi_batch)
        t1, psi2 = self.system.conditional_displacement(psi1, beta1, self.alpha)
        psi3 = mult(R1, psi2)
        t2, psi4 = self.system.conditional_displacement(psi3, beta2, self.alpha)
        psi5 = mult(R2, psi4)
        if autonomous:
            t3, psi6 = self.system.conditional_displacement(psi5, beta3, self.alpha)
            psi7, msmt_results = self.system.measure_and_reset(psi6)

        else:
            psi6, msmt_results = self.system.measure_and_reset(psi5)
            alpha_plus = beta3 / 2.0
            alpha_minus = -beta3 / 2.0
            t3, psi7 = self.feedback_step(psi6, msmt_results, alpha_plus, alpha_minus)
        total_time = t1 + t2 + t3
        return total_time, psi7

    def sharpen_step(self, psi_batch, phase=0, autonomous=True):
        R0 = self.system.Rxy(np.pi / 2.0, np.pi / 2.0)
        R1 = self.system.Rxy(-np.pi / 2.0, 0.0)

        # using CD not CT, hence the factor sqrt(2)
        epsilon = self.l * np.sinh(self.delta ** 2)
        beta1 = (
            (-1j * self.l * np.cosh(self.delta ** 2))
            * np.exp(1j * phase)
            / np.sqrt(2.0)
        )
        beta2 = (epsilon) * np.exp(1j * phase) / np.sqrt(2.0)

        psi1 = mult(R0, psi_batch)
        t1, psi2 = self.system.conditional_displacement(psi1, beta1, self.alpha)
        psi3 = mult(R1, psi2)
        if autonomous:
            t2, psi4 = self.system.conditional_displacement(psi3, beta2, self.alpha)
            psi5, result = self.system.measure_and_reset(psi4)
        else:
            psi4, msmt_results = self.system.measure_and_reset(psi3)
            alpha_plus = beta2 / 2.0
            alpha_minus = -beta2 / 2.0
            t2, psi5 = self.feedback_step(psi4, msmt_results, alpha_plus, alpha_minus)

        total_time = t1 + t2
        return total_time, psi5

    def trim_step(self, psi_batch, phase=0, autonomous=True):
        R0 = self.system.Rxy(np.pi / 2.0, np.pi / 2.0)
        R1 = self.system.Rxy(np.pi / 2.0, 0.0)
        # using CD not CT, hence the factor sqrt(2)
        epsilon = self.l * np.sinh(self.delta ** 2)
        beta1 = (epsilon) * np.exp(1j * phase) / np.sqrt(2.0)
        beta2 = (
            (-1j * self.l * np.cosh(self.delta ** 2))
            * np.exp(1j * phase)
            / np.sqrt(2.0)
        )

        psi1 = mult(R0, psi_batch)
        t1, psi2 = self.system.conditional_displacement(psi1, beta1, self.alpha)
        psi3 = mult(R1, psi2)
        if autonomous:
            t2, psi4 = self.system.conditional_displacement(psi3, beta2, self.alpha)
            psi5, result = self.system.measure_and_reset(psi4)
        else:
            psi4, msmt_results = self.system.measure_and_reset(psi3)
            alpha_plus = beta2 / 2.0
            alpha_minus = -beta2 / 2.0
            t2, psi5 = self.feedback_step(psi4, msmt_results, alpha_plus, alpha_minus)
        total_time = t1 + t2
        return total_time, psi5

    def wait_step(self, psi_batch, t):
        return self.system.wait(psi_batch, t)

    # if feedback result is -1 or 1, classical feedback displacement
    # will be applied to prepare the desired state.
    def measure_step(self, psi_batch, phase=0, correction_displacement=True):
        R0 = self.system.Rxy(np.pi / 2.0, np.pi / 2.0)
        R1 = self.system.Rxy(np.pi / 2.0, 0.0)
        R2 = self.system.Rxy(-np.pi / 2.0, np.pi / 2.0)

        # using CD not CT, hence the factor sqrt(2)
        epsilon = self.l * np.sinh(self.delta ** 2)
        beta1 = (epsilon / 2.0) * np.exp(1j * phase) / np.sqrt(2.0)
        beta2 = (
            (-1j * self.l * np.cosh(self.delta ** 2) / 2.0)
            * np.exp(1j * phase)
            / np.sqrt(2.0)
        )
        beta3 = (epsilon / 2.0) * np.exp(1j * phase) / np.sqrt(2.0)

        psi1 = mult(R0, psi_batch)
        t1, psi2 = self.system.conditional_displacement(psi1, beta1, self.alpha)
        psi3 = mult(R1, psi2)
        t2, psi4 = self.system.conditional_displacement(psi3, beta2, self.alpha)
        psi5 = mult(R2, psi4)
        t3, psi6 = self.system.conditional_displacement(psi5, beta3, self.alpha)
        psi7, result = self.system.measure_and_reset(psi6)
        # correction displacement
        if correction_displacement:
            D = self.system.displace(
                -1j * self.l * np.exp(1j * phase) / 4 / np.sqrt(2.0)
            )
            psi8 = mult(D, psi7)
        else:
            psi8 = psi7
        total_time = t1 + t2 + t3
        return total_time, psi8, result

    def feedback_step(self, psi_batch, msmt_results, alpha_plus, alpha_minus):
        return (
            0,
            self.system.classical_feedback_displacement(
                psi_batch, msmt_results, alpha_plus, alpha_minus
            ),
        )

    def prepare(self, psi_batch, state="+Z"):
        state = str.upper(state)
        if state[-1] == "Z":
            phase = 0
        elif state[-1] == "X":
            phase = np.pi / 2.0
        elif state[-1] == "Y":
            # todo: fix this case
            print("Y not implemented yet")
            return
            # phase = np.pi/4.0
            # self.l = self.l*np.sqrt(2.0)
        self.alpha_feedback = np.exp(1j * phase) * self.l / 2.0 / np.sqrt(2.0)
        alpha_plus = 0 if state[0] == "+" else self.alpha_feedback
        alpha_minus = self.alpha_feedback if state[0] == "+" else 0

        # First a measurement
        t_measure, psi1, msmt_result = self.measure_step(
            psi_batch, phase=phase, correction_displacement=True,
        )
        # now classical feedback
        t_feedback, psi2 = self.feedback_step(
            psi1, msmt_result, alpha_plus, alpha_minus
        )
        return t_measure + t_feedback, psi2
