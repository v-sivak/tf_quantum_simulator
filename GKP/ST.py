# %%
import numpy as np
from tf_quantum_simulator.hilbert_spaces.displaced_oscillator_qubit import *
from tf_quantum_simulator.utils import *
from two_mode_GKP.GKPSimulation import *
from tqdm import tqdm
from two_mode_GKP.gkp_tools import save_results

mult = tf.linalg.matvec

#%%
print("Tensorflow GPU test:")
print("is gpu available:" + str(tf.test.is_gpu_available()))
print("is built with cuda:" + str(tf.test.is_built_with_cuda()))
print(tf.config.list_physical_devices("GPU"))

#%%
params = {
    # Oscillator
    "chi": 2 * np.pi * 1e-6 * 29,
    "kappa": 1 / (245e3),
    "kerr": 0,
    # qubit
    "gamma_1": 1 / (50e3),
    "gamma_phi": 1 / (2 * 50e3),
    # Hilbert space size
    "N": 80,
    # Hilbert space size for intermediate calculation of displacement operators for tomography
    "N_large": 150,
    # Simulator discretization in ns
    "discrete_step_duration": 1.0,
}
print("Simulation parameters:")
print(params)
system = DisplacedOscillatorQubit(**params)
# GKP
l = 2 * np.sqrt(np.pi)
# the alpha which gives a CD in 1us
alpha = l / np.sqrt(2.0) / system._chi / 1e3
print("alpha: %.3f" % alpha)
#%%
delta = 0.25
GKP_params = {"l": l, "delta": delta, "alpha": alpha}
GKP = GKPSimulation(system, **GKP_params)
N = params["N"]
psi0 = Kronecker_product([basis(0, 2), basis(0, N)])[0]
n_traj = 500
psi0_batch = copy_state_to_batch(psi0, n_traj)
e = GKP.expect(psi0_batch, output=False)
# stabilization
steps1 = 50
steps2 = 500
phase = np.pi / 2.0
psis = [psi0_batch]
exp = []
exp.append(GKP.expect(psis[-1], output=False))
ts = [0]
steps = [0]
t = 0
t_wait = 1.5e3
psi = psis[-1]
for step in tqdm(range(steps1)):
    t_step, psi = GKP.sharpen_step(psi, phase, autonomous=False)
    exp.append(GKP.expect(psi, output=False))
    psis.append(psi)
    _, psi = GKP.wait_step(psi, t_wait)
    ts.append(t_step + t_wait + ts[-1])
    steps.append(steps[-1] + 1)
    t_step, psi = GKP.trim_step(psi, phase, autonomous=False)
    exp.append(GKP.expect(psi, output=False))
    psis.append(psi)
    _, psi = GKP.wait_step(psi, t_wait)
    ts.append(t_step + t_wait + ts[-1])
    steps.append(steps[-1] + 1)
    phase += np.pi / 2.0
t_step, psi = GKP.prepare(psis[-1], "+Z")
psis.append(psi)
exp.append(GKP.expect(psi, output=False))
_, psi = GKP.wait_step(psi, t_wait)
ts.append(ts[-1] + t_wait + t_step)
steps.append(steps[-1] + 1)
# _, psi = GKP.wait_step(psi, t_wait)
for step in tqdm(range(steps2)):
    t_step, psi = GKP.sharpen_step(psi, phase, autonomous=False)
    exp.append(GKP.expect(psi, output=False))
    psis.append(psi)
    _, psi = GKP.wait_step(psi, t_wait)
    ts.append(t_step + t_wait + ts[-1])
    steps.append(steps[-1] + 1)
    _, psi = GKP.wait_step(psi, t_wait)
    t_step, psi = GKP.trim_step(psi, phase, autonomous=False)
    exp.append(GKP.expect(psi, output=False))
    psis.append(psi)
    _, psi = GKP.wait_step(psi, t_wait)
    ts.append(t_step + t_wait + ts[-1])
    steps.append(steps[-1] + 1)
    phase += np.pi / 2.0
# convert list of dictionaries to dictionary of lists
results = {k: np.array([dic[k] for dic in exp]) for k in exp[0]}
# numpy
ts = np.array([float(t) for t in ts])
steps = np.array(steps)
results["ts"] = ts
results["steps"] = steps
results["params"] = params
results["GKP_params"] = GKP_params
name = "ST_old_params_delta_%.3f.npz" % delta
save_results(
    results, name,
)

# %%
