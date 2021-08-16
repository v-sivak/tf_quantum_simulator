# %%
%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
from tf_quantum_simulator.hilbert_spaces.displaced_oscillator_qubit import *
from tf_quantum_simulator.utils import *
from tf_quantum_simulator import config
from GKPSimulation import *
from tqdm import tqdm
from gkp_tools import plot_results, save_results, load_results

mult = tf.linalg.matvec

#%%
print("Tensorflow GPU test:")
print("is gpu available:" + str(tf.test.is_gpu_available()))
print("is built with cuda:" + str(tf.test.is_built_with_cuda()))
print(tf.config.list_physical_devices("GPU"))
# %%
params = {k: v for k, v in config.__dict__.items() if '__' not in k and 'np' not in k}
params['gamma_1'] = 0
params['gamma_2'] = 0
params['gamma_phi'] = 0
params['kappa'] = 0
print("Simulation parameters:")
print(params)
system = DisplacedOscillatorQubit(**params)

#%% GKP
l = 2*np.sqrt(np.pi)
alpha = 15.0
#the time for stabilizer CD, just so we write it down
t_stab = l/np.sqrt(2.0)/system._chi/alpha
print('T cd stab: %d' % t_stab)
delta = 0.275
GKP_params = {'l': l,
              'delta': delta,
              'alpha':alpha}
GKP = GKPSimulation(system, **GKP_params)
# %%
N = params['N']
psi0 = Kronecker_product([basis(0,2), basis(0,N)])[0]
n_traj = 5
psi0_batch = copy_state_to_batch(psi0,n_traj)
e = GKP.expect(psi0_batch, output=False)

# %%
#testing stabilization
steps1 = 10
steps2 = 10
phase = np.pi/2.0
psis = [psi0_batch]
exp = []
exp.append(GKP.expect(psis[-1], output=False))
ts = [0]
steps = [0]
t = 0
t_wait = 1e3
psi = psis[-1]
for step in tqdm(range(steps1)):
    t_step, psi = GKP.sbs_step(psi, phase, autonomous=True)
    exp.append(GKP.expect(psi, output=False))
    psis.append(psi)
    steps.append(steps[-1] + 1)
    _, psi = GKP.wait_step(psi, t_wait)
    ts.append(t_step + t_wait + ts[-1])
    phase += np.pi/2.0
t_step, psi = GKP.prepare(psis[-1], '+Z')
psis.append(psi)
exp.append(GKP.expect(psi, output=False))
_, psi = GKP.wait_step(psi, t_wait)
ts.append(ts[-1] + t_wait + t_step)
for step in tqdm(range(steps2)):
    t_step, psi = GKP.sbs_step(psi, phase, autonomous=True)
    exp.append(GKP.expect(psi, output=False))
    psis.append(psi)
    steps.append(steps[-1] + 1)
    _, psi = GKP.wait_step(psi, t_wait)
    ts.append(t_step + t_wait + ts[-1])
    phase += np.pi/2.0
# convert list of dictionaries to dictionary of lists
results = {k: np.array([dic[k] for dic in exp]) for k in exp[0]}
#numpy
ts = np.array([float(t) for t in ts])
steps = np.array(steps)
results['ts'] = ts
results['steps'] = steps
results['params'] = params
results['GKP_params'] = GKP_params
#save_results(results, 'sbs_new.npz',)
plot_results(results)
# %%
r2 = load_results('sbs_new.npz')
# %%
plot_results(r2)
# %%
