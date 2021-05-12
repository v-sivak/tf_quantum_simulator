# %%
%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
from tf_quantum_simulator.hilbert_spaces.displaced_oscillator_qubit import *
from tf_quantum_simulator.utils import *
from tf_quantum_simulator import config

mult = tf.linalg.matvec
# %%
params = {k: v for k, v in config.__dict__.items() if '__' not in k}
#override some parameters
params['gamma_1'] = 0
params['gamma_phi'] = 0
params['kappa'] = 0
params['kerr'] = 0
print("Simulation parameters:")
print(params)
system = DisplacedOscillatorQubit(**params)
#%%
N = params['N']
psi0 = Kronecker_product([basis(0,2), basis(0,N)])[0]
n_traj = 2000
psi0_batch = copy_state_to_batch(psi0,n_traj)
#%%
def print_expect(psi_batch):
    _, norm = normalize(psi_batch)
    norm = tf.reduce_mean(norm)
    print('average norm: %.3f' % norm)
    x = batch_psi_expect(psi_batch, system.sx)
    y = batch_psi_expect(psi_batch, system.sy)
    z = batch_psi_expect(psi_batch, system.sz)
    n = batch_psi_expect(psi_batch, system.n)
    q = batch_psi_expect(psi_batch, system.q)
    p = batch_psi_expect(psi_batch, system.p)
    print('x: %.3f y: %.3f z:%.3f' % (x,y,z))
    print('n: %.3f q: %.3f p:%.3f' % (n,q,p))
#%%
psi1, results = system.measure_and_reset(psi0_batch)
print_expect(psi1)
phi = 0.0
theta = np.pi/2.0
R = system.Rxy(theta, phi)
psi2 = mult(R, psi0_batch)
print_expect(psi2)
psi3, results2 = system.measure_and_reset(psi2)
print_expect(psi3)
