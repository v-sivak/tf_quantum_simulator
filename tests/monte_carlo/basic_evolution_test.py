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
params['discrete_step_duration'] = 1.0/5
print("Simulation parameters:")
print(params)
system = DisplacedOscillatorQubit(**params)
#%%
def print_expect(psi_batch):
    _, norm = normalize(psi_batch)
    norm = tf.reduce_mean(norm)
    print('average norm: %.5f' % norm)
    x = batch_psi_expect(psi_batch, system.sx)
    y = batch_psi_expect(psi_batch, system.sy)
    z = batch_psi_expect(psi_batch, system.sz)
    n = batch_psi_expect(psi_batch, system.n)
    I = batch_psi_expect(psi_batch, system.q)/np.sqrt(2.0)
    Q = batch_psi_expect(psi_batch, system.p)/np.sqrt(2.0)
    print('x: %.5f         y: %.5f         z:%.5f' % (x,y,z))
    print('n: %.5f Re(alpha): %.5f Im(alpha):%.5f' % (n,I,Q))

#%%
N = params['N']
psi0 = Kronecker_product([basis(0,2), basis(0,N)])[0]
n_traj = 1
psi0_batch = copy_state_to_batch(psi0,n_traj)
psi = psi0_batch
alpha = 10.0*1j
H = system._hamiltonian(alpha)
dt = params['discrete_step_duration']
prop = tf.linalg.expm(-1j*H*dt)

#%%
psis = [psi]
steps = 1000
for _ in range(steps):
    psi = mult(prop, psis[-1])
    psis.append(psi)
# %%
xvec = np.linspace(-4,4,61)
x,y = np.meshgrid(xvec,xvec)
alphas = x + 1j*y
W = system.wigner_batch(psis[-1], alphas)
# %%
plt.pcolormesh(xvec, xvec, np.real(W.numpy()), cmap='seismic', vmin=-1, vmax=+1)
plt.grid()
# %%
