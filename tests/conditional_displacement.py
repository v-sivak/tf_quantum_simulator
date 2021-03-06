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
params['discrete_step_duration'] = 1.0
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
def cd_and_back(psi, beta, alpha):
    R0 = system.Rxy(0, np.pi/2.0)
    psi1 = mult(R0, psi)
    print('psi1:')
    print_expect(psi1)
    t1, psi2 = system.conditional_displacement(psi1, -1*beta, alpha)
    print('psi2:')
    print_expect(psi2)
    t2, psi3 = system.conditional_displacement(psi2, beta, alpha)
    print('psi3:')
    print_expect(psi3)
    psi4, result = system.measure_and_reset(psi3)
    return psi4
#%%
N = params['N']
psi0 = Kronecker_product([basis(0,2), basis(0,N)])[0]
n_traj = 1
psi0_batch = copy_state_to_batch(psi0,n_traj)
psi = psi0_batch
steps = 20
beta = 4.0 # 2.8284271247461903
alpha = 40.0
psis = []
phase = 0
for i in range(steps):
    print('step %d' % i)
    psi = cd_and_back(psi, beta*np.exp(1j*phase), alpha)
    #precision_limit = tf.constant(1e-24, dtype=tf.float32)
    #psi_real = tf.where(tf.math.real(psi) < precision_limit*tf.ones(psi.shape, dtype=tf.float32),tf.zeros(psi.shape), tf.math.real(psi))
    #psi_imag = tf.where(tf.math.imag(psi) < precision_limit*tf.ones(psi.shape, dtype=tf.float32),tf.zeros(psi.shape), tf.math.imag(psi))
    #psi_real = tf.cast(psi_real, dtype=tf.complex64)
    #psi_imag = tf.cast(psi_imag, dtype=tf.complex64)
    #psi = psi_real + 1j*psi_imag
    #psi = tf.cast(psi_real + 1j*psi_imag, dtype=tf.complex64)
    psis.append(psi)
    phase += np.pi/2.0
    #psi, norm = normalize(psi)
    

# %%
xvec = np.linspace(-4,4,61)
x,y = np.meshgrid(xvec,xvec)
alphas = x + 1j*y
W = system.wigner_batch(psi3, alphas)
# %%
plt.pcolormesh(xvec, xvec, np.real(W.numpy()), cmap='seismic', vmin=-1, vmax=+1)
plt.grid()
# %%
