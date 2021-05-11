# %%
%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
from tf_quantum_simulator.hilbert_spaces.displaced_oscillator_qubit import *
from tf_quantum_simulator.utils import *
from tf_quantum_simulator import config
# %%
params = {k: v for k, v in config.__dict__.items() if '__' not in k}
#override some parameters
params['gamma_1'] = 0
params['gamma_phi'] = 0
params['kappa'] = 1/20e3
print("Simulation parameters:")
print(params)

system = DisplacedOscillatorQubit(**params)
#%%
N = params['N']
cat_alpha = 3
D = (system.displace(-cat_alpha) + system.displace(cat_alpha))
psi0 = Kronecker_product([basis(0,2), basis(0,N)])[0]
psi0, _ = normalize(tf.linalg.matvec(D, psi0))

#%%
alpha = 0.0
t_max = 5000
n_traj = 2000
save_frequency = 150
psi0_batch = copy_state_to_batch(psi0,n_traj)
result = system.simulate(psi0_batch, t_max, alpha, save_frequency=save_frequency)
#%%
n = system.n
n_exp = batch_psi_expect(result, n)
#%%
def n_decay(t, tau, n0 = 1):
    n = n0*np.exp(-t/tau)
    return n
steps = n_exp.shape[0]
ts = np.arange(steps)*params['discrete_step_duration']*save_frequency
ts_interpolate = np.linspace(ts[0], ts[-1], 101)
plt.plot(ts_interpolate, n_decay(ts_interpolate, 1/params['kappa'], n0=np.abs(cat_alpha)**2),'--',color='black')
plt.plot(ts, n_exp,'o', label='<n>')
plt.ylabel('<n>')
plt.xlabel('t ns')
plt.title('cat decay experiment')
# %%
xvec = np.linspace(-4,4,81)
x,y = np.meshgrid(xvec,xvec)
alphas = x + 1j*y
#%%
W = system.wigner_batch(result, alphas)
# %%
num = steps
fig, axs = plt.subplots(1, num, figsize=(12,2))
for i, w, t in zip(np.arange(len(W)),W,ts):
    axs[i].pcolormesh(xvec, xvec, np.real(w.numpy()), cmap='seismic', vmin=-1, vmax=+1)
    axs[i].set_xticks([])
    axs[i].set_yticks([])
    axs[i].set_title('t = %.2f us' % (t/1e3))
    axs[i].set_aspect('equal')
plt.show()
# %%
