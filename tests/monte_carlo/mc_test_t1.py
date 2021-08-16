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
params['gamma_1'] = 1/5e3
params['gamma_phi'] = 0
params['kappa'] = 0
params['discrete_step_duration'] = 20.0
print("Simulation parameters:")
print(params)
system = DisplacedOscillatorQubit(**params)
#%%
N = params['N']
psi0 = Kronecker_product([basis(1,2), basis(0,N)])[0]

#%%
alpha = 0.0
t_max = 10000
n_traj = 400
save_frequency = 100
psi0_batch = copy_state_to_batch(psi0,n_traj)
ts, result = system.simulate(psi0_batch, t_max, alpha, save_frequency=save_frequency)
sz = system.sz
z = batch_psi_expect(result, sz)
print(result.shape)
#%%
def sz_decay(t, tau):
    pe = np.exp(-t/tau)
    return 1-2*pe
steps = z.shape[0]
#ts = np.arange(steps)*params['discrete_step_duration']*save_frequency
ts_interpolate = np.linspace(ts[0], ts[-1], 101)
plt.plot(ts_interpolate, sz_decay(ts_interpolate, 1/params['gamma_1']),'--',color='black')
plt.plot(ts, z,'o', label='<sz>')
plt.ylabel('<sz>')
plt.xlabel('t ns')
plt.title('MC T1 experiment')
plt.savefig('t1_example.png', dpi=300)

# %%
