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
print("Simulation parameters:")
print(params)
system = DisplacedOscillatorQubit(**params)
#%%
N = params['N']
psi0 = Kronecker_product([basis(1,2), basis(0,N)])[0]

#%%
alpha = 0.0
t_max = 5000
n_traj = 2000
psi0_batch = copy_state_to_batch(psi0,n_traj)
result = system.simulate(psi0_batch, t_max, alpha, return_all_steps=True)
sz = system.sz
z = batch_psi_expect(result, sz)
#%%
def sz_decay(t, tau):
    pe = np.exp(-t/tau)
    return 1-2*pe
steps = int(t_max / params['discrete_step_duration'])
ts = np.arange(steps)*params['discrete_step_duration']
plt.plot(ts, z,'.', label='<sz>')
plt.plot(ts, sz_decay(ts, 1/params['gamma_1']),'--',color='black')
plt.ylabel('<sz>')
plt.xlabel('t ns')
plt.title('MC T1 experiment')

# %%
