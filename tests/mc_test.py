# %%
%load_ext autoreload
%autoreload 2
import numpy as np
import matplotlib.pyplot as plt
from tf_quantum_simulator.hilbert_spaces.displaced_oscillator_qubit import *
from tf_quantum_simulator.utils import tensor, basis, Kronecker_product, copy_state_to_batch
from tf_quantum_simulator import config
# %%
params = {k: v for k, v in config.__dict__.items() if '__' not in k}
print("Simulation parameters:")
print(params)
system = DisplacedOscillatorQubit(**params)
#%%
N = params['N']
psi0 = Kronecker_product([(basis(0,2) + basis(1,2))/np.sqrt(2.0), basis(0,N)])[0]
psi0_batch = copy_state_to_batch(psi0,100)
#%%
result = system.simulate(psi0_batch, 200.0)
print(result)
# %%
xvec = np.linspace(-4,4,61)
x,y = np.meshgrid(xvec,xvec)
alphas = x + 1j*y
W = system.wigner_batch(result, alphas)
# %%
plt.pcolormesh(xvec, xvec, np.real(W.numpy()), cmap='seismic', vmin=-1, vmax=+1)
plt.grid()

# %%
