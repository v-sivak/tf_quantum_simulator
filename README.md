This repo contains python scripts used for data analysis in the following experiment:

### Real-time quantum error correction beyond break-even
https://arxiv.org/abs/2211.09116
#### Paper abstract:

The ambition of harnessing the quantum for computation is at odds with the fundamental phenomenon of decoherence. The purpose of quantum error correction (QEC) is to counteract the natural tendency of a complex system to decohere. This cooperative process, which requires participation of multiple quantum and classical components, creates a special type of dissipation that removes the entropy caused by the errors faster than the rate at which these errors corrupt the stored quantum information. Previous experimental attempts to engineer such a process faced an excessive generation of errors that overwhelmed the error-correcting capability of the process itself. Whether it is practically possible to utilize QEC for extending quantum coherence thus remains an open question. We answer it by demonstrating a fully stabilized and error-corrected logical qubit whose quantum coherence is significantly longer than that of all the imperfect quantum components involved in the QEC process, beating the best of them with a coherence gain of G=2.27±0.07. We achieve this performance by combining innovations in several domains including the fabrication of superconducting quantum circuits and model-free reinforcement learning.

---
### Repo overview:

1. ***operators.py*** and ***utils.py*** contain basic constructions for simulating quantum evolution with [TensorFlow](https://www.tensorflow.org).

2. ***mc_simulator*** and ***simulate_gkp_logical_lifetime.py*** is a Monte Carlo simulation of quantum error correction (unfinished).

3. ***quantum_control*** directory contains an optimizer for quantum control operations based [Keras](https://keras.io/). Use case examples are provided in ***example_opt_state_prep.py***, ***example_opt_gkp_gate.py***, and ***example_opt_gkp_encoding_unitary.py***. 

4. ***plot_paper_figures*** directory contains data visualization scripts. Experimental data is available upon a reasonable request. 
