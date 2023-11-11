#!/usr/bin/env python
# coding: utf-8

# # VQE with Qiskit Aer Primitives
# 
# This notebook demonstrates how to leverage the [Qiskit Aer Primitives](https://qiskit.org/ecosystem/aer/apidocs/aer_primitives.html) to run both noiseless and noisy simulations locally. Qiskit Aer not only allows you to define your own custom noise model, but also to easily create a noise model based on the properties of a real quantum device. This notebook will show an example of the latter, to illustrate the general workflow of running algorithms with local noisy simulators.
# 
# For further information on the Qiskit Aer noise model, you can consult the [Qiskit Aer documentation](https://qiskit.org/ecosystem/aer/apidocs/aer_noise.html), as well the tutorial for [building noise models](https://qiskit.org/ecosystem/aer/tutorials/3_building_noise_models.html).

# The algorithm of choice is once again VQE, where the task consists on finding the minimum (ground state) energy of a Hamiltonian. As shown in previous tutorials, VQE takes in a qubit operator as input. Here, you will take a set of Pauli operators that were originally computed by Qiskit Nature for the H2 molecule, using the [SparsePauliOp](https://qiskit.org/documentation/stubs/qiskit.quantum_info.SparsePauliOp.html#sparsepauliop) class.

# In[1]:


from qiskit.quantum_info import SparsePauliOp

H2_op = SparsePauliOp.from_list(
    [
        ("II", -1.052373245772859),
        ("IZ", 0.39793742484318045),
        ("ZI", -0.39793742484318045),
        ("ZZ", -0.01128010425623538),
        ("XX", 0.18093119978423156),
    ]
)

print(f"Number of qubits: {H2_op.num_qubits}")


# As the above problem is still easily tractable classically, you can use `NumPyMinimumEigensolver` to compute a reference value to compare the results later.

# In[2]:


from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver
from qiskit.opflow import PauliSumOp

numpy_solver = NumPyMinimumEigensolver()
result = numpy_solver.compute_minimum_eigenvalue(operator=PauliSumOp(H2_op))
ref_value = result.eigenvalue.real
print(f"Reference value: {ref_value:.5f}")


# The following examples will all use the same ansatz and optimizer, defined as follows:

# In[3]:


# define ansatz and optimizer
from qiskit.circuit.library import TwoLocal
from qiskit.algorithms.optimizers import SPSA

iterations = 125
ansatz = TwoLocal(rotation_blocks="ry", entanglement_blocks="cz")
spsa = SPSA(maxiter=iterations)


# ## Performance *without* noise
# 
# Let's first run the `VQE` on the default Aer simulator without adding noise, with a fixed seed for the run and transpilation to obtain reproducible results. This result should be relatively close to the reference value from the exact computation.

# In[4]:


# define callback
# note: Re-run this cell to restart lists before training
counts = []
values = []


def store_intermediate_result(eval_count, parameters, mean, std):
    counts.append(eval_count)
    values.append(mean)


# In[5]:


# define Aer Estimator for noiseless statevector simulation
from qiskit.utils import algorithm_globals
from qiskit_aer.primitives import Estimator as AerEstimator

seed = 170
algorithm_globals.random_seed = seed

noiseless_estimator = AerEstimator(
    run_options={"seed": seed, "shots": 1024},
    transpile_options={"seed_transpiler": seed},
)


# In[6]:


# instantiate and run VQE
from qiskit.algorithms.minimum_eigensolvers import VQE

vqe = VQE(
    noiseless_estimator, ansatz, optimizer=spsa, callback=store_intermediate_result
)
result = vqe.compute_minimum_eigenvalue(operator=H2_op)

print(f"VQE on Aer qasm simulator (no noise): {result.eigenvalue.real:.5f}")
print(
    f"Delta from reference energy value is {(result.eigenvalue.real - ref_value):.5f}"
)


# You captured the energy values above during the convergence, so you can track the process in the graph below.

# In[7]:


import pylab

pylab.rcParams["figure.figsize"] = (12, 4)
pylab.plot(counts, values)
pylab.xlabel("Eval count")
pylab.ylabel("Energy")
pylab.title("Convergence with no noise")


# ## Performance *with* noise
# 
# Now, let's add noise to our simulation. In particular, you will extract a noise model from a (fake) device. As stated in the introduction, it is also possible to create custom noise models from scratch, but this task is beyond the scope of this notebook.
# 
# First, you need to get an actual device backend and from its `configuration` and `properties` you can setup a coupling map and a noise model to match the device. Note: You can also use this coupling map as the entanglement map for the variational form if you choose to.

# In[8]:


from qiskit_aer.noise import NoiseModel
from qiskit.providers.fake_provider import FakeVigo

# fake providers contain data from real IBM Quantum devices stored in Qiskit Terra,
# and are useful for extracting realistic noise models.
device = FakeVigo()

coupling_map = device.configuration().coupling_map
noise_model = NoiseModel.from_backend(device)

print(noise_model)


# Once the noise model is defined, you can run `VQE` using an Aer `Estimator`, where you can pass the noise model to the underlying simulator using the `backend_options` dictionary. Please note that this simulation will take longer than the noiseless one.

# In[9]:


noisy_estimator = AerEstimator(
    backend_options={
        "method": "density_matrix",
        "coupling_map": coupling_map,
        "noise_model": noise_model,
    },
    run_options={"seed": seed, "shots": 1024},
    transpile_options={"seed_transpiler": seed},
)


# Instead of defining a new instance of the `VQE` class, you can now simply assign a new estimator to our previous `VQE` instance. As the callback method will be re-used, you will also need to re-start the `counts` and `values` variables to be able to plot the convergence graph later on.

# In[10]:


# re-start callback variables
counts = []
values = []


# In[11]:


vqe.estimator = noisy_estimator

result1 = vqe.compute_minimum_eigenvalue(operator=H2_op)

print(f"VQE on Aer qasm simulator (with noise): {result1.eigenvalue.real:.5f}")
print(
    f"Delta from reference energy value is {(result1.eigenvalue.real - ref_value):.5f}"
)


# In[12]:


if counts or values:
    pylab.rcParams["figure.figsize"] = (12, 4)
    pylab.plot(counts, values)
    pylab.xlabel("Eval count")
    pylab.ylabel("Energy")
    pylab.title("Convergence with noise")


# ## Summary
# 

# In this tutorial, you compared three calculations for the H2 molecule ground state.  First, you produced a reference value using a classical minimum eigensolver. Then, you proceeded to run `VQE` using the Qiskit Aer `Estimator` with 1024 shots. Finally, you extracted a noise model from a backend and used it to define a new `Estimator` for noisy simulations. The results are:

# In[13]:


print(f"Reference value: {ref_value:.5f}")
print(f"VQE on Aer qasm simulator (no noise): {result.eigenvalue.real:.5f}")
print(f"VQE on Aer qasm simulator (with noise): {result1.eigenvalue.real:.5f}")


# You can notice that, while the noiseless simulation's result is closer to the exact reference value, there is still some difference. This is due to the sampling noise, introduced by limiting the number of shots to 1024. A larger number of shots would decrease this sampling error and close the gap between these two values.
# 
# As for the noise introduced by real devices (or simulated noise models), it could be tackled through a wide variety of error mitigation techniques. The [Qiskit Runtime Primitives](https://qiskit.org/documentation/partners/qiskit_ibm_runtime/) have enabled error mitigation through the `resilience_level` option. This option is currently available for remote simulators and real backends accessed via the Runtime Primitives, you can consult [this tutorial](https://qiskit.org/documentation/partners/qiskit_ibm_runtime/tutorials/Error-Suppression-and-Error-Mitigation.html) for further information.

# In[14]:


import qiskit.tools.jupyter

get_ipython().run_line_magic('qiskit_version_table', '')
get_ipython().run_line_magic('qiskit_copyright', '')

