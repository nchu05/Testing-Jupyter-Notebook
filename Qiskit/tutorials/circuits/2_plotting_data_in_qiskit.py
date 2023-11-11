#!/usr/bin/env python
# coding: utf-8

# # Qiskit Visualizations

# In[1]:


from qiskit import *
from qiskit.visualization import plot_histogram


# ## Plot Histogram <a name='histogram'></a>
# 
# To visualize the data from a quantum circuit run on a real device or `qasm_simulator` we have made a simple function 
# 
# `plot_histogram(data)`
# 
# As an example we make a 2-qubit Bell state

# In[2]:


# quantum circuit to make a Bell state 
bell = QuantumCircuit(2, 2)
bell.h(0)
bell.cx(0, 1)

meas = QuantumCircuit(2, 2)
meas.measure([0,1], [0,1])

# execute the quantum circuit 
backend = BasicAer.get_backend('qasm_simulator') # the device to run on
circ = bell.compose(meas)
result = backend.run(transpile(circ, backend), shots=1000).result()
counts  = result.get_counts(circ)
print(counts)


# In[3]:


plot_histogram(counts)


# ### Options when plotting a histogram
# 
# The `plot_histogram()` has a few options to adjust the output graph. The first option is the `legend` kwarg. This is used to provide a label for the executions. It takes a list of strings used to label each execution's results. This is mostly useful when plotting multiple execution results in the same histogram. The `sort` kwarg is used to adjust the order the bars in the histogram are rendered. It can be set to either ascending order with `asc` or descending order with `desc`. The `number_to_keep` kwarg takes an integer for the number of terms to show, the rest are grouped together in a single bar called rest. You can adjust the color of the bars with the `color` kwarg which either takes a string or a list of strings for the colors to use for the bars for each execution. You can adjust whether labels are printed above the bars or not with the `bar_labels` kwarg. The last option available is the `figsize` kwarg which takes a tuple of the size in inches to make the output figure.

# In[4]:


# Execute 2-qubit Bell state again
second_result = backend.run(transpile(circ, backend), shots=1000).result()
second_counts  = second_result.get_counts(circ)
# Plot results with legend
legend = ['First execution', 'Second execution']
plot_histogram([counts, second_counts], legend=legend)


# In[5]:


plot_histogram([counts, second_counts], legend=legend, sort='desc', figsize=(15,12), 
               color=['orange', 'black'], bar_labels=False)


# ### Using the output from plot_histogram()
# 
# When using the `plot_histogram()` function, it returns a `matplotlib.Figure` for the rendered visualization. Jupyter notebooks understand this return type and render it for us in this tutorial, but when running outside of Jupyter you do not have this feature automatically. However, the `matplotlib.Figure` class natively has methods to both display and save the visualization. You can call `.show()` on the returned object from `plot_histogram()` to open the image in a new window (assuming your configured matplotlib backend is interactive). Or alternatively you can call `.savefig('out.png')` to save the figure to `out.png`. The `savefig()` method takes a path so you can adjust the location and filename where you're saving the output.

# ## Plot State <a name='state'></a>

# In many situations you want to see the state of a quantum computer. This could be for debugging. Here we assume you have this state (either from simulation or state tomography) and the goal is to visualize the quantum state. This requires exponential resources, so we advise to only view the state of small quantum systems. There are several functions for generating different types of visualization of a quantum state
# 
# ```
# plot_state_city(quantum_state)
# plot_state_qsphere(quantum_state)
# plot_state_paulivec(quantum_state)
# plot_state_hinton(quantum_state)
# plot_bloch_multivector(quantum_state)
# ```
# 
# A quantum state is either a density matrix $\rho$ (Hermitian matrix) or statevector $|\psi\rangle$ (complex vector). The density matrix is related to the statevector by 
# 
# $$\rho = |\psi\rangle\langle \psi|,$$
# 
# and is more general as it can represent mixed states (positive sum of statevectors) 
# 
# $$\rho = \sum_k p_k |\psi_k\rangle\langle \psi_k |.$$
# 
# The visualizations generated by the functions are:
# 
# - `'plot_state_city'`: The standard view for quantum states where the real and imaginary (imag) parts of the density matrix are plotted like a city.
# 
# - `'plot_state_qsphere'`: The Qiskit unique view of a quantum state where the amplitude and phase of the state vector are plotted in a spherical ball. The amplitude is the thickness of the arrow and the phase is the color. For mixed states it will show different `'qsphere'` for each component.
# 
# - `'plot_state_paulivec'`: The representation of the density matrix using Pauli operators as the basis $\rho=\sum_{q=0}^{d^2-1}p_jP_j/d$.
# 
# - `'plot_state_hinton'`: Same as `'city'` but where the size of the element represents the value of the matrix element.
# 
# - `'plot_bloch_multivector'`: The projection of the quantum state onto the single qubit space and plotting on a bloch sphere.

# In[6]:


from qiskit.visualization import plot_state_city, plot_bloch_multivector
from qiskit.visualization import plot_state_paulivec, plot_state_hinton
from qiskit.visualization import plot_state_qsphere


# In[7]:


# execute the quantum circuit 
backend = BasicAer.get_backend('statevector_simulator') # the device to run on
result = backend.run(transpile(bell, backend)).result()
psi  = result.get_statevector(bell)


# In[8]:


plot_state_city(psi)


# In[9]:


plot_state_hinton(psi)


# In[10]:


plot_state_qsphere(psi)


# In[11]:


plot_state_paulivec(psi)


# In[12]:


plot_bloch_multivector(psi)


# Here we see that there is no information about the quantum state in the single qubit space as all vectors are zero. 

# ### Options when using state plotting functions
# 
# The various functions for plotting quantum states provide a number of options to adjust how the plots are rendered. Which options are available depends on the function being used.

# **plot_state_city()** options
# 
# - **title** (str): a string that represents the plot title
# - **figsize** (tuple): figure size in inches (width, height).
# - **color** (list): a list of len=2 giving colors for real and imaginary components of matrix elements.

# In[13]:


plot_state_city(psi, title="My City", color=['black', 'orange'])


# **plot_state_hinton()** options
# 
# - **title** (str): a string that represents the plot title
# - **figsize** (tuple): figure size in inches (width, height).

# In[14]:


plot_state_hinton(psi, title="My Hinton")


# **plot_state_paulivec()** options
# 
# - **title** (str): a string that represents the plot title
# - **figsize** (tuple): figure size in inches (width, height).
# - **color** (list or str): color of the expectation value bars.

# In[15]:


plot_state_paulivec(psi, title="My Paulivec", color=['purple', 'orange', 'green'])


# **plot_state_qsphere()** options
# 
# - **figsize** (tuple): figure size in inches (width, height).

# **plot_bloch_multivector()** options
# 
# - **title** (str): a string that represents the plot title
# - **figsize** (tuple): figure size in inches (width, height).

# In[16]:


plot_bloch_multivector(psi, title="My Bloch Spheres")


# ### Using the output from state plotting functions
# 
# When using any of the state plotting functions it returns a `matplotlib.Figure` for the rendered visualization. Jupyter notebooks understand this return type and render it for us in this tutorial, but when running outside of Jupyter you do not have this feature automatically. However, the `matplotlib.Figure` class natively has methods to both display and save the visualization. You can call `.show()` on the returned object to open the image in a new window (assuming your configured matplotlib backend is interactive). Or alternatively you can call `.savefig('out.png')` to save the figure to `out.png` in the current working directory. The `savefig()` method takes a path so you can adjust the location and filename where you're saving the output.

# ## Plot Bloch Vector <a name='bloch'></a>
# 
# A standard way of plotting a quantum system is using the Bloch vector. This only works for a single qubit and takes as input the Bloch vector. 
# 
# The Bloch vector is defined as $[x = \mathrm{Tr}[X \rho], y = \mathrm{Tr}[Y \rho], z = \mathrm{Tr}[Z \rho]]$, where $X$, $Y$, and $Z$ are the Pauli operators for a single qubit and $\rho$ is the density matrix.
# 

# In[17]:


from qiskit.visualization import plot_bloch_vector


# In[18]:


plot_bloch_vector([0,1,0])


# ### Options for plot_bloch_vector()
# 
# - **title** (str): a string that represents the plot title
# - **figsize** (tuple): Figure size in inches (width, height).

# In[19]:


plot_bloch_vector([0,1,0], title='My Bloch Sphere')


# ### Adjusting the output from plot_bloch_vector()
# 
# When using the `plot_bloch_vector` function it returns a `matplotlib.Figure` for the rendered visualization. Jupyter notebooks understand this return type and render it for us in this tutorial, but when running outside of Jupyter you do not have this feature automatically. However, the `matplotlib.Figure` class natively has methods to both display and save the visualization. You can call `.show()` on the returned object to open the image in a new window (assuming your configured matplotlib backend is interactive). Or alternatively you can call `.savefig('out.png')` to save the figure to `out.png` in the current working directory. The `savefig()` method takes a path so you can adjust the location and filename where you're saving the output.

# In[20]:


import qiskit.tools.jupyter
get_ipython().run_line_magic('qiskit_version_table', '')
get_ipython().run_line_magic('qiskit_copyright', '')
