#!/usr/bin/env python
# coding: utf-8

# # Visualizing a Quantum Circuit

# In[1]:


from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister


# ## Drawing a Quantum Circuit
# 
# When building a quantum circuit, it often helps to draw the circuit. This is supported natively by a `QuantumCircuit` object. You can either call `print()` on the circuit, or call the `draw()` method on the object. This will render a [ASCII art version](https://en.wikipedia.org/wiki/ASCII_art) of the circuit diagram.

# In[2]:


# Build a quantum circuit
circuit = QuantumCircuit(3, 3)

circuit.x(1)
circuit.h(range(3))
circuit.cx(0, 1)
circuit.measure(range(3), range(3));


# In[3]:


print(circuit)


# In[4]:


circuit.draw()


# ## Alternative Renderers for Circuits
# 
# A text output is useful for quickly seeing the output while developing a circuit, but it doesn't provide the most flexibility in its output. There are two alternative output renderers for the quantum circuit. One uses [matplotlib](https://matplotlib.org/), and the other uses [LaTeX](https://www.latex-project.org/), which leverages the [qcircuit package](https://github.com/CQuIC/qcircuit). These can be specified by using `mpl` and `latex` values for the `output` kwarg on the draw() method.

# In[5]:


# Matplotlib Drawing
circuit.draw(output='mpl')


# ## Controlling output from circuit.draw()
# 
# By default, the `draw()` method returns the rendered image as an object and does not output anything. The exact class returned depends on the output specified: `'text'` (the default) returns a `TextDrawer` object, `'mpl'` returns a `matplotlib.Figure` object, and `latex` returns a `PIL.Image` object. Having the return types enables modifying or directly interacting with the rendered output from the drawers. Jupyter notebooks understand these return types and render them for us in this tutorial, but when running outside of Jupyter, you do not have this feature automatically. However, the `draw()` method has optional arguments to display or save the output. When specified, the `filename` kwarg takes a path to which it saves the rendered output. Alternatively, if you're using the `mpl` or `latex` outputs, you can leverage the `interactive` kwarg to open the image in a new window (this will not always work from within a notebook but will be demonstrated anyway).

# ## Customizing the output
# 
# Depending on the output, there are also options to customize the circuit diagram rendered by the circuit.
# 
# ### Disable Plot Barriers and Reversing Bit Order
# The first two options are shared among all three backends. They allow you to configure both the bit orders and whether or not you draw barriers. These can be set by the `reverse_bits` kwarg and `plot_barriers` kwarg, respectively. The examples below will work with any output backend; `mpl` is used here for brevity.

# In[8]:


# Draw a new circuit with barriers and more registers

q_a = QuantumRegister(3, name='qa')
q_b = QuantumRegister(5, name='qb')
c_a = ClassicalRegister(3)
c_b = ClassicalRegister(5)

circuit = QuantumCircuit(q_a, q_b, c_a, c_b)

circuit.x(q_a[1])
circuit.x(q_b[1])
circuit.x(q_b[2])
circuit.x(q_b[4])
circuit.barrier()
circuit.h(q_a)
circuit.barrier(q_a)
circuit.h(q_b)
circuit.cswap(q_b[0], q_b[1], q_b[2])
circuit.cswap(q_b[2], q_b[3], q_b[4])
circuit.cswap(q_b[3], q_b[4], q_b[0])
circuit.barrier(q_b)
circuit.measure(q_a, c_a)
circuit.measure(q_b, c_b);


# In[9]:


# Draw the circuit
circuit.draw(output='mpl')


# In[10]:


# Draw the circuit with reversed bit order
circuit.draw(output='mpl', reverse_bits=True)


# In[11]:


# Draw the circuit without barriers
circuit.draw(output='mpl', plot_barriers=False)


# In[12]:


# Draw the circuit without barriers and reverse bit order
circuit.draw(output='mpl', plot_barriers=False, reverse_bits=True)


# ### Backend-specific customizations
# 
# Some available customizing options are specific to a backend. The `line_length` kwarg for the `text` backend can be used to set a maximum width for the output. When a diagram is wider than the maximum, it will wrap the diagram below. The `mpl` backend has the `style` kwarg, which is used to customize the output. The `scale` option is used by the `mpl` and `latex` backends to scale the size of the output image with a multiplicative adjustment factor. The `style` kwarg takes in a `dict` with multiple options, providing a high level of flexibility for changing colors, changing rendered text for different types of gates, different line styles, etc. Available options are:
# 
# - **textcolor** (str): The color code to use for text. Defaults to `'#000000'`
# - **subtextcolor** (str): The color code to use for subtext. Defaults to `'#000000'`
# - **linecolor** (str): The color code to use for lines. Defaults to `'#000000'`
# - **creglinecolor** (str): The color code to use for classical register lines `'#778899'`
# - **gatetextcolor** (str): The color code to use for gate text `'#000000'`
# - **gatefacecolor** (str): The color code to use for gates. Defaults to `'#ffffff'`
# - **barrierfacecolor** (str): The color code to use for barriers. Defaults to `'#bdbdbd'`
# - **backgroundcolor** (str): The color code to use for the background. Defaults to `'#ffffff'`
# - **fontsize** (int): The font size to use for text. Defaults to 13
# - **subfontsize** (int): The font size to use for subtext. Defaults to 8
# - **displaytext** (dict): A dictionary of the text to use for each element
#     type in the output visualization. The default values are:
#     
#     
#         'id': 'id',
#         'u0': 'U_0',
#         'u1': 'U_1',
#         'u2': 'U_2',
#         'u3': 'U_3',
#         'x': 'X',
#         'y': 'Y',
#         'z': 'Z',
#         'h': 'H',
#         's': 'S',
#         'sdg': 'S^\\dagger',
#         't': 'T',
#         'tdg': 'T^\\dagger',
#         'rx': 'R_x',
#         'ry': 'R_y',
#         'rz': 'R_z',
#         'reset': '\\left|0\\right\\rangle'
#     
#     
#     You must specify all the necessary values if using this. There is
#     no provision for an incomplete dict passed in.
# - **displaycolor** (dict): The color codes to use for each circuit element.
#     By default, all values default to the value of `gatefacecolor` and
#     the keys are the same as `displaytext`. Also, just like
#     `displaytext`, there is no provision for an incomplete dict passed
#     in.
# - **latexdrawerstyle** (bool): When set to True, enable LaTeX mode, which will
#     draw gates like the `latex` output modes.
# - **usepiformat** (bool): When set to True, use radians for output.
# - **fold** (int): The number of circuit elements at which to fold the circuit.
#     Defaults to 20
# - **cregbundle** (bool): If set True, bundle classical registers.
# - **showindex** (bool): If set True, draw an index.
# - **compress** (bool): If set True, draw a compressed circuit.
# - **figwidth** (int): The maximum width (in inches) for the output figure.
# - **dpi** (int): The DPI to use for the output image. Defaults to 150.
# - **creglinestyle** (str): The style of line to use for classical registers.
#     Choices are `'solid'`, `'doublet'`, or any valid matplotlib
#     `linestyle` kwarg value. Defaults to `doublet`.

# In[13]:


# Set line length to 80 for above circuit
circuit.draw(output='text')


# In[14]:


# Change the background color in mpl

style = {'backgroundcolor': 'lightgreen'}

circuit.draw(output='mpl', style=style)


# In[15]:


# Scale the mpl output to 1/2 the normal size
circuit.draw(output='mpl', scale=0.5)


# ## circuit_drawer() as function
# 
# If you have an application where you prefer to draw a circuit with a self-contained function instead of as a method of a circuit object, you can directly use the `circuit_drawer()` function, which is part of the public stable interface from `qiskit.tools.visualization`. The function behaves identically to the `circuit.draw()` method, except that it takes in a circuit object as required argument.
# 
# <div class="alert alert-block alert-info">
# <b>Note:</b> In Qiskit Terra <b> <= 0.7, </b> the default behavior for the circuit_drawer() function is to use the <i>latex</i> output backend, and in <b>0.6.x</b> that includes a fallback to <i>mpl</i> if <i>latex</i> fails for any reason. Starting with release <b> > 0.7, </b>the default changes to the <i>text</i> output.
# </div>

# In[17]:


from qiskit.tools.visualization import circuit_drawer


# In[18]:


circuit_drawer(circuit, output='mpl', plot_barriers=False)


# In[19]:


import qiskit.tools.jupyter
get_ipython().run_line_magic('qiskit_version_table', '')
get_ipython().run_line_magic('qiskit_copyright', '')


# In[ ]:




