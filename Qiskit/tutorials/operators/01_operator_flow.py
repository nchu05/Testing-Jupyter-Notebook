#!/usr/bin/env python
# coding: utf-8

# # Operator Flow

# ## Introduction
# 
# Qiskit provides classes representing states and operators and sums, tensor products, and compositions thereof. These algebraic constructs allow us to build expressions representing operators.
# 
# We introduce expressions by building them from Pauli operators. In subsequent sections we explore in more detail operators and states, how they are represented, and what we can do with them. In the last section we construct a state, evolve it with a Hamiltonian, and compute expectation values of an observable.
# 
# ### Pauli operators, sums, compositions, and tensor products
# 
# The most important base operators are the Pauli operators.
# The Pauli operators are represented like this.
# 

# In[1]:


from qiskit.opflow import I, X, Y, Z
print(I, X, Y, Z)


# These operators may also carry a coefficient.

# In[2]:


print(1.5 * I)
print(2.5 * X)


# These coefficients allow the operators to be used as terms in a sum.

# In[3]:


print(X + 2.0 * Y)


# Tensor products are denoted with a caret, like this.

# In[4]:


print(X^Y^Z)


# Composition is denoted by the `@` symbol.

# In[5]:


print(X @ Y @ Z)


# In the preceding two examples, the tensor product and composition of Pauli operators were immediately reduced to the equivalent (possibly multi-qubit) Pauli operator. If we tensor or compose more complicated objects, the result is objects representing the unevaluated operations. That is, algebraic expressions.

# For example, composing two sums gives

# In[6]:


print((X + Y) @ (Y + Z))


# And tensoring two sums gives

# In[7]:


print((X + Y) ^ (Y + Z))


# Let's take a closer look at the types introduced above. First the Pauli operators.

# In[8]:


(I, X)


# Each Pauli operator is an instance of `PauliOp`, which wraps an instance of `qiskit.quantum_info.Pauli`, and adds a coefficient `coeff`. In general, a `PauliOp` represents a weighted tensor product of Pauli operators.

# In[9]:


2.0 * X^Y^Z


# For the encoding of the Pauli operators as pairs of Boolean values, see the documentation for `qiskit.quantum_info.Pauli`.
# 
# All of the objects representing operators, whether as "primitive"s such as `PauliOp`, or algebraic expressions carry a coefficient

# In[10]:


print(1.1 * ((1.2 * X)^(Y + (1.3 * Z))))


# In the following we take a broader and deeper look at Qiskit's operators, states, and the building blocks of quantum algorithms.

# 
# ## Part I: State Functions and Measurements
# 
# Quantum states are represented by subclasses of the class `StateFn`. There are four representations of quantum states: `DictStateFn` is a sparse representation in the computational basis, backed by a `dict`. `VectorStateFn` is a dense representation in the computational basis backed by a numpy array. `CircuitStateFn` is backed by a circuit and represents the state obtained by executing the circuit on the all-zero computational-basis state. `OperatorStateFn` represents mixed states via a density matrix. (As we will see later, `OperatorStateFn` is also used to represent observables.)
# 
# Several `StateFn` instances are provided for convenience. For example `Zero, One, Plus, Minus`.

# In[11]:


from qiskit.opflow import (StateFn, Zero, One, Plus, Minus, H,
                           DictStateFn, VectorStateFn, CircuitStateFn, OperatorStateFn)


# `Zero` and `One` represent the quantum states $|0\rangle$ and $|1\rangle$. They are represented via `DictStateFn`.

# In[12]:


print(Zero, One)


# `Plus` and `Minus`, representing states $(|0\rangle + |1\rangle)/\sqrt{2}$ and $(|0\rangle - |1\rangle)/\sqrt{2}$ are represented via circuits. `H` is a synonym for `Plus`.

# In[13]:


print(Plus, Minus)


# Indexing into quantum states is done with the `eval` method. These examples return the coefficients of the `0` and `1` basis states. (Below, we will see that the `eval` method is used for other computations, as well.)

# In[14]:


print(Zero.eval('0'))
print(Zero.eval('1'))
print(One.eval('1'))
print(Plus.eval('0'))
print(Minus.eval('1'))


# The dual vector of a quantum state, that is the *bra* corresponding to a *ket* is obtained via the `adjoint` method. The `StateFn` carries a flag `is_measurement`, which is `False` if the object is a ket and `True` if it is a bra.

# Here, we construct $\langle 1 |$.

# In[15]:


One.adjoint()


# For convenience, one may obtain the dual vector with a tilde, like this

# In[16]:


~One


# ### Algebraic operations and predicates
# 
# Many algebraic operations and predicates between `StateFn`s are supported, including:
# 
# * `+` - addition
# * `-` - subtraction, negation (scalar multiplication by -1)
# * `*` - scalar multiplication
# * `/` - scalar division
# * `@` - composition
# * `^` - tensor product or tensor power (tensor with self n times)
# * `**` - composition power (compose with self n times)
# * `==` - equality
# * `~` - adjoint, alternating between a State Function and Measurement
# 
# Be very aware that these operators obey the [Python rules for operator precedence](https://docs.python.org/3/reference/expressions.html#operator-precedence), which might not be what you expect mathematically.  For example, `I^X + X^I` will actually be parsed as `I ^ (X + X) ^ I == 2 * (I^X^I)` because Python evaluates `+` before `^`.  In these cases, you can use the methods (`.tensor()`, etc) or parentheses.

# `StateFn`s carry a coefficient. This allows us to multiply states by a scalar, and so to construct sums.

# Here, we construct $(2 + 3i)|0\rangle$.

# In[17]:


(2.0 + 3.0j) * Zero


# Here, we see that adding two `DictStateFn`s returns an object of the same type. We construct $|0\rangle + |1\rangle$.

# In[18]:


print(Zero + One)


# Note that you must normalize states by hand. For example, to construct $(|0\rangle + |1\rangle)/\sqrt{2}$, we write

# In[19]:


import math

v_zero_one = (Zero + One) / math.sqrt(2)
print(v_zero_one)


# In other cases, the result is a symbolic representation of a sum. For example, here is a representation of $|+\rangle + |-\rangle$.

# In[20]:


print(Plus + Minus)


# The composition operator is used to perform an inner product, which by default is held in an unevaluated form. Here is a representation of $\langle 1 | 1 \rangle$.

# In[21]:


print(~One @ One)


# Note that the `is_measurement` flag causes the (bra) state `~One` to be printed `DictMeasurement`.

# Symbolic expressions may be evaluated with the `eval` method.

# In[22]:


(~One @ One).eval()


# In[23]:


(~v_zero_one @ v_zero_one).eval()


# Here is $\langle - | 1 \rangle = \langle (\langle 0| - \langle 1|)/\sqrt{2} | 1\rangle$.

# In[24]:


(~Minus @ One).eval()


# The composition operator `@` is equivalent to calling the `compose` method.

# In[25]:


print((~One).compose(One))


# Inner products may also be computed using the `eval` method directly, without constructing a `ComposedOp`.

# In[26]:


(~One).eval(One)


# Symbolic tensor products are constructed as follows. Here is $|0\rangle \otimes |+\rangle$.

# In[27]:


print(Zero^Plus)


# This may be represented as a simple (not compound) `CircuitStateFn`.

# In[28]:


print((Zero^Plus).to_circuit_op())


# Tensor powers are constructed using the caret `^` as follows. Here are $600 (|11111\rangle + |00000\rangle)$, and $|10\rangle^{\otimes 3}$.

# In[29]:


print(600 * ((One^5) + (Zero^5)))
print((One^Zero)^3)


# The method `to_matrix_op` converts to `VectorStateFn`.

# In[30]:


print(((Plus^Minus)^2).to_matrix_op())
print(((Plus^One)^2).to_circuit_op())
print(((Plus^One)^2).to_matrix_op().sample())


# Constructing a StateFn is easy. The `StateFn` class also serves as a factory, and can take any applicable primitive in its constructor and return the correct StateFn subclass. Currently the following primitives can be passed into the constructor, listed alongside the `StateFn` subclass they produce:
# 
# * str (equal to some basis bitstring) -> DictStateFn
# * dict  -> DictStateFn
# * Qiskit Result object -> DictStateFn
# * list -> VectorStateFn
# * np.ndarray -> VectorStateFn
# * Statevector -> VectorStateFn
# * QuantumCircuit -> CircuitStateFn
# * Instruction -> CircuitStateFn
# * OperatorBase -> OperatorStateFn

# In[31]:


print(StateFn({'0':1}))
print(StateFn({'0':1}) == Zero)

print(StateFn([0,1,1,0]))

from qiskit.circuit.library import RealAmplitudes
print(StateFn(RealAmplitudes(2)))


# ## Part II: `PrimitiveOp`s
# 
# The basic Operators are subclasses of `PrimitiveOp`. Just like `StateFn`, `PrimitiveOp` is also a factory for creating the correct type of `PrimitiveOp` for a given primitive. Currently, the following primitives can be passed into the constructor, listed alongside the `PrimitiveOp` subclass they produce:
# 
# * Terra's Pauli -> PauliOp
# * Instruction -> CircuitOp
# * QuantumCircuit -> CircuitOp
# * 2d List -> MatrixOp
# * np.ndarray -> MatrixOp
# * spmatrix -> MatrixOp
# * Terra's quantum_info.Operator -> MatrixOp

# In[32]:


from qiskit.opflow import X, Y, Z, I, CX, T, H, S, PrimitiveOp


# ### Matrix elements

# The `eval` method returns a column from an operator. For example, the Pauli $X$ operator is represented by a `PauliOp`. Asking for a column returns an instance of the sparse representation, a `DictStateFn`.

# In[33]:


X


# In[34]:


print(X.eval('0'))


# It follows that indexing into an operator, that is obtaining a matrix element, is performed with two calls to the `eval` method.

# We have $X = \left(\begin{matrix} 0 & 1 \\
#                             1 & 0
#                             \end{matrix} \right)$. And the matrix element $\left\{X \right\}_{0,1}$ is

# In[35]:


X.eval('0').eval('1')


# Here is an example using the two qubit operator `CX`, the controlled `X`, which is represented by a circuit.

# In[36]:


print(CX)
print(CX.to_matrix().real) # The imaginary part vanishes.


# In[37]:


CX.eval('01')  # 01 is the one in decimal. We get the first column.


# In[38]:


CX.eval('01').eval('11')  # This returns element with (zero-based) index (1, 3)


# ### Applying an operator to a state vector

# Applying an operator to a state vector may be done with the `compose` method (equivalently, `@` operator). Here is a representation of $X | 1 \rangle = |0\rangle$.

# In[39]:


print(X @ One)


# A simpler representation, the `DictStateFn` representation of $|0\rangle$, is obtained with `eval`.

# In[40]:


(X @ One).eval()


# The intermediate `ComposedOp` step may be avoided by using `eval` directly.

# In[41]:


X.eval(One)


# Composition and tensor products of operators are effected with `@` and `^`. Here are some examples.

# In[42]:


print(((~One^2) @ (CX.eval('01'))).eval())

print(((H^5) @ ((CX^2)^I) @ (I^(CX^2)))**2)
print((((H^5) @ ((CX^2)^I) @ (I^(CX^2)))**2) @ (Minus^5))
print(((H^I^I)@(X^I^I)@Zero))


# In[43]:


print(~One @ Minus)


# ## Part III: `ListOp` and subclasses

# ### `ListOp`
# 
# `ListOp` is a container for effectively vectorizing operations over a list of operators and states.

# In[44]:


from qiskit.opflow import ListOp

print((~ListOp([One, Zero]) @ ListOp([One, Zero])))


# For example, the composition above is distributed over the lists (`ListOp`) using the simplification method `reduce`.

# In[45]:


print((~ListOp([One, Zero]) @ ListOp([One, Zero])).reduce())


# ### `ListOp`s: `SummedOp`, `ComposedOp`, `TensoredOp`
# 
# `ListOp`, introduced above, is useful for vectorizing operations. But, it also serves as the superclass for list-like composite classes.
# If you've already played around with the above, you'll notice that you can easily perform operations between `OperatorBase`s which we may not know how to perform efficiently in general (or simply haven't implemented an efficient procedure for yet), such as addition between `CircuitOp`s. In those cases, you may receive a `ListOp` result (or subclass thereof) from your operation representing the lazy execution of the operation. For example, if you attempt to add together a `DictStateFn` and a `CircuitStateFn`, you'll receive a `SummedOp` representing the sum of the two. This composite State function still has a working `eval` (but may need to perform a non-scalable computation under the hood, such as converting both to vectors).
# 
# These composite `OperatorBase`s are how we construct increasingly complex and rich computation out of `PrimitiveOp` and `StateFn` building blocks.
# 
# Every `ListOp` has four properties:
# 
# * `oplist` - The list of `OperatorBase`s which may represent terms, factors, etc.
# * `combo_fn` - The function taking a list of complex numbers to an output value which defines how to combine the outputs of the `oplist` items. For broadcasting simplicity, this function is defined over NumPy arrays.
# * `coeff` - A coefficient multiplying the primitive. Note that `coeff` can be int, float, complex or a free `Parameter` object (from `qiskit.circuit` in Terra) to be bound later using `my_op.bind_parameters`.
# * `abelian` - Indicates whether the Operators in `oplist` are known to mutually commute (usually set after being converted by the `AbelianGrouper` converter).
# 
# Note that `ListOp` supports typical sequence overloads, so you can use indexing like `my_op[4]` to access the `OperatorBase`s in `oplist`.

# ### `OperatorStateFn`
# 
# We mentioned above that `OperatorStateFn` represents a density operator. But, if the `is_measurement` flag is `True`, then `OperatorStateFn` represents an observable. The expectation value of this observable can then be constructed via `ComposedOp`. Or, directly, using `eval`. Recall that the `is_measurement` flag (property) is set via the `adjoint` method.

# Here we construct the observable corresponding to the Pauli $Z$ operator. Note that when printing, it is called `OperatorMeasurement`.

# In[46]:


print(StateFn(Z).adjoint())
StateFn(Z).adjoint()


# Here, we compute $\langle 0 | Z | 0 \rangle$, $\langle 1 | Z | 1 \rangle$, and $\langle + | Z | + \rangle$, where $|+\rangle = (|0\rangle + |1\rangle)/\sqrt{2}$.

# In[47]:


print(StateFn(Z).adjoint().eval(Zero))
print(StateFn(Z).adjoint().eval(One))
print(StateFn(Z).adjoint().eval(Plus))


# ## Part IV: Converters
# 
# Converters are classes that manipulate operators and states and perform building blocks of algorithms. Examples include changing the basis of operators and Trotterization.
# Converters traverse an expression and perform a particular manipulation or replacement, defined by the converter's `convert()` method, of the Operators within. Typically, if a converter encounters an `OperatorBase` in the recursion which is irrelevant to its conversion purpose, that `OperatorBase` is left unchanged.

# In[48]:


import numpy as np
from qiskit.opflow import I, X, Y, Z, H, CX, Zero, ListOp, PauliExpectation, PauliTrotterEvolution, CircuitSampler, MatrixEvolution, Suzuki
from qiskit.circuit import Parameter
from qiskit import Aer


# ### Evolutions, `exp_i()`, and the `EvolvedOp`
# 
# Every `PrimitiveOp` and `ListOp` has an `.exp_i()` function such that `H.exp_i()` corresponds to $e^{-iH}$. In practice, only a few of these Operators have an efficiently computable exponentiation (such as MatrixOp and the PauliOps with only one non-identity single-qubit Pauli), so we need to return a placeholder, or symbolic representation, (similar to how `SummedOp` is a placeholder when we can't perform addition). This placeholder is called `EvolvedOp`, and it holds the `OperatorBase` to be exponentiated in its `.primitive` property.
# 
# Qiskit operators fully support parameterization, so we can use a `Parameter` for our evolution time here. Notice that there's no "evolution time" argument in any function. The Operator flow exponentiates whatever operator we tell it to, and if we choose to multiply the operator by an evolution time, $e^{-iHt}$, that will be reflected in our exponentiation parameters.

# #### Weighted sum of Pauli operators
# A Hamiltonian expressed as a linear combination of multi-qubit Pauli operators may be constructed like this.

# In[49]:


two_qubit_H2 =  (-1.0523732 * I^I) + \
                (0.39793742 * I^Z) + \
                (-0.3979374 * Z^I) + \
                (-0.0112801 * Z^Z) + \
                (0.18093119 * X^X)


# Note that `two_qubit_H2` is represented as a `SummedOp` whose terms are `PauliOp`s.

# In[50]:


print(two_qubit_H2)


# Next, we multiply the Hamiltonian by a `Parameter`. This `Parameter` is stored in the `coeff` property of the `SummedOp`. Calling `exp_i()` on the result wraps it in `EvolvedOp`, representing exponentiation.

# In[51]:


evo_time = Parameter('θ')
evolution_op = (evo_time*two_qubit_H2).exp_i()
print(evolution_op) # Note, EvolvedOps print as exponentiations
print(repr(evolution_op))


# We construct `h2_measurement`, which represents `two_qubit_H2` as an observable.

# In[52]:


h2_measurement = StateFn(two_qubit_H2).adjoint()
print(h2_measurement)


# We construct a Bell state $|\Phi_+\rangle$ via $\text{CX} (H\otimes I) |00\rangle$.

# In[53]:


bell = CX @ (I ^ H) @ Zero
print(bell)


# Here is the expression $H e^{-iHt} |\Phi_+\rangle$.

# In[54]:


evo_and_meas = h2_measurement @ evolution_op @ bell
print(evo_and_meas)


# Typically, we want to approximate $e^{-iHt}$ using two-qubit gates. We achieve this with the `convert` method of `PauliTrotterEvolution`, which traverses expressions applying trotterization to all `EvolvedOp`s encountered. Although we use `PauliTrotterEvolution` here, there are other possibilities, such as `MatrixEvolution`, which performs the exponentiation exactly.

# In[55]:


trotterized_op = PauliTrotterEvolution(trotter_mode=Suzuki(order=2, reps=1)).convert(evo_and_meas)
# We can also set trotter_mode='suzuki' or leave it empty to default to first order Trotterization.
print(trotterized_op)


# `trotterized_op` contains a `Parameter`. The `bind_parameters` method traverses the expression binding values to parameter names as specified via a `dict`. In this case, there is only one parameter.

# In[56]:


bound = trotterized_op.bind_parameters({evo_time: .5})


# `bound` is a `ComposedOp`. The second factor is the circuit. Let's draw it to verify that the binding has taken place.

# In[57]:


bound[1].to_circuit().draw()


# ### Expectations
# 
# `Expectation`s are converters that enable the computation of expectation values of observables. They traverse an Operator tree, replacing `OperatorStateFn`s (observables) with equivalent instructions which are more amenable to
# computation on quantum or classical hardware. For example, if we want to measure the expectation value of an Operator `o` expressed as a sum of Paulis with respect to some state function, but can only access diagonal measurements on quantum hardware, we can create an observable `~StateFn(o)` and use a ``PauliExpectation`` to convert it to a diagonal measurement and circuit pre-rotations to append to the state.
# 
# Another interesting `Expectation` is the `AerPauliExpectation`, which converts the observable into a `CircuitStateFn` containing a special expectation snapshot instruction which `Aer` can execute natively with high performance.

# In[58]:


# Note that XX was the only non-diagonal measurement in our H2 Observable
print(PauliExpectation(group_paulis=False).convert(h2_measurement))


# By default `group_paulis=True`, which will use the `AbelianGrouper` to convert the `SummedOp` into groups of mutually qubit-wise commuting Paulis. This reduces circuit execution overhead, as each group can share the same circuit execution.

# In[59]:


print(PauliExpectation().convert(h2_measurement))


# Note that converters act recursively, that is, they traverse an expression applying their action only where possible. So we can just convert our full evolution and measurement expression. We could have equivalently composed the converted `h2_measurement` with our evolution `CircuitStateFn`. We proceed by applying the conversion on the entire expression.

# In[60]:


diagonalized_meas_op = PauliExpectation().convert(trotterized_op)
print(diagonalized_meas_op)


# Now we bind multiple parameter values into a `ListOp`, followed by `eval` to evaluate the entire expression. We could have used `eval` earlier if we bound earlier, but it would not be efficient. Here, `eval` will convert our `CircuitStateFn`s to `VectorStateFn`s through simulation internally.

# In[61]:


evo_time_points = list(range(8))
h2_trotter_expectations = diagonalized_meas_op.bind_parameters({evo_time: evo_time_points})


# Here are the expectation values $\langle \Phi_+| e^{iHt} H e^{-iHt} |\Phi_+\rangle$ corresponding to the different values of the parameter.

# In[62]:


h2_trotter_expectations.eval()


# ### Executing `CircuitStateFn`s with the `CircuitSampler`
# 
# The `CircuitSampler` traverses an Operator and converts any `CircuitStateFn`s into approximations of the resulting state function by a `DictStateFn` or `VectorStateFn` using a quantum backend. Note that in order to approximate the value of the `CircuitStateFn`, it must 1) send the state function through a depolarizing channel, which will destroy all phase information and 2) replace the sampled frequencies with **square roots** of the frequency, rather than the raw probability of sampling (which would be the equivalent of sampling the **square** of the state function, per the Born rule).

# In[63]:


sampler = CircuitSampler(backend=Aer.get_backend('aer_simulator'))
# sampler.quantum_instance.run_config.shots = 1000
sampled_trotter_exp_op = sampler.convert(h2_trotter_expectations)
sampled_trotter_energies = sampled_trotter_exp_op.eval()
print('Sampled Trotterized energies:\n {}'.format(np.real(sampled_trotter_energies)))


# Note again that the circuits are replaced by dicts with ***square roots*** of the circuit sampling probabilities. Take a look at one sub-expression before and after the conversion:

# In[64]:


print('Before:\n')
print(h2_trotter_expectations.reduce()[0][0])
print('\nAfter:\n')
print(sampled_trotter_exp_op[0][0])


# In[65]:


import qiskit.tools.jupyter
get_ipython().run_line_magic('qiskit_version_table', '')
get_ipython().run_line_magic('qiskit_copyright', '')

