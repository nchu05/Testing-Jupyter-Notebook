#!/usr/bin/env python
# coding: utf-8

# # Obtaining information about your `backend`
# 
# #### _Note: All the attributes of the backend are described in detail in the [Qiskit Backend Specifications](https://arxiv.org/pdf/1809.03452.pdf). This page reviews a subset of the spec._
# 
# Programming a quantum computer at the microwave pulse level requires more information about the device than is required at the circuit level. A quantum circuit is built for an abstract quantum computer -- it will yield the same quantum state on any quantum computer (except for varying performance levels). A pulse schedule, on the other hand, is so specific to the device, that running one program on two different backends is not expected to have the same result, even on perfectly noiseless systems.
# 
# As a basic example, imagine a drive pulse `q0_X180` calibrated on qubit 0 to enact an $X180$ pulse, which flips the state of qubit 0. If we use the samples from that pulse on qubit 1 on the same device, or qubit 0 on another device, we do not know what the resulting state will be -- but we can be pretty sure it won't be an $X180$ operation. The qubits are each unique, with various drive coupling strengths. If we have specified a frequency for the drive pulse, it's very probable that pulse would have little effect on another qubit, which has its own resonant frequency.
# 
# With that, we have motivated why information from the backend may be very useful at times for building Pulse schedules. The information included in a `backend` is broken into three main parts:
# 
#  - [**Configuration**](#Configuration): static backend features
#  - [**Properties**](#Properties): measured and reported backend characteristics
#  - [**Defaults**](#Defaults): default settings for the OpenPulse-enabled backend
#  
# which are each covered in the following sections. While all three of these contain interesting data for Pulse users, the defaults are _only_ provided for backends enabled with OpenPulse.
# 
# The first thing you'll need to do is grab a backend to inspect. Here we use a mocked backend that contains a snapshot of data from the real OpenPulse-enabled backend.

# In[1]:


from qiskit.providers.fake_provider import FakeHanoi

backend = FakeHanoi()


# ## Configuration
# 
# The configuration is where you'll find data about the static setup of the device, such as its name, version, the number of qubits, and the types of features it supports.
# 
# Let's build a description of our backend using information from the `backend`'s config.

# In[2]:


config = backend.configuration()

# Basic Features
print("This backend is called {0}, and is on version {1}. It has {2} qubit{3}. It "
      "{4} OpenPulse programs. The basis gates supported on this device are {5}."
      "".format(config.backend_name,
                config.backend_version,
                config.n_qubits,
                '' if config.n_qubits == 1 else 's',
                'supports' if config.open_pulse else 'does not support',
                config.basis_gates))


# Neat! All of the above configuration is available for any backend, whether enabled with OpenPulse or not, although it is not an exhaustive list. There are additional attributes available on Pulse backends. Let's go into a bit more detail with those.
# 
# The **timescale**, `dt`, is backend dependent. Think of this as the inverse sampling rate of the control rack's arbitrary waveform generators. Each sample point and duration in a Pulse `Schedule` is given in units of this timescale.

# In[3]:


config.dt  # units of seconds


# The configuration also provides information that is useful for building measurements. Pulse supports three measurement levels: `0: RAW`, `1: KERNELED`, and `2: DISCRIMINATED`. The `meas_levels` attribute tells us which of those are supported by this backend. To learn how to execute programs with these different levels, see this page -- COMING SOON.

# In[4]:


config.meas_levels


# For backends which support measurement level 0, the sampling rate of the control rack's analog-to-digital converters (ADCs) also becomes relevant. The configuration also has this info, where `dtm` is the time per sample returned:

# In[5]:


config.dtm


# The measurement map, explained in detail on [this page COMING SOON], is also found here.

# In[6]:


config.meas_map


# The configuration also supplies convenient methods for getting channels for your schedule programs. For instance:

# In[7]:


config.drive(0)


# In[8]:


config.measure(0)


# In[9]:


config.acquire(0)


# It is a matter of style and personal preference whether you use `config.drive(0)` or `DriveChannel(0)`.
# 
# ## Properties
# 
# The `backend` properties contain data that was measured and optionally reported by the provider. Let's see what kind of information is reported for qubit 0.

# In[10]:


props = backend.properties()


# In[11]:


def describe_qubit(qubit, properties):
    """Print a string describing some of reported properties of the given qubit."""

    # Conversion factors from standard SI units
    us = 1e6
    ns = 1e9
    GHz = 1e-9

    print("Qubit {0} has a \n"
          "  - T1 time of {1} microseconds\n"
          "  - T2 time of {2} microseconds\n"
          "  - U2 gate error of {3}\n"
          "  - U2 gate duration of {4} nanoseconds\n"
          "  - resonant frequency of {5} GHz".format(
              qubit,
              properties.t1(qubit) * us,
              properties.t2(qubit) * us,
              properties.gate_error('sx', qubit),
              properties.gate_length('sx', qubit) * ns,
              properties.frequency(qubit) * GHz))

describe_qubit(0, props)


# Properties are not guaranteed to be reported, but backends without Pulse access typically also provide this data.
# 
# ## Defaults
# 
# Unlike the other two sections, `PulseDefaults` are only available for Pulse-enabled backends. It contains the default program settings run on the device.

# In[12]:


defaults = backend.defaults()


# ### Drive frequencies
# 
# Defaults contains the default frequency settings for the drive and measurement signal channels:

# In[13]:


q0_freq = defaults.qubit_freq_est[0]  # Hz
q0_meas_freq = defaults.meas_freq_est[0]  # Hz

GHz = 1e-9
print("DriveChannel(0) defaults to a modulation frequency of {} GHz.".format(q0_freq * GHz))
print("MeasureChannel(0) defaults to a modulation frequency of {} GHz.".format(q0_meas_freq * GHz))


# ### Pulse Schedule definitions for QuantumCircuit instructions
# 
# Finally, one of the most important aspects of the `backend` for `Schedule` building is the `InstructionScheduleMap`. This is a basic mapping from a circuit operation's name and qubit to the default pulse-level implementation of that instruction. 

# In[14]:


calibrations = defaults.instruction_schedule_map
print(calibrations)


# Rather than build a measurement schedule from scratch, let's see what was calibrated by the backend to measure the qubits on this device:

# In[15]:


measure_schedule = calibrations.get('measure', range(config.n_qubits))
measure_schedule.draw(backend=backend)


# This can easily be appended to your own Pulse `Schedule` (`sched += calibrations.get('measure', <qubits>) << sched.duration`)!
# 
# Likewise, each qubit will have a `Schedule` defined for each basis gate, and they can be appended directly to any `Schedule` you build.

# In[16]:


# You can use `has` to see if an operation is defined. Ex: Does qubit 3 have an x gate defined?
calibrations.has('x', 3)


# In[17]:


# Some circuit operations take parameters. U1 takes a rotation angle:
calibrations.get('u1', 0, P0=3.1415)


# While building your schedule, you can also use `calibrations.add(name, qubits, schedule)` to store useful `Schedule`s that you've made yourself.
# 
# On this [page](07_pulse_scheduler.ipynb), we'll show how to schedule `QuantumCircuit`s into Pulse `Schedule`s.

# In[18]:


import qiskit.tools.jupyter
get_ipython().run_line_magic('qiskit_version_table', '')
get_ipython().run_line_magic('qiskit_copyright', '')

