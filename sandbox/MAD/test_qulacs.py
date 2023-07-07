import sys, os
import numpy as np
import qulacs 
from qulacs import QuantumState
from qulacs import QuantumCircuit
from qulacs.gate import H
from qulacs import Observable

## parameters ## 
verbose = True

# generate n qubit stat
n = 5

if verbose:
	print(f"\nGenerating {n}-qubit state.")

# initialize state of qubits
state = QuantumState(n)
state.set_zero_state()

# build quantum circuit
circuit = QuantumCircuit(n)
for i in range(n):
	# add Hadamar gate to each wire
	gate = H(i)
	print(gate)
	circuit.add_gate(gate)
print(circuit)

# apply hadamar gate to all bits
# generating a uniform super position
circuit.update_quantum_state(state)

# measure the distribution by sampling
# is it even?
obs = Observable(n)
data = obs.get_expectation_value(state)
print(data)
# print(state.get_vector())
