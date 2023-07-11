import sys, os
import pandas as pd
import numpy as np
import qulacs 
from qulacs import QuantumState
from qulacs import QuantumCircuit
from qulacs.gate import H, RX, RY, RZ, CNOT
import matplotlib.pyplot as plt
import seaborn as sns

## PARAMETERS ## 
# none


## FUNCTIONS ## 
""" initialize bit vector into n-qubit state """
def stateprep(x):

	# create state with as many qubits as there bits
	state = QuantumState(len(x))

	# intialize all qubits in "0" state
	state.set_zero_state()

	# loop through bit vector
	for i in range(len(x)):
		state.set_classical_value(i, x[i])
		# if the bit is one
		if x[i] == 1:
			print(i)
			# flip the corresponding qubit
			# state.set_classical_value(i, 1)

	return state


""" creates an n-qubit circuit, where each qubit
	is rotated by an RX, RY, and RZ matrix, and 
	all neighboring qubits are connected by CNOT
	gates. """
def layer (w):

	# initialize circuit
	n = len(w)
	circuit = QuantumCircuit(n)

	# for each wire
	for i in range(n):
		# apply each R gate
		for j in range(3):
			if j == 0:
				# if j == 0, apply RX gate to wire n
				gate = RX (i, w[i][j])
			elif j == 1:
				# if j == 1, aply RY gate to wire n
				gate = RY (i, w[i][j])
			elif j == 2:
				# if j == 2, apply RZ gate to write n
				gate = RZ (i, w[i][j])

			# add gate to circuit
			circuit.add_gate(gate)

	# for each wire
	for i in range(n - 1):
		# target bit
		j = i + 1 
		# apply control bit to target bit
		gate = CNOT (i, j)
		# add to circuit
		circuit.add_gate(gate)

	return circuit

""" Creates prediction between x and y by:

	1. intializes n-qubit state that represents the 
	computational basis of the bit-vector x
	2. apply variational circuit layer n-times
	3. calculate the Pauli-Z expectation value 
	of the qubits after applying the circuit.
	"""
def variational_circuit(W, x):
	
	state = stateprep(x)
	print(state.get_vector())
	exit()

	l = len(W)
	for i in range(l):
		circuit = layer(W[i])

	return 0


""" defnes variational classifier, returns
	expectation value."""
def variational_classifier(W, b, x):
	## TODO :: CHECKS!! 
	return variational_circuit(W, x) + b

## ARGUMENTS ## 
# number of qubits in quantum circuit
n = 4
# number of layers in quantum circuit architechture
l = 2


## SCRIPT ##

# create array of random weights
W = np.random.normal(0, 0.01, size=(l, n, 3))

print(variational_classifier(W, 0., [0, 1, 0, 1]))

# examine circuit