import sys, os
import pandas as pd
import numpy as np
import qulacs 
from qulacs import QuantumState
from qulacs import QuantumCircuit
from qulacs import PauliOperator
from qulacs import Observable
from qulacs.gate import H, X, RX, RY, RZ, CNOT
import matplotlib.pyplot as plt
import seaborn as sns

## PARAMETERS ## 
# none


## FUNCTIONS ## 
""" measure distribution of quantum states """
def sample_state_distribution (s, m, save_path = None):
	
	# number of qunits in state
	n = s.get_qubit_count()

	# sample quantum state m-times
	data = s.sampling(m)
	hist = [format(value, "b").zfill(n) for value in data]


	state_dist = pd.DataFrame(dict(State=hist))
	g = sns.displot(data=state_dist, x="State", discrete=True)
	plt.suptitle(f"({n} Qubits Sampled {m} Times)")
	plt.title("Distribution of Quantum Measured Quantum States")
	plt.xlabel("Unique Quantum State")
	plt.ylabel("Number of Times Measured")
	if save_path is None:
		# show the plot
		plt.show()
	else:
		# save the state to the path
		plt.savefig(save_path, dpi = 400, bboxinches = 'tight')


""" initialize bit vector x into n-qubit state with
	m measurement wires. """
def qubit_encoding(x, n, m):

	# x :: bit vector with n elements, molecular fingerprint
	# n :: number of elements in x 
	# m :: number of measurement wires

	# create state with as many qubits as there bits
	# plus the number of measurement wires
	state = QuantumState(n + m)
	# intialize all qubits in "0" state
	state.set_zero_state()

	# build quantum circuit that changes quantum zero state
	# to quantum state representing bit vector
	# (all measurement bits remain |0> state)
	c = QuantumCircuit(n + m)
	# loop through bit vector
	for i in range(n):
		# if the bit is one
		if x[i] == 1:
			# add a not gate to the qubit corresponding
			# to that qubit
			c.add_X_gate(i)

	# apply circuit to quantum state, return the state
	c.update_quantum_state(state)
	return state

""" initialize qubit state by encoding the amplitudes
	of the bit vector """
def amplitude_encoding():
	pass


""" circuit that models a TNN classification network
	DOI: https://doi.org/10.1038/s41534-018-0116-9"""
def TNNclassifier (w, x):

	# number of measurement wires
	m = 1
	# number of qubits
	n = len(x)
	# initialize bit vector into quantum quivalent
	state = qubit_encoding(n, m)

	# for each circuit layer
	l = len(W)
	for i in range(l):
		# generate quantum circuit with weights
		c = TNNcircuit(W[i])
		# apply circuit with weights to qubits
		c.update_quantum_state(state)

	# can probably make list a global variable 
	# (do not need to define each iteration of variational circuit)
	obs = Observable(n)
	# palui z operator measured on mth qubit, scale by constant 1
	obs.add_operator(1., 'Z 0')

	return obs.get_expectation_value(state)

""" Builds TNN quantum circuit TNN classification
	protocol. Applies unitary weights to unitary
	qubit operations, returns circuit. """
def TNNcircuit (w):

	# initialize circuit
	# number of qubits plus one (M)
	n = len(w) + 1
	circuit = QuantumCircuit(n)

	pass


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

	# number of qubits
	n = len(x)
	
	# initialize bit vector into quantum state
	state = stateprep(x, n, 0)

	# for each circuit
	l = len(W)
	for i in range(l):
		# generate quantum circuit with weights
		c = layer(W[i])
		# update quantum state with 
		c.update_quantum_state(state)
		# measure state distributions
		# sample_state_distribution(state, 1000)

	# can probably make list a global variable 
	# (do not need to define each iteration of variational circuit)
	obs = Observable(n)
	# palui z operator measured on 0th qubit, scale by constant 1
	obs.add_operator(1., 'Z 0')

	return obs.get_expectation_value(state)


""" defnes variational classifier, returns
	expectation value."""
def variational_classifier(W, b, x):
	## TODO :: CHECKS!! 
	return variational_circuit(W, x) + b


""" measure squared error  and accuray between 
	measurements and classification labels. """
def error (labels, predictions):
	# tolerance used to measure if a prediction is correct
	tol = 1e-5
	# initialize loss, accuracy measurements
	loss = 0.
	accuracy = 0
	# compare all labels and predictions
	for l, p in zip(labels, predictions):
		loss = loss + (l - p) ** 2
		if abs(l - p) < tol:
			accuracy = accuracy + 1
	# normalize loss, accuracy by the data size
	loss = loss / len(labels)
	accuracy = accuracy / len(labels)
	return loss, accuracy

## ARGUMENTS ## 
# number of qubits in quantum circuit
n = 4
# number of layers in quantum circuit architechture
l = 3


## SCRIPT ##

# create array of random weights
W = np.random.normal(0, 0.1, size=(l, n, 3))

print(variational_classifier(W, 0., [0, 1, 0, 1])) 