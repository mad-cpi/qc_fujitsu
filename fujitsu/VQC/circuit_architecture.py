import sys, os
import math, cmath
import pandas as pd
import numpy as np
import torch
from numpy import pi
from qulacs import QuantumState, QuantumCircuit, Observable
from qulacs.gate import RX, RY, RZ, CNOT, H, DenseMatrix, SWAP
from abc import abstractmethod, ABC
import matplotlib.pyplot as plt
import seaborn as sns

## PARAMETERS ## 
# average of random numbers generation for initial unitary weights
norm_avg = 0.
# standard deviation of random numbers generated for initial unitary weights
norm_std = 0.1
# upper boundary of each universal unitary operation
upper_unitary_boundary = 2. * math.pi 
# lower boundary of each universal unitary operation
lower_unitary_boundary = -2. * math.pi
# default QFT status, determine if QFT is applied to
# quantum state in computational basis before applying
# variational circuit layers / uinitary transormations
default_QFT_status = False

## METHODS UTILIZEDS BY CIRCUIT ARCHITECTURE ##

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

""" method the reverses the order of a bit string
	(in qulacs, the rightmost bit corresponds to
	the zeroth qubit). """
def reverse(s):
    s = s[::-1]
    return s

""" method used to initialize qubit state in computational basis corresponding
	to bit string passed to the method. """
def qubit_encoding (x):

	# reverse bit string due to qulacs nomenclature
	# (in qulacs, 0th bit is represented by the right now digit,
	# not the leftmost)
	x = reverse(x)

	# get the number of qubits in the string, initialize quantum state
	# in the computationa zero basis
	n = len(x)
	s = QuantumState(n)
	s.set_zero_state()

	# initialize circuit used to set qubit state in
	# corresponding computational basis
	c = QuantumCircuit(n)
	# for each integer in the binary string
	for i in range(len(x)):
		# if the bit is one
		if x[i] == 1:
			# add X gate to corresponding qubit
			c.add_X_gate(i)

	# use circuit to update quantum state, return to user
	c.update_quantum_state(s)
	return s

""" method that initializes unitatary weights randomly along a normal distrubution,
	with boundaries that correspond to the upper and lower period boundaries of any
	qubit rotation (last weight in list corresponds to the bias used to shift result
	of classification circuit liniarly, which is used for all classification circuits). """
def random_unitary_weights (n_weights, avg = norm_avg, std = norm_std):
	# initialize weights randomly along a normal distribution
	weights = np.random.normal(avg, std, size=n_weights)

	# assign boundaries to each weight 
	# (last weight in list correspond to linear shift, 
	#	and is assigned different boundaries)
	bounds = [] # bounds of weights
	# initialized as N x 2 array
	for i in range(len(weights)):
		if i < (len(weights) - 1):
			bounds.append((lower_unitary_boundary, upper_unitary_boundary))
		else:
			# the final value in the set is the bias applied to each circuit
			bounds.append((-1, 1))

	return weights, bounds

""" method that generates a circuit, which translates an n-qubit state which is initially in
	the zero computational basis to a the zero fourier basis."""
def qft_init(circuit,n):

    n_init = n
    while n >0:
        a = n_init-n
        circuit.add_gate(H(a)) # Apply the H-gate to the most significant qubit
        for qubit in range(n-1):
            x= 1j*pi/2**(qubit+1)
            den_gate = DenseMatrix([a,a+qubit+1],[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,cmath.exp(x)]])
            circuit.add_gate(den_gate)
        n = n-1
    return circuit

""" creates circruit that adds k to m in the fourier basis state """
def add_k_fourier(k, circuit, n):
	# TODO :: check that k is an integer with the same bit size as m

	# get binary string representation of string

	# for each qubit in the m-fourier basis state
	for x in range(n):
		gate = RZ(x, k * np.pi/(2**(n-x-1)))
        # add gate to circuit
		circuit.add_gate(gate)
	
	# return circuit to user
	return circuit

""" method used to initialize qubit state as fourier basis corresponding to bit string
	passed to the method. """
def QFT_encoding(x):

	# embed x as the computational basis
	n = len(x)
	s = qubit_encoding(x)

	# convert the bit string to an integer, embed the integer in the fourier basis
	k = s.sampling(1)
	cQFT = QuantumCircuit(n)
	cQFT = qft_init(cQFT, n)
	cQFT = add_k_fourier(k[0], cQFT, n)


	# reset state, return the fourier basis to the user
	s = QuantumState(n)
	cQFT.update_quantum_state(s)
	return s

""" TODO :: add method to quantifying qubit circuits (?? forgot what I meant by this) """


## CIRCUIT ARCHITECTURE CLASS METHODS ## 
class ClassificationCircuit (ABC):
	def __init__(self, qubits):
		self.qubits = qubits
		self.set_layers() # number of layers depends on the circuit architecture
		self.set_observable()
		self.set_QFT_status() # initialize QFT operation, default is off

	## generic circuit architecture circuit methods

	""" method for initializing the QFT status."""
	def set_QFT_status(self, status = False):

		# initialize the QFT status
		if status == False or status == True:
			self.QFT_status = status 
		else:
			self.QFT_status = default_QFT_status

	""" methed used by all classification circuits. weights passed to method (W) 
		are used to make prediction / classification for bit string passed to method (x). """
	def classify(self, Wb, x):

		# remove bias from list of weights
		b = Wb[-1]
		W = Wb[:-1]

		# initialize qubit state
		if self.QFT_status ==  True:
			# embed string as fourier basis
			state = QFT_encoding(x)
		else:
			# default is embed bit string as computational basis
			state = qubit_encoding(x)

		# apply circuit layers to quantum state
		for i in range(self.layers):
			# generate a quantum circuit with the specified weights
			c = self.layer(W, i)
			# apply circuit, update quantum state
			c.update_quantum_state(state)

		# make prediction based on expectation value, return to user
		return self.obs.get_expectation_value(state) + b

	""" method that sets the number of layers in the circuit, which is unique
		depending on the circuit architecture. """
	@abstractmethod
	def set_layers(self):
		pass

	""" method the sets the qubit whose state is observed after circuit layers
		have been applied to initial qubit state. """
	@abstractmethod
	def set_observable(self):
		pass

	""" method used by circuit initialize weights of unitary operations """
	@abstractmethod
	def initial_weights(self):
		pass

	""" method returns the weights used for a certain circuit layer,
		specified by i. """
	@abstractmethod
	def get_layer_weights(self, W, i):
		pass

	""" method that returns the wires that should be operated on by the 
		circuit layer corresponding to i """
	@abstractmethod
	def get_layer_qubits(self, i):
		pass

	""" method that describes the circuit applied to quantum circuit
		which contains operations for single layer of classification circuit """
	@abstractmethod
	def layer(self, W, i):
		pass



class VariationalClassifier(ClassificationCircuit):

	# implementation inspired by: 
	# https://pennylane.ai/qml/demos/tutorial_variational_classifier
	# Schuld, Maria, et al. "Circuit-centric quantum classifiers." Physical Review A 101.3 (2020): 032308.

	""" set the number of layers for the variational classification
		circuit """
	def set_layers(self):
		self.layers = 2

	""" set observable for VC circuit """
	def set_observable (self):
		self.obs = Observable(self.qubits)
		self.obs.add_operator(1, 'Z 0')

	""" method that initializes the weights of a circuit according
		to the circuit architecture, number of layers, qubits, etc. """
	def initial_weights(self):

		# determine the number of weights that should be initialized for the circuit
		n_weights = self.layers * self.qubits * 3 + 1

		return random_unitary_weights(n_weights = n_weights)

	""" given a list of weights for the entire circuit, returns an array of weights
		corresponding to the circuit layer specified by i"""
	def get_layer_weights(self, W, i):

		# reshape list containing layer weights
		w = W.reshape(self.layers, self.qubits, 3) # reshape according to number of layers
		# return the weights corresponding to the layer to the user
		return w[i]

	""" get qubits that are operated on by layer of variational classification
		circuit, which for this architecture is all qubits. """
	def get_layer_qubits (self, i):
		q = [x for x in range(self.qubits)]
		return q

	""" build circuit that based on weights passed to method. """
	def layer(self, W, i):
		# initialize circuit
		n = self.qubits
		circuit = QuantumCircuit(n)

		# get the circuit weights from the list
		w = self.get_layer_weights(W, i)

		# for each wire
		for j in range(n):
			# apply each R gate
			for k in range(3):
				# if j == 1:
				# 	print(f"LAYER ({i}), QUBIT ({j}), GATE ({k}), WEIGHT ({w[j][k]})")
				if k == 0:
					# if j == 0, apply RX gate to wire n
					gate = RX (j, w[j][k])
				elif k == 1:
					# if j == 1, aply RY gate to wire n
					gate = RY (j, w[j][k])
				elif k == 2:
					# if j == 2, apply RZ gate to write n
					gate = RZ (j, w[j][k])

				# add gate to circuit
				circuit.add_gate(gate)

		# for each wire
		for i in range(n):
			# in the schulgin data set, the first 5 features
			# in the 16 bit fingerprint are all the same for the 
			# entire dataset, so do not apply control bits to those features
			# if i >= 5 or i == (n-1):
			# target bit
			j = i + 1 
			# periodic wrap at boundaries
			if j >= n:
				j = 0
			# apply control bit to target bit
			gate = CNOT (i, j)
			# add to circuit
			circuit.add_gate(gate)

		return circuit


class TreeTensorNetwork(ClassificationCircuit):

	# implementation inspired by: 
	# https://www.nature.com/articles/s41534-018-0116-9
	# Grant, Edward, et al. "Hierarchical quantum classifiers." npj Quantum Information 4.1 (2018): 65.

	""" set the number of layers for the TTN classification circuit,
		which depends on the number of qubits in the circuit. """
	def set_layers(self):

		# check the number of qubits that should be 
		# input used into the quantum circuit
		n = self.qubits
		fact = 2
		layer = 1
		is_valid = False
		if (n != 0):
			# the number of qubits must be non-zero
			# check that n is an exact factor of two
			while True:
				if n == fact: 
					# the number is exactly factorable by two
					is_valid = True
					break
				elif (fact < n):
					# continue to increase factor by two 
					fact = fact * 2
					layer += 1
				else:
					# i is greater than n,
					# and therefore not an exact factor of 2
					is_valid = False
					break


		# check that the number of qubits assligned to the circuit is correct
		if (not is_valid):
			print(f"\nError :: number of qubits ({self.qubits}) assigned to the circuit must by an integer of log2(N).")
			exit()

		# assign the layers to the object
		self.layers = layer

	""" set observable to TTN circuit """
	def set_observable(self):
		self.obs = Observable(self.qubits)
		# the qubit which is obserbed depends on the overall number of qubits
		operator = 'Z {:d}'.format(10)
		self.obs.add_operator(1, operator)

	""" intialize weights of classification circuit according to the TTN circuit
		architecture. """
	def initial_weights(self):

		# initialize the weights used for the unitary operations
		# the number of weights depends on the number of layers in the circuit
		# determine the number of weights
		n_weights = 0
		for l in range(self.layers):
			n_weights += 3 * int(self.qubits / (2 ** l))
		n_weights += 1

		# initialize the weights randomly, return to user
		return random_unitary_weights(n_weights = n_weights)

	""" method that reshapes list of unitary weights into an array that is organized
		by the operations that are performed for each layer of the classification circuit. """
	def get_layer_weights(self, W, i):	

		w = np.zeros((1))
		# initialize index lower boundary for weights in list
		lw = 0
		# loop through all layers until at final layer
		for j in range(i + 1):
			# determine the number of weights that correspond to the layer
			n_weights = int(self.qubits / (2 ** (j)))
			# set the index corresponding to the upper layer of the weight
			up = lw + 3 * n_weights
			# reshape the list indicies corresponding the layer to a 3 x N array
			w = W[lw:up].reshape(n_weights, 3)
			# assign the lower boundary for the next layer as the upper boundary
			# for the current layer
			lw = up

		# return the layer corresponding to i to the user
		return w

	""" builds circuit that corresponds to a series of unitary operations, which is used to update
		the quantum state passed through the classification circuit ."""
	def layer (self, W, i):

		w = self.get_layer_weights(W, i)
		q = self.get_layer_qubits(i)

		# NOTE :: len(q) and len(W) should be equal
		if len(w) != len(q):
			print(f"Length of TTN layer weights ({len(w)}) and layer qubits ({len(q)}) are not equal.")
			exit()

		# initialize circuit for n qubits
		circuit = QuantumCircuit(self.qubits)

		# apply unitary operations for each wire in q
		for j in range(len(q)):
			# apply each R gate
			for k in range(3):
				if k == 0:
					# if j == 0, apply RX gate to wire n
					gate = RX (q[j], w[j][k])
				elif k == 1:
					# if j == 1, aply RY gate to wire n
					gate = RY (q[j], w[j][k])
				elif k == 2:
					# if j == 2, apply RZ gate to write n
					gate = RZ (q[j], w[j][k])

				# add gate to circuit
				circuit.add_gate(gate)

		# apply CNOT gates for each wire in q with its neighbor (in the list)
		for j in range(0, int(len(q) / 2), 2):
			gate = CNOT (q[j], q[j+1])
			# add gate to circuit
			circuit.add_gate(gate)

		return circuit

	""" get layer qubits for TTN circut architecture """
	def get_layer_qubits (self, i):

		# initialize the list of wires that interact with one another
		# start with list of all cubits
		q = [x for x in range(self.qubits)]

		if i > 0:
			# if not the first layer
			for l in range(i):
				# for each layer
				# remove every other qubit from the list
				# start with empty array
				q_new = []
				for j in range(int(len(q) / 2)):
					if j % 2 == 0:
						q_new.append(q[2 * j + 1])
					else:
						q_new.append(q[2 * j ])
				# update array of qubits with hald of qubits removed
				q = q_new

		return q

