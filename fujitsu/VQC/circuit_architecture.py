import sys, os
import math
import pandas as pd
import numpy as np
from qulacs import QuantumState, QuantumCircuit, Observable
from qulacs.gate import RX, RY, RZ, CNOT
from abc import abstractmethod, ABC

## PARAMETERS ## 
# average of random numbers generation for initial unitary weights
norm_avg = 0.
# standard deviation of random numbers generated for initial unitary weights
norm_std=0.1
# upper boundary of each universal unitary operation
upper_unitary_boundary = 2. * math.pi 
# lower boundary of each universal unitary operation
lower_unitary_boundary = -2. * math.pi
# default QFT status, determine if QFT is applied to
# quantum state in computational basis before applying
# variational circuit layers / uinitary transormations
default_QFT_status = True

## METHODS UTILIZEDS BY CIRCUIT ARCHITECTURE ##

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
		if x[i] == '1':
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

""" TODO :: add method to quantifying qubit circuits (?? forgot what I meant by this) """


## CIRCUIT ARCHITECTURE CLASS METHODS ## 
class ClassificationCircuit (ABC):
	def __init__(self, qubits):
		self.qubits = qubits
		self.set_layers() # number of layers depends on the
		# circuit architecture
		self.set_QFT_status()
		# TODO :: parameterize the observable 
		self.obs = Observable(self.qubits)
		self.obs.add_operator(1, 'Z 15')

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
	def classify(self, W, x):

		# reshape weights into arrays representing unitary weights for
		# each layer of classification circuit, and circuit bias
		w, b = self.reweight(W)

		# make prediction, return to method
		return self.predict (w, x) + b

	""" method that sets the number of layers in the circuit, which is unique
		depending on the circuit architecture. """
	@abstractmethod
	def set_layers(self):
		pass

	""" method used by circuit initialize weights of unitary operations """
	@abstractmethod
	def initial_weights(self):
		pass

	""" method used by circuit to reshape list of weights """
	@abstractmethod
	def reweight(self, W):
		pass

	""" method used by circuit to make predictions with weights """
	@abstractmethod
	def predict (self, w, x):
		pass 

	""" method that describes the circuit applied to quantum circuit
		which contains operations for single layer of classification circuit """
	@abstractmethod
	def layer(self, w):
		pass


class VariationalClassifier(ClassificationCircuit):

	# implementation inspired by: 
	# https://pennylane.ai/qml/demos/tutorial_variational_classifier
	# Schuld, Maria, et al. "Circuit-centric quantum classifiers." Physical Review A 101.3 (2020): 032308.

	""" set the number of layers for the variational classification
		circuit """
	def set_layers(self):
		self.layers = 2

	""" method that initializes the weights of a circuit according
		to the circuit architecture, number of layers, qubits, etc. """
	def initial_weights(self):

		# determine the number of weights that should be initialized for the circuit
		n_weights = self.layers * self.qubits * 3 + 1

		return random_unitary_weights(n_weights = n_weights)

	""" method used to reshape list of unitary weights into array containing
		weights for each layer, and bias used to shift prediction. """
	def reweight(self, W):
		# get the layer weights and bias for circuit
		b = W[-1] # bias is last element in list
		w = W[:-1].reshape(self.layers, self.qubits, 3) # remove bias, reshape according to number of layers
		# return to user
		return w, b

	""" quantum variational circuit function that predicts the classification of 
		input information x according the function operations specified by the 
		unitary weights for each circuit layer. """
	def predict (self, w, x):
		
		# initialize qubit state
		if self.QFT_status ==  True:
			print (f"TODO :: implement QFT embedding.")
		else:
			# default is embed bit string as computational basis
			state = qubit_encoding(x)


		# apply circuit layers to quantum state
		for i in range(self.layers):
			# generate a quantum circuit with the specified weights
			c = self.layer(w[i])
			# apply circuit, update quantum state
			c.update_quantum_state(state)

		# measure the expectation value of the circuit
		return self.obs.get_expectation_value(state)

	""" build circuit that based on weights passed to method. """
	def layer(self, w):
		# initialize circuit
		n = self.qubits
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
		for i in range(n):
			# in the schulgin data set, the first 5 features
			# in the 16 bit fingerprint are all the same for the 
			# entire dataset, so do not apply control bits to those features
			if i >= 5 or i == (n-1):
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
			print(f"\nError :: number of qubits ({self.qubit}) assigned to the circuit must by an integer of log2(N).")
			exit()
		elif (verbose):
			print(f"\nThere are {layer} layers in the {self.qubit} qubit TNN classifier.\n")

		# assign the layers to the object
		self.layers = layer

	""" intialize weights of classification circuit according to the TTN circuit
		architecture. """
	def initial_weights(self):

		# initialize the weights used for the unitary operations
		# the number of weights depends on the number of layers in the circuit
		# determine the number of weights
		n_weights = 0
		for l in range(self.layer):
			n_weights += 3 * int(self.qubits / (2 ** l))

		# initialize the weights randomly, return to user
		return random_unitary_weights(n_weights = n_weights)

	""" method that reshapes list of unitary weights into an array that is organized
		by the operations that are performed for each layer of the classification circuit. """
	def reweight(self, W):
		pass

	""" function that uses a list of refactored weights to perform a series of unitary operations
		the predict the classification of an object as either 0 or 1 """
	def predict (self, w, x):
		return 0

	""" builds circuit that corresponds to a series of unitary operations, which is used to update
		the quantum state passed through the classification circuit ."""
	def predict (self, w):
		pass

