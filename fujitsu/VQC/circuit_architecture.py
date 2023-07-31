import sys, os
import math
import pandas as pd
import numpy as np
from qulacs import QuantumState, QuantumCircuit

## PARAMETERS ## 
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


## CIRCUIT ARCHITECTURE CLASS METHODS ## 
class VC:
	def __init__(self, qubits):
		self.qubits = qubits
		self.layers = 2
		self.set_QFT_status()

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

	## VC circuit architecture methods ## 

	""" method that initializes the weights of a circuit according
		to the circuit architecture, number of layers, qubits, etc. """
	def initial_weights(self):

		# create array of initial weights
		W_init = np.random.normal(0., 0.1, size=(self.layers * self.qubits * 3 + 1)) 

		# create array out boundary values corresponding to each weight
		bounds = [] # bounds of weights
		# initialized as N x 2 array
		for i in range(len(W_init)):
			if i < (len(W_init) - 1):
				bounds.append((lower_unitary_boundary, upper_unitary_boundary))
			else:
				# the final value in the set is the bias applied to each circuit
				# no bounds on linear shift
				bounds.append((None, None))

		return W_init, bounds

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

		print (x)
		exit()

		# apply circuit layers to quantum state

		# measure the expectation value of the circuit





