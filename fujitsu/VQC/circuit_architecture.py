import sys, os
import math
import pandas as pd
import numpy as np

## PARAMETERS ## 
# upper boundary of each universal unitary operation
upper_unitary_boundary = 2. * math.pi 
# lower boundary of each universal unitary operation
lower_unitary_boundary = -2. * math.pi

## CIRCUIT ARCHITECTURE / METHODS ## 
class VC:
	def __init__(self, qubits):
		self.qubits = qubits
		self.layers = 2

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



