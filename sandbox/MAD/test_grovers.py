from qulacs import QuantumState, QuantumCircuit, ParametricQuantumCircuit
from qulacs.gate import CZ, RY, RZ, merge, H, DenseMatrix, to_matrix_gate
from qulacs import Observable
import pandas as pd
import numpy as np
from numpy import pi
import math
import matplotlib.pyplot as plt
import seaborn as sns
from QFT_arithmetic import sum_QFT

## PARAMETERS ## 
# maximum number of allowable qubits that can be initialized
max_qubits = 5


""" creates and returns 3 qubit circuit which contains qAND operation ."""
def qAND ():# test creation of 3-qubit dense matrix gate

	# dense matrix describing qAND operation
	matrix_qAND =  [[1, 0, 0, 0, 0, 0, 0, 0],
					[0, 1, 0, 0, 0, 0, 0, 0],
					[0, 0, 1, 0, 0, 0, 0, 0],
					[0, 0, 0, 1, 0, 0, 0, 0],
					[0, 0, 0, 0, 1, 0, 0, 0],
					[0, 0, 0, 0, 0, 1, 0, 0],
					[0, 0, 0, 0, 0, 0, 0, 1],
					[0, 0, 0, 0, 0, 0, 1, 0]]

	# bits that perform qAND operation
	bits = [0, 1, 2]

	# create circuit
	gate_qAND = DenseMatrix(bits, matrix_qAND)

	# create 3-bit circuit, add gate, return to user
	c = QuantumCircuit(3)
	c.add_gate(gate_qAND)
	return c

""" method that embeds in integer as the computational basis
	of an n-qubit state. """
def reverse(string):
    string = string[::-1]
    return string

""" method that generates a state described by a string, where
	the length of the string describes the number of qubits in
	the state, and each character describes how each qubit should
	be initialize:

		0 = computational basis of 0
		1 = computational basis of 1
		+ = positive superposition between 0 and 1
		- = negative superposition between - and 1

	"""
def gen_state(bit_string):

	# reverse bit string due to qulacs nomenclature
	# (in qulacs, 0th bit is represented by the right now digit,
	# not the leftmost)
	bit_string = reverse(bit_string)

	# get the number of qubits in the string, initialize quantum state
	# in the computationa zero basis
	n = len(bit_string)
	s = QuantumState(n)
	s.set_zero_state()

	# create circuit that initializes the quantum state in one
	# described by the bit string passed to the method
	c = QuantumCircuit(n)
	for i in range(len(bit_string)):
		# add gates to circuit depending on the operation
		# described by the character in the bit string
		b = bit_string[i]
		if b == '0':
			# the state of the bit corresponding to the character
			# has already been initialized in the zero state
			pass 
		elif b == '1':
			# add an X gate to the circuit for the qubit corresponding 
			# to the character, which flips the bit from 0 to 1
			c.add_X_gate(i)
		elif b == '+':
			# apply hadamar gate to circuit for qubit corresponding
			# to the character, which places the qubit in a positive
			# super position
			c.add_H_gate(i)
		elif b == '-':
			# add an X gate and an H gate to the circuit for the 
			# qubit corresponding to the character, which places the qubit
			# in a negative superposition
			c.add_X_gate(i)
			c.add_H_gate(i)

	# apply circuit to state, return the state to the user
	c.update_quantum_state(s)
	return s

""" Returns circuit that applies grover's algorithm according to 
	"Quantum Computationa and Quantum Information" by Nielsen and 
	Chuang, Box 6.1 """
def grovers():
	# 
	pass

""" measure distribution of quantum states.

	BITS :: list of bits that should be selected from the sampled stats
	LSB :: boolean corresponding to the bit containing the 0th state.
		if LSB == true, then right most bit is the 0th state, else leftmost is 0th. """
def sample_state_distribution (s, m, LSB = False, bits = None, save_path = None):
	
	# number of qunits in state
	n = s.get_qubit_count()

	# sample quantum state m-times
	data = s.sampling(m)
	hist = [format(value, "b").zfill(n) for value in data]
	if not LSB:
		hist = [reverse(val) for val in hist]

	# this is a bit confusing
	if not bits is None:
		# get only the bits corresponding to those in the list
		if LSB:
			for i in range(len(hist)):
				binary = hist[i]
				binary_selected = ''
				for j in range(len(bits) - 1, -1, -1):
					k = len(binary) - bits[j] - 1
					binary_selected += binary[k]
				hist[i] = binary_selected
		else:
			for i in range(len(hist)):
				binary = hist[i]
				binary_selected = ''
				for j in range(len(bits) - 1, -1, -1):
					binary_selected += binary[j]
				hist[i] = binary_selected


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


# create 3 bit quantum state
init_state = '++-'
state = gen_state(init_state)

# apply qAND gate to circuit, measure results
c_qAND = qAND()
c_qAND.update_quantum_state(state)
sample_state_distribution(state, 10000)

# perform sign flip operation
c_flip = grovers(2, 1)


