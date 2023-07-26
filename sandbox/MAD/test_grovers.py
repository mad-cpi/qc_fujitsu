from qulacs import QuantumState, QuantumCircuit, ParametricQuantumCircuit
from qulacs.gate import CZ, RY, RZ, merge, H, DenseMatrix, to_matrix_gate
from qulacs import Observable
import pandas as pd
import numpy as np
from numpy import pi
import math
import matplotlib.pyplot as plt
import seaborn as sns
from QFT_arithmetic import integer_basis_embedding
from QFT_arithmetic import sum_QFT
from QFT_arithmetic import qft, inverse_qft

## PARAMETERS ## 
# maximum number of allowable qubits that can be initialized
max_qubits = 5

""" method that embeds Fourier basis of values contained in
	control qubits in the target qubits."""
def embed_QFT(control_qubits, target_qubits):

	c_QFT = QuantumCircuit(len(target_qubits))

	for c in range(len(control_qubits)):
		val = 2 ** (len(control_qubits) - c - 1)
		for t in range(len(target_qubits)):
			gate = RZ(target_qubits[t], -val * np.pi / (2 ** t))
			gate_mat = to_matrix_gate(gate)
			print(control_qubits[c])
			gate_mat.add_control_qubit(control_qubits[c], 1)
			print(gate)
			print(gate_mat)

	return c_QFT

""" method that embeds in integer as the computational basis
	of an n-qubit state. """
def reverse(string):
    string = string[::-1]
    return string

""" embed the value of an integer m within the state of 
	a list of target qubits that are passed to the method.

	NOTE :: the binary representation of m must no more than
	the number of target qubits that were specified. """
def integer_basis_embedding (m, state, target_qubits):

	# get binary string representation of string
	binary = format(m, 'b')
	rev_binary = reverse(binary)   

	# check that the number of binary bits is not
	# greater than the number of target qubits
	if (len(binary) > len(target_qubits)):
		print(f"The number of target qubits ({len(target_qubits)}) is not enough for the binary string ({binary})")
		exit()

	# initialize circuit used to set qubit state in
	# corresponding computational basis
	print(f"Storing binary ({binary}) in qubit registar ({target_qubits}).")
	c = QuantumCircuit(state.get_qubit_count())
	# store each binary used to describe the integer
	# backwards, starting from the last qubit in the list
	for x in range(len(binary) - 1, -1, -1):
		# if the bit is one
		if binary[x] == '1':
			# determine the appropriate qubit
			q = len(target_qubits) - (len(binary) - x - 1)
			# add X gate to corresponding qubit
			c.add_X_gate(target_qubits[q])

	# set quantum state in computational basis, return to user
	c.update_quantum_state(state)

	return state

def test_QFT(m):

	# translate computational basis of control qubits 
	# to the fourier basis stored in the target qubits

	# initialize quantum state that will contain both computational
	# and fourier basis of integer m
	comp_qubits = [x for x in range(len(format(m, 'b')))]
	four_qubits = [x for x in range(len(comp_qubits), len(comp_qubits) * 2)]
	m_state = QuantumState(len(comp_qubits) + len(four_qubits))
	m_state.set_zero_state()

	# place integer in the computational basis
	m_state = integer_basis_embedding(m, m_state, comp_qubits)
	int_in = m_state.sampling(1)
	# TODO :: get method that pulls integer from binary string with right LSB
	print(f"The input integer is {int_in[0]}")
	# sample_state_distribution(m_state, 1000, bits = comp_qubits, LSB = False)

	# convert m to the Fourier basis
	c = embed_QFT(comp_qubits, four_qubits)

	# m_state = QuantumState(m)
	# m_state.set_zero_state()
	# QFT = QuantumCircuit(m)
	# QFT = qft(QFT, m)
	# QFT.update_quantum_state(m_state)


	# remove m from Fourier basis
	# iQFT = QuantumCircuit(n)
	# iQFT = inverse_qft(iQFT, n)
	# iQFT.update_quantum_state(m_state)

	# # check the integer 
	# int_out = m_state.sampling(1)
	# print(f"The output integer after QFT and iQFT is {int_out[0]}.")


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


test_QFT(6)

