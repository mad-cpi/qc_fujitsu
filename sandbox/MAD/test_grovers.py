from qulacs import QuantumState, QuantumCircuit, ParametricQuantumCircuit
from qulacs.gate import CZ, RY, RZ, merge, H, DenseMatrix
from qulacs import Observable
import pandas as pd
import numpy as np
from numpy import pi
import math
import matplotlib.pyplot as plt
import seaborn as sns

## PARAMETERS ## 
# maximum number of allowable qubits that can be initialized
max_qubits = 5

def qft(circuit,n):
    for hadam in range(n):
        n = n-1
        circuit.add_gate(H(n)) # Apply the H-gate to the most significant qubit
        for qubit in range(n):
        
            den_gate = DenseMatrix([qubit,n],[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,math.exp(pi/2**(n))]])
            circuit.add_gate(den_gate)

    return circuit

def inverse_qft(circuit,n):
    for hadam in range(n):
        n = n-1
        circuit.add_gate(H(n)) # Apply the H-gate to the most significant qubit
        for qubit in range(n):
            den_gate = DenseMatrix([qubit,n],[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,math.exp(pi/2**(n))]])
            circuit.add_gate(den_gate)

    inverse_circuit = circuit.get_inverse()
    return inverse_circuit  


""" method that embeds in integer as the computational basis
	of an n-qubit state. """
def integer_basis_embedding (m):

	# get binary string representation of string
	binary = format(m, 'b')
	
	# initialize n-qubit state
	n = len(binary)
	# check that the length is not above the allowable limit
	if n > max_qubits:
		print(f"Number of qubits N ({n}) is above the allowable limit for a desktop computer.")
		exit()
	state = QuantumState(n)
	state.set_zero_state()

	# initialize circuit used to set qubit state in
	# corresponding computational basis
	c = QuantumCircuit(n)
	# for each integer in the binary string
	for x in range(len(binary)):
		# if the bit is one
		if binary[x] == '1':
			# add X gate to corresponding qubit
			c.add_X_gate(x)

	# set quantum state in computational basis, return to user
	c.update_quantum_state(state)
	return state, n

""" creates circruit that adds k to m in the fourier basis state """
def add_k_fourier(k, n):
	# TODO :: check that k is an integer with the same bit size as m

	c = QuantumCircuit(n)
	# for each qubit in the m-fourier basis state
	for x in range(n):
		# create RZ gate 
		gate = RZ(x, k * np.pi / (2 ** x))
		# add gate to circuit
		c.add_gate(gate)

	# return circuit to user
	return c

""" method that uses the quantum fourier transfer to add 
	two integers that are represented in the computational
	basis. """
def sum_QFT (m, k):

 	# TODO :: set limits for m and k so that
 	# too many qubits aren't initiated

 	# embed m as n qubits via the computational basis
 	mk_state, n = integer_basis_embedding(m)
 	sample_state_distribution(mk_state, 1000)

 	# place m in the Fourier basis 
 	QFT = QuantumCircuit(n)
 	QFT = qft(QFT, n)
 	QFT.update_quantum_state(mk_state)
 	# sample_state_distribution(mk_state, 1000)

 	# embed k in m_state via QFT
 	ck = add_k_fourier(k, n)
 	ck.update_quantum_state(mk_state)
 	# sample_state_distribution(mk_state, 1000)

 	# apply inverse fourier transform to the mk state
 	iQFT = QuantumCircuit(n)
 	iQFT = inverse_qft(iQFT, n)
 	iQFT.update_quantum_state(mk_state)
 	sample_state_distribution(mk_state, 1000)


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

sum_QFT(23, 2)