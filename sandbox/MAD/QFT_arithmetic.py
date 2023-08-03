from qulacs import QuantumState, QuantumCircuit, ParametricQuantumCircuit
from qulacs.gate import CZ, RY, RZ, merge, H, DenseMatrix, SWAP
from qulacs import Observable
import pandas as pd
import numpy as np
from numpy import pi
import cmath
import matplotlib.pyplot as plt
import seaborn as sns

## PARAMETERS ## 
# maximum number of allowable qubits that can be initialized
max_qubits = 5

def qft(circuit,n):

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

def inverse_qft(circuit,n):
    circuit = qft(circuit,n)
    inverse_circuit = circuit.get_inverse()
    return inverse_circuit  


""" method that embeds in integer as the computational basis
	of an n-qubit state. """

def reverse(string):
    string = string[::-1]
    return string


""" creates circruit that adds k to m in the fourier basis state """
def add_k_fourier(k,circuit, n):
	# TODO :: check that k is an integer with the same bit size as m

	# get binary string representation of string

	# for each qubit in the m-fourier basis state
	for i in range(k):
		for x in range(n):
			gate = RZ(x, -np.pi/(2**(n-x-1)))
	        # add gate to circuit
			circuit.add_gate(gate)
	
	# return circuit to user
	return circuit


""" creates circruit that subtracts k to m in the fourier basis state """
def sub_k_fourier(k,circuit, n):
	# TODO :: check that k is an integer with the same bit size as m

	# for each qubit in the m-fourier basis state
	# for i in range(k):
	for x in range(n):
		gate = RZ(x, k * np.pi/(2**(n-x-1)))
        # add gate to circuit
		circuit.add_gate(gate)

	# return circuit to user
	return circuit

""" method that uses the quantum fourier transfer to add 
	two integers that are represented in the computational
	basis. """
def sum_QFT (m, k):

	# TODO :: set limits for m and k so that
	# too many qubits aren't initiated
	print(f"Adding {m} and {k}.")

	# embed m as n qubits via the computational basis

	n = 5
	state = QuantumState(n)
	state.set_zero_state()

	# place m in the Fourier basis 
	QFT = QuantumCircuit(n)

	QFT = qft(QFT, n)
	QFT.update_quantum_state(state)
	ck = QuantumCircuit(n)
	# embed k in m_state via QFT
	ck = add_k_fourier(m, ck, n)
	ck = add_k_fourier(k, ck, n)
	ck.update_quantum_state(state)

	# apply inverse fourier transform to the mk state
	iQFT = QuantumCircuit(n)
	iQFT = inverse_qft(iQFT, n)

	iQFT.update_quantum_state(state)
	int_out = state.sampling(10)
	b = '{:0{width}b}'.format(int_out[0], width=n)
	rev_int = int(b[::-1], 2)
	print(f"The output integer is {rev_int}.")


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

if __name__ == '__main__':
	sum_QFT(15,5)


