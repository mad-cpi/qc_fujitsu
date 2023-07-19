from qulacs import QuantumState, QuantumCircuit, ParametricQuantumCircuit
from qulacs.gate import CZ, RY, RZ, merge, H, DenseMatrix
from qulacs import Observable
import numpy as np
from numpy import pi
import math


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

n = 4  # Number of Qubits 
circuit = QuantumCircuit(n)
qft_circuit = qft(circuit,n)
inv_circuit = inverse_qft(circuit,n)
state = QuantumState(n) 
#state.set_zero_state()
state.set_Haar_random_state() # Generating a random inital state 
print(state.get_vector())

qft_circuit.update_quantum_state(state)
print(state.get_vector())

inv_circuit.update_quantum_state(state)
print(state.get_vector())    # checking if the same input is received 
