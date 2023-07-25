from qulacs import QuantumState, QuantumCircuit, ParametricQuantumCircuit
from qulacs.gate import CZ, RY, RZ, merge, H, DenseMatrix
from qulacs import Observable
import numpy as np
from numpy import pi
import cmath


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
