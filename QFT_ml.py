from qulacs import QuantumState, QuantumCircuit, ParametricQuantumCircuit
from qulacs.gate import CZ, RY, RZ, merge, H, DenseMatrix, SWAP
from qulacs import Observable
import pandas as pd
import numpy as np
from numpy import pi
import cmath
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.optimize import minimize
import matplotlib.pyplot as plt



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

""" creates circruit that adds k to m in the fourier basis state """
def add_k_fourier(k,circuit, n):
	# TODO :: check that k is an integer with the same bit size as m

	# for each qubit in the m-fourier basis state
	for x in range(n):
		# create RZ gate 
		gate = RZ(x, -k * np.pi / (2 ** (n-x-1)))
		# add gate to circuit
		circuit.add_gate(gate)

	# return circuit to user
	return circuit

def reverse(string):
    string = string[::-1]
    return string

def U_in(m,n):
     
    binary = format(m, 'b')
    binary = reverse(binary)   # Qulacs is arranged in a different bit format where the LSB is on the right
	
	# initialize n-qubit state

    c = QuantumCircuit(n)
	# for each integer in the binary string

    for x in range(len(binary)):
		# if the bit is one
        if binary[x] == '1':
			# add X gate to corresponding qubit
            c.add_X_gate(x)

    return c


def set_U_out(QFT_out, theta):
    parameter_count = QFT_out.get_parameter_count()
    for i in range(parameter_count):
        QFT_out.set_parameter(i, theta[i])   # Incrementing the 4th and 5th wires


# Function that gives prediction value y(x_i, theta) of the model from input x_i
def qcl_pred(U_out, data_x):

    n = 12
    state = QuantumState(n)
    state.set_zero_state()


    obs = Observable(n)
    obs.add_operator(2.,'Z 10')
    obs.add_operator(2.,'Z 11')

    #add_k_fourier(k,circuit, n):

    Initial_ciruit = U_in(data_x,n)

    # Calculate output state
    Initial_ciruit.update_quantum_state(state)

    U_out.update_quantum_state(state)

    # Output of the model
    res = obs.get_expectation_value(state)
    print(res)

    return res

def cost_func(theta,QFT_out, data_train):
    '''
    theta: ndarray of length c_depth * nqubit * 3
    '''
    # update the parameter theta of U_out

    
    y_train = np.array([2, 2, 2, 2, 2, 2,0 , 0])
    set_U_out(QFT_out,theta)

    # calculate basing on data of num_x_train in total
    y_pred = [qcl_pred(QFT_out, x) for x in data_train]

    # quadratic loss
    L = ((y_pred - y_train)**2).mean()
    #print(L)

    return L     



def ml_embedding(QFT_wires, data_train):
    
    
    QFT_out = ParametricQuantumCircuit(QFT_wires)
    QFT_out = qft(QFT_out, QFT_wires)

    #QFT_out.add_H_gate(6)
    #QFT_out.add_H_gate(7)

    for i in range(2):
        #QFT_out.add_parametric_RX_gate(i,2.0 * np.pi * np.random.rand())
        an = 0.392
        QFT_out.add_parametric_RZ_gate(10+i,an)
        QFT_out.add_parametric_RY_gate(10+i,1.392)

    parameter_count = QFT_out.get_parameter_count()
    theta_init = [QFT_out.get_parameter(ind) for ind in range(parameter_count)]
    


    #QFT_out.add_CNOT_gate(QFT_wires,QFT_wires+1)

    # Learning (takes 14 seconds with the writer's PC)
    result = minimize(cost_func, theta_init, args=(QFT_out, data_train), method='COBYLA')    

    theta_opt = result.x
    print(theta_opt)

    return theta_opt,QFT_out


n= 12

QFT_wires = 12
Control_wires = 2

# Training Data 
Data_train = [50, 56, 57, 60, 70, 72, 100, 120]

theta_opt,QFT_out = ml_embedding(QFT_wires, Data_train)

# Including the update theta in the prediction model


set_U_out(QFT_out,theta_opt)

res=[]
for i in range(2**12):
    state = QuantumState(12)
    state.set_zero_state()
    # Initialize as |01>
    #state.set_computational_basis(0b11001000)
    in_circuit = U_in(i,12)

    in_circuit.update_quantum_state(state)
    QFT_out.update_quantum_state(state)

    obs = Observable(n)
    obs.add_operator(2.,'Z 10')
    obs.add_operator(2.,'Z 11')
    # Output of the model
    a = obs.get_expectation_value(state)
    res.append(a)
    
plt.plot(res)
plt.show()