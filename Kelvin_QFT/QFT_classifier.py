# Class for the QFT classifier which implements the QFT circuit along with the training and predictor functions
# the model is atrue Quantum computing algortihm demonstrating quantum supremacy 
#
# Author: Kelvin Dsouza
# 

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


# Inputs will be the 
# Qubits : Number of qubits to be used in the classifier model
# Train: Whether the data will be used to train the data or to predict it 
# restart = True - if restart is true then a new training set will be calculated 
# Method - choosing which algorithm will be used to train the model 
# QFT_res - No. of bits of the qft to be used to train and predict the model 
# Gate - pauli's gate to determine the observable  default - Z 


class QFT_classifier:
    def __init__(self,qubits,train=False, restart = False):
        self.qubits= qubits
        # Training Data 

    
    def classifier_train(self,x_train, y_train):
        theta_opt = self.ml_embedding(x_train,y_train)
        return theta_opt


    def QFT_predictor(self,theta_opt,data_pred):
        state = QuantumState(self.qubits)
        state.set_zero_state()

        in_circuit = self.U_in(data_pred)
        in_circuit.update_quantum_state(state)

        QFT_out = ParametricQuantumCircuit(self.qubits)
        QFT_out = self.qft(QFT_out)

        #QFT_out.add_parametric_RX_gate((self.qubits-2)+i,2.0 * np.pi * np.random.rand())
        for i in range(3):
            QFT_out.add_parametric_RZ_gate(self.qubits-4+i,2.0 * np.pi * np.random.rand())
            QFT_out.add_parametric_RY_gate(self.qubits-4+i,2.0 * np.pi * np.random.rand())

        
        self.set_U_out(QFT_out, theta_opt)

        QFT_out.update_quantum_state(state)

        obs = Observable(self.qubits)
        
        obs.add_operator(2.,'Z 7')
        obs.add_operator(2.,'Z 8')
        obs.add_operator(2.,'Z 9')
        # Output of the model
        a = obs.get_expectation_value(state)

        return a
    
    # Function to initiate the QFT on a circuit 

    def qft(self,circuit):
        n = self.qubits
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

    # function to calculate the inverse QFT 

    def inverse_qft(self,circuit,n):
        circuit = self.qft(circuit,n)
        inverse_circuit = circuit.get_inverse()
        return inverse_circuit  

    # function to initialize the circuit to a set value 
    def reverse(self, string):
        string = string[::-1]
        return string


    def U_in(self, data_in):
     
        binary = format(data_in, 'b')
        binary = self.reverse(binary)   # Qulacs is arranged in a different bit format where the LSB is on the right
        #print(binary)
	    # initialize n-qubit state
        c = QuantumCircuit(self.qubits)
	    # for each integer in the binary string

        for x in range(len(binary)):
		    # if the bit is one
            if binary[x] == '1':
			    # add X gate to corresponding qubit
                c.add_X_gate(x)

        return c


    def set_U_out(self,QFT_out, theta):
        parameter_count = QFT_out.get_parameter_count()
        for i in range(parameter_count):
            QFT_out.set_parameter(i, theta[i])   # Incrementing the 4th and 5th wires


    # Function that gives prediction value y(x_i, theta) of the model from input x_i
    def qcl_pred(self,U_out, data_x):

        n = self.qubits
        state = QuantumState(n)
        state.set_zero_state()

        obs = Observable(n)
        obs.add_operator(2.,'Z 7')
        obs.add_operator(2.,'Z 8')
        obs.add_operator(2.,'Z 9')

        Initial_ciruit = self.U_in(data_x)

        # Calculate output state
        Initial_ciruit.update_quantum_state(state)

        U_out.update_quantum_state(state)

        # Output of the model
        res = obs.get_expectation_value(state)

        return res

    def cost_func(self,theta,QFT_out, x_train, y_train):
        '''
        theta: ndarray of length c_depth * nqubit * 3
        '''
        # update the parameter theta of U_out

        y_train = 2*np.array(y_train)
        self.set_U_out(QFT_out,theta)

        # calculate basing on data of num_x_train in total
        y_pred = [self.qcl_pred(QFT_out, x) for x in x_train]

        # quadratic loss
        L = ((y_pred - y_train)**2).mean()

        return L     

                                                    
    def ml_embedding(self, x_train,y_train):
     
        QFT_out = ParametricQuantumCircuit(self.qubits)
        QFT_out = self.qft(QFT_out)

        for i in range(3):
            QFT_out.add_parametric_RZ_gate(self.qubits-4+i,2.0 * np.pi * np.random.rand())
            QFT_out.add_parametric_RY_gate(self.qubits-4+i,2.0 * np.pi * np.random.rand())
   
        parameter_count = QFT_out.get_parameter_count()
        theta_init = [QFT_out.get_parameter(ind) for ind in range(parameter_count)]
    
        # Minimizing the cost function based on the algorithm 

        result = minimize(self.cost_func, theta_init, args=(QFT_out, x_train, y_train), method='Nelder-Mead')    

        theta_opt = result.x

        # Averaging the expectation 

        self.set_U_out(QFT_out,theta_opt)
        y_pred = [self.qcl_pred(QFT_out, x) for x in x_train]
        Ypred_avg = np.mean(y_pred)

        return theta_opt, Ypred_avg




