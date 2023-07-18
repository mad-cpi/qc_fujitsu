import sys, os
import math
import pandas as pd
import numpy as np
import qulacs
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from qulacs import QuantumState
from qulacs import QuantumCircuit
from qulacs import PauliOperator
from qulacs import Observable
from qulacs.gate import H, X, RX, RY, RZ, CNOT
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import seaborn as sns

## PARAMETERS ## 
# execute script verbosely
verbose = True


## FUNCTIONS ## 
""" load data for variational classifier training """
def load_data (dataset):
	
	if dataset == "parity_function":
		X, Y = load_parity_data()
	elif dataset == "activity_data":
		X, Y = load_activity_data()
	else:
		print(f"No instructions for loading DATASET ({dataset}).")
		return [], []

	if verbose: 
		print(f"\nLoading DATA ({dataset}) .. \n")
		for i in range(len(Y)):
			print("X = {}, Y = {:.2f}".format(X[i], Y[i]))

	return X, Y

""" instructions for loading data set that contains
	chemical structure and activity data """
def load_activity_data():
	# load data from file
	file_path = "./data/test_activity_set.csv"
	data = pd.read_csv(file_path)

	# load X array (bit vector)
	smi = data['SMILES'].tolist()
	# encode fingerprint bit string with as many bits as qubits
	fp = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), \
		radius = 3, nBits = n, useFeatures = True).ToBitString() for x in smi]

	# translate bit string to bit vector
	X = np.zeros((len(fp), n), dtype=int)
	for i in range(len(fp)):
		# print(fp[i])
		for j in range(n):
			if fp[i][j] == "1":
				# replace 0 with 1
				X[i,j] = 1

	# load Y array (activity)
	Y = np.array(data['single-class-label'])
	
	return X, Y


""" instructions for loading parity data set """
def load_parity_data ():
	file_path = "./data/parity_function.dat"
	data = np.loadtxt(file_path)
	# load X and Y arrays
	X = np.array(data[:, :-1])
	Y = np.array(data[:,  -1])
	# scale Y array to match range of Pali-Z expectation values
	Y = Y * 2 - np.ones(len(Y)) # shift label form [0, 1] to [-1, 1]

	return X, Y

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


""" initialize bit vector x into n-qubit state with
	m measurement wires. """
def qubit_encoding(x, n, m):

	# x :: bit vector with n elements, molecular fingerprint
	# n :: number of elements in x 
	# m :: number of measurement wires

	# create state with as many qubits as there bits
	# plus the number of measurement wires
	state = QuantumState(n + m)
	# intialize all qubits in "0" state
	state.set_zero_state()

	# build quantum circuit that changes quantum zero state
	# to quantum state representing bit vector
	# (all measurement bits remain |0> state)
	c = QuantumCircuit(n + m)
	# loop through bit vector
	for i in range(n):
		# if the bit is one
		if x[i] == 1:
			# add a not gate to the qubit corresponding
			# to that qubit (not gate flips qubit 
			# deterministically from 0 to 1)
			c.add_X_gate(i)

	# apply circuit to quantum state, return the state
	c.update_quantum_state(state)
	return state

""" initialize qubit state by encoding the amplitudes
	of the bit vector """
def amplitude_encoding():
	pass


""" circuit that models a TNN classification network
	DOI: https://doi.org/10.1038/s41534-018-0116-9"""
def TNNclassifier (w, x):

	# number of measurement wires
	m = 1
	# number of qubits
	n = len(x)
	# initialize bit vector into quantum quivalent
	state = qubit_encoding(n, m)

	# for each circuit layer
	l = len(W)
	for i in range(l):
		# generate quantum circuit with weights
		c = TNNcircuit(W[i])
		# apply circuit with weights to qubits
		c.update_quantum_state(state)

	# can probably make list a global variable 
	# (do not need to define each iteration of variational circuit)
	obs = Observable(n)
	# palui z operator measured on mth qubit, scale by constant 1
	obs.add_operator(1., 'Z 0')

	return obs.get_expectation_value(state)

""" Builds TNN quantum circuit TNN classification
	protocol. Applies unitary weights to unitary
	qubit operations, returns circuit. """
def TNNcircuit (w):

	# initialize circuit
	# number of qubits plus one (M)
	n = len(w) + 1
	circuit = QuantumCircuit(n)

	pass


""" creates an n-qubit circuit, where each qubit
	is rotated by an RX, RY, and RZ matrix, and 
	all neighboring qubits are connected by CNOT
	gates. """
def layer (w):

	# initialize circuit
	n = len(w)
	circuit = QuantumCircuit(n)

	# for each wire
	for i in range(n):
		# apply each R gate
		for j in range(3):
			if j == 0:
				# if j == 0, apply RX gate to wire n
				gate = RX (i, w[i][j])
			elif j == 1:
				# if j == 1, aply RY gate to wire n
				gate = RY (i, w[i][j])
			elif j == 2:
				# if j == 2, apply RZ gate to write n
				gate = RZ (i, w[i][j])

			# add gate to circuit
			circuit.add_gate(gate)

	# for each wire
	for i in range(n):
		# in the schulgin data set, the first 5 features
		# in the 16 bit fingerprint are all the same for the 
		# entire dataset, so do not apply control bits to those features
		if i >= 5 or i == (n-1):
			# target bit
			j = i + 1 
			if j >= n:
				j = 0
			# apply control bit to target bit
			gate = CNOT (i, j)
			# add to circuit
			circuit.add_gate(gate)

	return circuit

""" Creates prediction between x and y by:

	1. intializes n-qubit state that represents the 
	computational basis of the bit-vector x
	2. apply variational circuit layer n-times
	3. calculate the Pauli-Z expectation value 
	of the qubits after applying the circuit.
	"""
def variational_circuit(W, x):

	# number of qubits
	n = len(x)
	
	# initialize bit vector into quantum state
	state = stateprep(x, n, 0)

	# for each circuit
	l = len(W)
	for i in range(l):
		# generate quantum circuit with weights
		c = layer(W[i])
		# update quantum state with 
		c.update_quantum_state(state)
		# debug :: measure state distributions
		# sample_state_distribution(state, 1000)

	# can probably make list a global variable 
	# (do not need to define each iteration of variational circuit)
	obs = Observable(n)
	# palui z operator measured on 0th qubit, scale by constant 1
	obs.add_operator(1., 'Z 0')

	return obs.get_expectation_value(state)


""" defnes variational classifier, returns
	expectation value."""
def variational_classifier(W, b, x):
	## TODO :: CHECKS!! 
	return variational_circuit(W, x) + b


""" measure squared error  and accuray between 
	measurements and classification labels. """
def error (labels, predictions):
	# tolerance used to measure if a prediction is correct
	tol = 5e-2
	# initialize loss, accuracy measurements
	loss = 0.
	accuracy = 0
	# compare all labels and predictions
	for l, p in zip(labels, predictions):
		# print("CLASS : {:0.5f}, PRED : {:0.5f}".format(l, p))
		loss = loss + (l - p) ** 2
		if abs(l - p) < tol:
			accuracy = accuracy + 1
	# normalize loss, accuracy by the data size
	loss = loss / len(labels)
	accuracy = accuracy / len(labels)
	return loss, accuracy


""" calculates the error of weights used for variational
	classifier circuit against training set. """
def cost_function (W):

	global n_it
	global thershold

	# increament threshold every x iterations
	if n_it % n_inc == 0:
		print(f"Threshold increased from {thershold} to {(thershold + thershold_it)}.")
		thershold += thershold_it

	# remap weights
	b = W[-1] # last value corresponds to bias
	W = W[:-1].reshape(l, n ,3) 
	# everything else are for gate operations

	# make random testing batch
	batch_index = np.random.randint(0, high = len(X), size = n_batch)
	X_batch = X[batch_index]
	Y_batch = Y[batch_index]

	# make predictions for all X values
	# if values are above the treshhold
	# assume the predctions are correct
	Y_pred = []
	for i in range(len(X_batch)):
		x = X_batch[i]
		y = variational_classifier(W, b, x)
		if abs(y) > thershold:
			y = np.sign(y)
		Y_pred.append(y)
	# Y_pred = [np.sign(variational_classifier(W, b, x)) for x in X]
	# Y_pred = [variational_classifier(W, b, x) for x in X]

	# measure and return loss of predictions against expected values
	n_it = n_it + 1
	cost, acc = error(Y_batch, Y_pred)
	if verbose:
		# inform user of the cost and error of n iterations
		print("Iteration: {:5d} | Cost: {:0.5f} | Accuracy : {:0.5f}"\
			.format(n_it, cost, acc))

	return cost

## ARGUMENTS ## 
# number of qubits in quantum circuit
n = 16
# number of layers in quantum circuit architechture
l = 2

## HYPERPARAMETERS ## 
n_it = 0 # number of times that the cost function is called
thershold = 0. # value used to coarse grain binary results
n_inc = 25 # number of iteractions required for threshold incrementation
thershold_it = 0.05 # amount to increase threshold by every n_inc iterations
batch_size = 0.8 # precentage of data used to create predictions


## SCRIPT ##

X, Y = load_data(dataset = "activity_data")
n_batch = math.ceil(len(X) * batch_size)
# number of data points in each batch

# create array of random weights
W_init = np.random.normal(0., 0.1, size=(l * n * 3 + 1)) 
bnds = [] # bounds of weights
# initialized as N x 2 array
for i in range(len(W_init)):
	if i < (len(W_init) - 1):
		lwr_bnd = - 2. * math.pi # lower bound of each universal unitary opration
		upp_bnd = 2. * math.pi # upper bound of each universal unitary operation
		bnds.append((lwr_bnd, upp_bnd))
	else:
		# the final value in the set is the bias applied to each circuit
		# no bounds on linear shift
		bnds.append((None, None))

# print(cost_function(W_init))
opt = minimize (cost_function, W_init, method = 'Powell', bounds = bnds)

# optimal weight
W_opt = opt.x
print("\nFinal value of error function after optimization: {:0.3f}.".format(opt.fun))

# TODO :: SAVE OPTIMAL WEIGHTS!



