import sys, os
import math, cmath
import pandas as pd
import numpy as np
from numpy import pi
from qulacs import QuantumState, QuantumCircuit, Observable
from qulacs.gate import RX, RY, RZ, CNOT, H, DenseMatrix, SWAP
from abc import abstractmethod, ABC
import matplotlib.pyplot as plt
import seaborn as sns
from mpi4py import MPI

## PARAMETERS ## 
# maximum number of qubits that can used to initialize a circuit on
# a desktop computer. If the number of qubits passed to a circuit is 
# greater than this number, the circuit will attempt to import mpi4py,
# which is only avalible on the quantum simulator
max_desktop_qubits = 24
# average of random numbers generation for initial unitary weights
norm_avg = 0.
# standard deviation of random numbers generated for initial unitary weights
norm_std = 0.1
# upper boundary of each universal unitary operation
upper_unitary_boundary = 2. * math.pi 
# lower boundary of each universal unitary operation
lower_unitary_boundary = -2. * math.pi
# default QFT status, determine if QFT is applied to
# quantum state in computational basis before applying
# variational circuit layers / uinitary transormations
default_QFT_status = False

## METHODS UTILIZEDS BY CIRCUIT ARCHITECTURE ##

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

""" method that initializes unitatary weights randomly along a normal distrubution,
	with boundaries that correspond to the upper and lower period boundaries of any
	qubit rotation (last weight in list corresponds to the bias used to shift result
	of classification circuit liniarly, which is used for all classification circuits). """
def random_unitary_weights (n_weights, avg = norm_avg, std = norm_std):
	# initialize weights randomly along a normal distribution
	weights = np.random.normal(avg, std, size=n_weights)

	# assign boundaries to each weight 
	# (last weight in list correspond to linear shift, 
	#	and is assigned different boundaries)
	bounds = [] # bounds of weights
	# initialized as N x 2 array
	for i in range(len(weights)):
		if i < (len(weights) - 1):
			bounds.append((lower_unitary_boundary, upper_unitary_boundary))
		else:
			# the final value in the set is the bias applied to each circuit
			bounds.append((-1, 1))

	return weights, bounds

""" method for getting the state vector of a quantum state which has been operated
	on by a quantum circuit. The way that the program retrieves the state vector depends
	on which MPI has been used to simulate the quantum state."""
def get_state_vector(state):
    if state.get_device_name() == 'multi-cpu':
        mpicomm = MPI.COMM_WORLD
        mpisize = mpicomm.Get_size()
        vec_part = state.get_vector()
        len_part = len(vec_part)
        vector_len = len_part * mpisize
        vector = np.zeros(vector_len, dtype=np.complex128)
        mpicomm.Allgather([vec_part, MPI.DOUBLE_COMPLEX],
                          [vector, MPI.DOUBLE_COMPLEX])
        return vector
    else:
        return state.get_vector()

""" method for loading the state vector into a qulacs quantum state. The way that the
	quantum state should be loaded deopends on the whether the quantum state is integrated 
	with MPI. """
def load_state_vector(state, vector):
    """
    Loads the given entire state vector into the given state.

    Args:
        state (qulacs.QuantumState): a quantum state
        vector: a state vector to load
    """
    if state.get_device_name() == 'multi-cpu':
        mpicomm = MPI.COMM_WORLD
        mpirank = mpicomm.Get_rank()
        mpisize = mpicomm.Get_size()
        vector_len = len(vector)
        idx_start = vector_len // mpisize * mpirank
        idx_end = vector_len // mpisize * (mpirank + 1)
        state.load(vector[idx_start:idx_end])
    else:
        state.load(vector)

## QUBIT ENCODING METHODS ##	

""" method the reverses the order of a bit string
	(in qulacs, the rightmost bit corresponds to
	the zeroth qubit). """
def reverse(s):
    s = s[::-1]
    return s

""" method that returns circuit used to initialize qubit state in 
	computational basis corresponding to bit string passed to the method. """
def basis_encoding_circuit (x):

	# reverse bit string due to qulacs nomenclature
	# (in qulacs, 0th bit is represented by the right now digit,
	# not the leftmost)
	x = reverse(x)

	# get the number of qubits in the string, initialize quantum state
	# in the computationa zero basis
	n = len(x)

	# initialize circuit used to set qubit state in
	# corresponding computational basis
	c = QuantumCircuit(n)
	# for each integer in the binary string
	for i in range(len(x)):
		# if the bit is one
		if x[i] == 1:
			# add X gate to corresponding qubit
			c.add_X_gate(i)

	# return the circuit to the user
	return c

""" method used to return the circuit that initializes a qubit state as fourier basis 
	corresponding to bit string passed to the method. """
def QFT_encoding_circuit(x):

	# embed x as the computational basis
	n = len(x)
	s = basis_encoding(x) # TODO :: change to basis encoding circuit

	# convert the bit string to an integer, embed the integer in the fourier basis
	k = s.sampling(1)
	cQFT = QuantumCircuit(n)
	cQFT = qft_init(cQFT, n)
	cQFT = add_k_fourier(k[0], cQFT, n)

	# return the circuit to the user
	return cQFT

""" method that generates a circuit, which translates an n-qubit state which is initially in
	the zero computational basis to a the zero fourier basis."""
def qft_init(circuit,n):

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

""" creates circruit that adds k to m in the fourier basis state """
def add_k_fourier(k, circuit, n):
	# TODO :: check that k is an integer with the same bit size as m

	# get binary string representation of string

	# for each qubit in the m-fourier basis state
	for x in range(n):
		gate = RZ(x, k * np.pi/(2**(n-x-1)))
        # add gate to circuit
		circuit.add_gate(gate)
	
	# return circuit to user
	return circuit

""" method for encoding bit-string in initial quantum state as wave amplitudes """
def amp_encoding (x):

	# TODO :: there would be a few ways to speed this method up
	# 	- storing amplitudes, rather than calculating them each iteration
	# 	- use np.arrays to quick calculate the norm

	# check that the length of the bit string passed to the method matches
	# the appropriate length based on the number of qubits in the circuit
	vec_len = len(x)
	n = int(math.log(vec_len, 2))

	# calculate the norm of the bit string
	norm = 0.
	for i in x:
		norm += i ** 2
	norm = math.sqrt(norm)

	# initialize the numpy array that will contain the amplitude encoded
	# store the amplitude normilized state vector
	sv = np.array([])
	for i in x:
		sv = np.append(sv, [i / norm])

	# # initialize the qubits using the state vector
	# state = QuantumState(n)
	# state.load(a)

	return sv

""" TODO :: add method to quantifying qubit circuits (?? forgot what I meant by this) """


## CIRCUIT ARCHITECTURE CLASS METHODS ## 
class ClassificationCircuit (ABC):
	def __init__(self, qubits, state_prep_method):
		self.set_circuit_qubits(qubits)
		self.state_prep = state_prep_method
		self.set_layers() # number of layers depends on the circuit architecture
		self.set_observable()
		self.set_QFT_status() # initialize QFT operation, default is off

	## SETTERS AND GETTERS ##

	# set the number of circuit qubits
	def set_circuit_qubits(self, qubits):

		if qubits > max_desktop_qubits:

			# import mpi4py, check rank
			comm = MPI.COMM_WORLD
			rank = (comm.Get_rank() + 1)
			print("\nInitializing ")

			# TODO :: there is definitely a better way to perform
			# this logical operation

			# use rank to determine how many qubits can be initialized
			MPI_err = False # boolean that determines if there is an error
			# between the number of qubits that the circuit is simulating, 
			# and the number of compute nodes the program has access to
			if qubits <= 30:
				# in order to simulation at least 30 qubits
				if rank < 1:
					# the program must at least have access to one compute node
					MPI_err = True
			elif qubits <= 31:
				# in order to simulate at least 31 qubits
				if rank < 2:
					# the program must have access to at least two compute nodes
					MPI_err = True
			elif qubits <= 32:
				# in order to simulate at least 32 qubits
				if rank < 4:
					# the program must have access to at least four compute nodes
					MPI_err = True
			elif qubits <= 33:
				# in order to simulate 33 qubits
				if rank < 8:
					# the program must have access to at least eight compute nodes
					MPI_err = True
			elif qubits <= 34:
				# in order to simulate 34 qubits
				if rank < 16:
					# the program must have access to at least sixteen compute nodes
					MPI_err = True
			elif qubits <= 35:
				# in order to simulation 35 qubits
				if rank < 32:
					# the program must have access to at least thiry-two compute nodes
					MPI_err = True
			elif qubits <= 36:
				# in order to simulate 36 qubits
				if rank < 64:
					# the program must have access to at least sixty-four qubits
					MPI_err = True
			elif qubits <= 37:
				# in order to simulate 37 qubits
				if rank < 128:
					# the program must have access to at least one hundred and twenty eight compute nodes
					MPI_err = True
			elif qubits <= 38:
				# in order to simulate 38 qubits
				if rank < 256:
					# the program must have access to at least two hundred and fifty-six qubits
					MPI_err = True
			elif qubits <= 39:
				# in order to simulate 39 qubits
				if rank < 512:
					# the program must have access to at least five hundrd and twelve qubits
					MPI_err = True
			else:
				# impossible to simulation any greater number of qubits
				MPI_err = True

			# if an error has occured, inform the user and exit
			if MPI_err:
				print(f"ERROR :: Attempting to intialize {qubits} qubits with only {rank} compute nodes.")
				exit()
			else:
				# if no error has occured, assign the qubits to the circuit
				self.qubits = qubits
				self.use_MPI = True


		else:

			# assign the qubits to the circuit and proceed
			self.qubits = qubits
			self.use_MPI = False

	## generic circuit architecture circuit methods

	""" method for initializing the QFT status."""
	def set_QFT_status(self, status = False):

		# initialize the QFT status
		if status == False or status == True:
			self.QFT_status = status 
		else:
			self.QFT_status = default_QFT_status

	""" method that translates an array of bit strings into array of vectors that describe the 
		initial qubit state. Vectors can be used to initialize the qubit state directly, rather
		than translating a bit string to a quantum state iterativly on each classify call. The 
		state vector that is returned depends on the state prep method that is assigned to the 
		circuit. Method returns an array of state vectors."""
	def batch_state_prep(self, X):

		# for each bit string
		SV = []
		for x in X:
			# initialize the circuit that sets the qubit state
			if self.state_prep ==  "QFT_encoding":
				# get quantum circuit
				c = QFT_encoding_circuit(x)
				# initialize qunautm state
				if self.use_MPI:
					s = QuantumState(self.qubits, use_multi_cpu = True)
				else:
					s = QuantumState(self.qubits)
				s.set_zero_state()
				c.update_quantum_state(s)
				sv = get_state_vector(s)
				SV.append(sv)
			elif self.state_prep == "AmplitudeEmbedding":
				# amplitude encoding returns the state automatically
				sv = amp_encoding(x)
				SV.append(sv)
			else:
				# default is embed bit string as computational basis
				c = basis_encoding_circuit(x)
				# initialize the quantum state
				if self.use_MPI:
					s = QuantumState(self.qubits, use_multi_cpu = True)
				else:
					s = QuantumState(self.qubits)
				s.set_zero_state() # initialize the zero state
				c.update_quantum_state(s) # update the quantum state
				# sv = get_state_vector(s) # return the state vector from the parsing function
				sv = s.get_vector()
				SV.append(sv) # add the state vector to the list
		# return the array to the user
		return SV


	""" method used by all classification circuits. weights passed to method are used to make
		predictions / classifications for an array of bit strings passed to the method. """
	def classifications (self, Wb, state_vectors = None, bit_strings = None):

		# remove bias from list of weights
		b = Wb[-1]
		W = Wb[:-1]

		# if state vectors were not passed to the method
		if state_vectors is None and bit_strings is None:
			# must pass at least one two method
			print("Must pass either array of state vectors or bit strings to method.")
			exit()
		elif state_vectors is None:
			# convert the bit strings to state vectors
			state_vectors = self.batch_state_prep(bit_strings)
		# otherwise, an array of state vectors were passed to the method

		# initialize a circuit for each layer in the circuit architecture
		C = []
		for i in range(self.layers):
			C.append(self.layer(W, i))

		# exit()
		# for each state vector in the list, make a prediction
		predictions = []
		for sv in state_vectors:
			# initialize the quantum state
			if self.use_MPI:
				s = QuantumState(self.qubits, use_multi_cpu = True)
			else:
				s = QuantumState(self.qubits)

			# load the state vector into the quantum state
			# load_state_vector(s, sv)
			s.load(sv)

			# apply the circuit operations to the initial quantum state
			for c in C:
				c.update_quantum_state(s)

			# once the quantum state has been updated
			# measure the qubit states and add the prediction to the list
			predictions.append(self.obs.get_expectation_value(s) + b)

		# return the predictions to the user
		return predictions


	""" methed used by all classification circuits. weights passed to method (W) 
		are used to make prediction / classification for a single bit strings or
		state vector representing the bit strings passed to method. """
	def classify(self, Wb, state_vector = None, bit_string = None):

		# remove bias from list of weights
		b = Wb[-1]
		W = Wb[:-1]

		# load initial quantum state
		if state_vector is None and bit_string is None:
			# must specify either the state vector or the bit string
			print (f"ERROR :: Ansatz prediction requires either bit string or state_vector.")
			exit()
		elif state_vector is None:
			# initialize qubit state using the bit string
			if self.state_prep ==  "QFT_encoding":
				# embed string as fourier basis
				state = QFT_encoding_circuit(bit_string)
			elif self.state_prep == "AmplitudeEmbedding":
				# embed state as amplitudes
				state = amp_encoding(bit_string)
			else:
				# default is embed bit string as computational basis
				c = basis_encoding_circuit(bit_string)
				# initialize the quantum state
				if self.use_MPI:
					state = QuantumState(self.qubits, use_multi_cpu = True)
				else:
					state = QuantumState(self.qubits)
				state.set_zero_state() # initialize the zero state
				c.update_quantum_state(state) # update the quantum state
				# sv = get_state_vector(s) # return the state vector from the parsing function
				# sv = state.get_vector()
		else:
			# use the state vector to load the initial qubit state
			# initialize the state
			if self.use_MPI:
				state = QuantumState(self.qubits, use_multi_cpu = True)
			else:
				state = QuantumState(self.qubits)
			# load the state vector into the quantum state
			# load_state_vector(state, state_vector)
			state.load(state_vector)

		# apply circuit layers to quantum state
		for i in range(self.layers):
			# generate a quantum circuit with the specified weights
			c = self.layer(W, i)
			# apply circuit, update quantum state
			c.update_quantum_state(state)

		# make prediction based on expectation value, return to user
		return self.obs.get_expectation_value(state) + b

	""" method that sets the number of layers in the circuit, which is unique
		depending on the circuit architecture. """
	@abstractmethod
	def set_layers(self):
		pass

	""" method the sets the qubit whose state is observed after circuit layers
		have been applied to initial qubit state. """
	@abstractmethod
	def set_observable(self):
		pass

	""" method used by circuit initialize weights of unitary operations """
	@abstractmethod
	def initial_weights(self):
		pass

	""" method returns the weights used for a certain circuit layer,
		specified by i. """
	@abstractmethod
	def get_layer_weights(self, W, i):
		pass

	""" method that returns the wires that should be operated on by the 
		circuit layer corresponding to i """
	@abstractmethod
	def get_layer_qubits(self, i):
		pass

	""" method that describes the circuit applied to quantum circuit
		which contains operations for single layer of classification circuit """
	@abstractmethod
	def layer(self, W, i):
		pass


class VariationalClassifier(ClassificationCircuit):

	# implementation inspired by: 
	# https://pennylane.ai/qml/demos/tutorial_variational_classifier
	# Schuld, Maria, et al. "Circuit-centric quantum classifiers." Physical Review A 101.3 (2020): 032308.

	""" set the number of layers for the variational classification
		circuit """
	def set_layers(self):
		self.layers = 2

	""" set observable for VC circuit """
	def set_observable (self):
		self.obs = Observable(self.qubits)
		self.obs.add_operator(1, 'Z 0')

	""" method that initializes the weights of a circuit according
		to the circuit architecture, number of layers, qubits, etc. """
	def initial_weights(self):

		# determine the number of weights that should be initialized for the circuit
		n_weights = self.layers * self.qubits * 3 + 1

		return random_unitary_weights(n_weights = n_weights)

	""" given a list of weights for the entire circuit, returns an array of weights
		corresponding to the circuit layer specified by i"""
	def get_layer_weights(self, W, i):

		# reshape list containing layer weights
		w = W.reshape(self.layers, self.qubits, 3) # reshape according to number of layers
		# return the weights corresponding to the layer to the user
		return w[i]

	""" get qubits that are operated on by layer of variational classification
		circuit, which for this architecture is all qubits. """
	def get_layer_qubits (self, i):
		q = [x for x in range(self.qubits)]
		return q

	""" build circuit that based on weights passed to method. """
	def layer(self, W, i):
		# initialize circuit
		n = self.qubits
		circuit = QuantumCircuit(n)

		# get the circuit weights from the list
		w = self.get_layer_weights(W, i)

		# for each wire
		for j in range(n):
			# apply each R gate
			for k in range(3):
				# if j == 1:
				# 	print(f"LAYER ({i}), QUBIT ({j}), GATE ({k}), WEIGHT ({w[j][k]})")
				if k == 0:
					# if j == 0, apply RX gate to wire n
					gate = RX (j, w[j][k])
				elif k == 1:
					# if j == 1, aply RY gate to wire n
					gate = RY (j, w[j][k])
				elif k == 2:
					# if j == 2, apply RZ gate to write n
					gate = RZ (j, w[j][k])

				# add gate to circuit
				circuit.add_gate(gate)

		# for each wire
		for i in range(n):
			# in the schulgin data set, the first 5 features
			# in the 16 bit fingerprint are all the same for the 
			# entire dataset, so do not apply control bits to those features
			# if i >= 5 or i == (n-1):
			# target bit
			j = i + 1 
			# periodic wrap at boundaries
			if j >= n:
				j = 0
			# apply control bit to target bit
			gate = CNOT (i, j)
			# add to circuit
			circuit.add_gate(gate)

		return circuit


class TreeTensorNetwork(ClassificationCircuit):

	# implementation inspired by: 
	# https://www.nature.com/articles/s41534-018-0116-9
	# Grant, Edward, et al. "Hierarchical quantum classifiers." npj Quantum Information 4.1 (2018): 65.

	""" set the number of layers for the TTN classification circuit,
		which depends on the number of qubits in the circuit. """
	def set_layers(self):

		# check the number of qubits that should be 
		# input used into the quantum circuit
		n = self.qubits
		fact = 2
		layer = 1
		is_valid = False
		if (n != 0):
			# the number of qubits must be non-zero
			# check that n is an exact factor of two
			while True:
				if n == fact: 
					# the number is exactly factorable by two
					is_valid = True
					break
				elif (fact < n):
					# continue to increase factor by two 
					fact = fact * 2
					layer += 1
				else:
					# i is greater than n,
					# and therefore not an exact factor of 2
					is_valid = False
					break


		# check that the number of qubits assligned to the circuit is correct
		if (not is_valid):
			print(f"\nError :: number of qubits ({self.qubits}) assigned to the circuit must by an integer of log2(N).")
			exit()

		# assign the layers to the object
		self.layers = layer

	""" set observable to TTN circuit """
	def set_observable(self):
		self.obs = Observable(self.qubits)
		# the qubit which is obserbed depends on the overall number of qubits
		operator = 'Z {:d}'.format(10) # TODO :: this is hard coded
		self.obs.add_operator(1., operator)

	""" intialize weights of classification circuit according to the TTN circuit
		architecture. """
	def initial_weights(self):

		# initialize the weights used for the unitary operations
		# the number of weights depends on the number of layers in the circuit
		# determine the number of weights
		n_weights = 0
		for l in range(self.layers):
			n_weights += 3 * int(self.qubits / (2 ** l))
		n_weights += 1

		# initialize the weights randomly, return to user
		return random_unitary_weights(n_weights = n_weights)

	""" method that reshapes list of unitary weights into an array that is organized
		by the operations that are performed for each layer of the classification circuit. """
	def get_layer_weights(self, W, i):	

		w = np.zeros((1))
		# initialize index lower boundary for weights in list
		lw = 0
		# loop through all layers until at final layer
		for j in range(i + 1):
			# determine the number of weights that correspond to the layer
			n_weights = int(self.qubits / (2 ** (j)))
			# set the index corresponding to the upper layer of the weight
			up = lw + 3 * n_weights
			# reshape the list indicies corresponding the layer to a 3 x N array
			w = W[lw:up].reshape(n_weights, 3)
			# assign the lower boundary for the next layer as the upper boundary
			# for the current layer
			lw = up

		# return the layer corresponding to i to the user
		return w

	""" builds circuit that corresponds to a series of unitary operations, which is used to update
		the quantum state passed through the classification circuit ."""
	def layer (self, W, i):

		w = self.get_layer_weights(W, i)
		q = self.get_layer_qubits(i)

		# NOTE :: len(q) and len(W) should be equal
		if len(w) != len(q):
			print(f"Length of TTN layer weights ({len(w)}) and layer qubits ({len(q)}) are not equal.")
			exit()

		# initialize circuit for n qubits
		circuit = QuantumCircuit(self.qubits)

		# apply unitary operations for each wire in q
		for j in range(len(q)):
			# apply each R gate
			for k in range(3):
				if k == 0:
					# if j == 0, apply RX gate to wire n
					gate = RX (q[j], w[j][k])
				elif k == 1:
					# if j == 1, aply RY gate to wire n
					gate = RY (q[j], w[j][k])
				elif k == 2:
					# if j == 2, apply RZ gate to write n
					gate = RZ (q[j], w[j][k])

				# add gate to circuit
				circuit.add_gate(gate)

		# apply CNOT gates for each wire in q with its neighbor (in the list)
		for j in range(0, int(len(q) / 2), 2):
			gate = CNOT (q[j], q[j+1])
			# add gate to circuit
			circuit.add_gate(gate)

		return circuit

	""" get layer qubits for TTN circut architecture """
	def get_layer_qubits (self, i):

		# initialize the list of wires that interact with one another
		# start with list of all cubits
		q = [x for x in range(self.qubits)]

		if i > 0:
			# if not the first layer
			for l in range(i):
				# for each layer
				# remove every other qubit from the list
				# start with empty array
				q_new = []
				for j in range(int(len(q) / 2)):
					if j % 2 == 0:
						q_new.append(q[2 * j + 1])
					else:
						q_new.append(q[2 * j ])
				# update array of qubits with hald of qubits removed
				q = q_new

		return q



