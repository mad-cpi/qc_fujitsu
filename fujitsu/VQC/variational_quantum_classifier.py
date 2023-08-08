import sys, os
import pandas as pd
import numpy as np
import math
# rdkit libraries
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
from scipy.optimize import minimize
# VQC circuit architectures
from fujitsu.VQC.circuit_architecture import VariationalClassifier, TreeTensorNetwork

## PARAMETERS ## 

## VQC class default parameters used for initialization ##
# TODO :: add list of acceptable circuit architectures

# max number of qubits that can be assigned
# to the classification circuit (for safety,
# on any desktop this number shouldnt be greater than 20)
max_qubits = 20

# minimum number of qubits that can be assigned
# to any classification circuit
min_qubits = 1

# default type of circuit used for VQC
default_VQC_circuit = 'VC'

# default method used to encode the qubit state
default_state_prep_method = 'BasisEmbedding'


## fingerprinting specifications ## 
# TODO :: add list of acceptable fingerprints

# default fingerprint type used for encoding 
# smile strings to bit vectors
default_fp_type = "rdkfp"

# default connectivity radius used for encoding 
# smile strings to bit vectors
default_fp_radius = 3



## TODO :: add VQC exception

## VQC OPTIMIZATION METHODS ##

""" method used to optimize the current weights of a variational circuit according
	to the dataset sotored within the variational classification circuit
	object. """
def optimize(vqc):

	# initialize the iteration count for the circuit
	vqc.initialize_optimization_iterations()

	# translate bit strings to state vectors before processing
	print(f"\nTranslating fingerprints to quantum state vectors ..")
	vqc.SV = vqc.circuit.batch_state_prep(vqc.X)

	# TODO :: set the number of times that the optimization function stores
	# 		stats associated with the optimization function

	# optimize the weights associated with the circuit
	W_init = vqc.W
	print(f"Optimizing VQC circuit ..")
	opt = minimize (vqc.cost_function, vqc.W, method = 'Powell', bounds = vqc.B)

	# assign the optimal weights to the classification circuit
	vqc.W = opt.x
	print("\nFinal value of error function after optimization: {:0.3f}.".format(opt.fun))

	# TODO :: save weights after optimization has ended

	# TODO :: return the stats for the circuit optimization process
	return None

""" method used to define the error associated with a set of circuit weights,
	calculated by find the norm between a set of predict classifications and
	their real values."""
def error (predictions, classifications):
	# tolerance used to measure if a prediction is correct
	tol = 5e-2
	# initialize norm, accuracy, cross entropy measurements
	norm = 0.
	accuracy = 0
	ce = 0.
	# compare all labels and predictions
	for i in range(len(predictions)):

		# pull the prediction and classification values
		p = predictions[i]
		c = classifications[i]

		# calculate model accuracy
		if abs(c - np.sign(p)) < tol:
			accuracy = accuracy + 1

		# rescale classification and prediction to values between zero and one
		p = (p / 2) + 0.5
		c = (c / 2) + 0.5

		# calculate binary cross_entropy loss
		# coarse grain the prediction values to avoid log(0) calculations
		if p < tol:
			# if the prediction is within the tolerance of the value 0
			# replace the prediction with one within the range of the tolerance
			p = tol
		elif p > (1. - tol):
			# if the prediction is within the tolerance of the value 1.
			# replace the prediction with a value at the limit of the tolerance
			tol = 1. - tol

		ce += -(c * math.log(p) + (1 - c) * math.log(1 - p))

		# calculate euclidean norm loss
		norm = norm + (c - p) ** 2

	# normalize norm, accuracy, cross entropy by the data size
	norm = norm / len(predictions)
	accuracy = accuracy / len(predictions)
	ce = ce / len(predictions)
	return norm, accuracy, ce


## VQC CLASS METHODS ##
# variational quantum classifier 
class VQC:
	""" initialization routine for VQC object. """
	def __init__(self, qubits, state_prep = None):

		# initialize the number of qubits input 
		# into the classification circuit
		self.initialize_VQC_qubits(qubits)

		# initialize the number of classical bits, according to 
		# how the classical information are encoded in the qubits
		self.initialize_qubit_state_prep_method(state_prep_method = state_prep, \
			default = default_state_prep_method)

		# initialize the architecture type used for VQC
		self.circuit = None

		# initialize X and Y data sets as empty
		self.X = None
		self.Y = None
		self.SV = None
		# array that contains state vectors used to initialize 
		# quantum state during circuit calculations

		# array containing weights of unitary operations
		# and their upper and lower boundaries
		self.W = None
		self.B = None

	""" method used to wrap qubit initialization. """
	def initialize_VQC_qubits(self, qubits):

		# check that the number of qubits assigned to
		# the VQC is acceptable
		if (qubits <= max_qubits and qubits >= min_qubits):
			# assign the qubits
			self.qubits = qubits
		else:
			# the number is outside of the range
			# throw an error
			print(f" Error assigning qubits. Number passed to method outside of the allowable range.")
			print(f" Q ({qubits}) is outside of Q_MIN ({min_qubits}) and Q_MAX ({max_qubits}).")
			exit()

	""" method used to initialize method for qubit state preperation. 
		The state preperation method determines the number of classical
		bits that should be used to load the dataset. """
	def initialize_qubit_state_prep_method(self, state_prep_method, default):
		# if no state prep specification was provided by the user
		if state_prep_method == None:
			# assign the default
			state_prep_method = default

		# assign the state prep method
		if state_prep_method == "BasisEmbedding":
			self.state_prep = "BasisEmbedding"
			# for basis embeddeding, the number of classical bits is the same as
			# the number of qubits
			self.classical_bits = self.qubits
		elif state_prep_method == "AmplitudeEmbedding":
			self.state_prep = "AmplitudeEmbedding"
			# for amplitude embedding, the number of classical bits is 2 ^ N_qubits
			self.classical_bits = 2 ** self.qubits
		else:
			print(f"TODO :: Implement {state_prep_method} qubit state preperation method.")
			exit()

	""" method used to the set the number used to enumerate the number
		of times that a circuit has been optimized to zero """
	def initialize_optimization_iterations(self):
		# set the counter to zero
		self.n_it = 0

	""" method used to load smile strings and activity classifications
		from csv file. """
	def load_data (self, path, smile_col, class_col, fp_type = default_fp_type, fp_radius = default_fp_radius, BAE = False, verbose = False):

		# check that the path to the specified file exists
		if not os.path.exists(path):
			print(f" PATH ({path}) does not exist. Cannot load dataset.")
			exit()

		# inform user, loaded csv
		print(f"\nLoading SMILES from ({path}) ..")
		df = pd.read_csv(path)
		# check that headers are in the dataframe, load data
		if not smile_col in df.columns:
			print(f" SMILE COL ({smile_col}) not in FILE ({path}). Unable to load smile strings.")
			exit()
		elif not class_col in df.columns:
			print(f" CLASSIFICATION COL ({class_col}) not in FILE ({path}). Unable to load classification data.")
			exit()

		# load dataset, inform user
		smi = df[smile_col].tolist()
		print(f"Translating SMILES to {self.classical_bits}-bit vector with {fp_type} ..")

		# generate bit vector fingerprints
		if BAE == True:
			# use autoencoder to translate high-dimensional bit vector
			# to low dimensional bit vector
			print(f" TODO :: implement autoencoder.")
			exit()
		else:
			fp = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), \
				radius = fp_radius, nBits = self.classical_bits, useFeatures = True).ToBitString() for x in smi]

		# translate fingerprint vectors to binary strings
		self.X = np.zeros((len(fp), self.classical_bits), dtype=int)
		for i in range(len(fp)):
			# print(fp[i])
			for j in range(self.qubits):
				if fp[i][j] == "1":
					# replace 0 with 1
					self.X[i,j] = 1

		# load classification data
		self.Y = np.array(df[class_col])
		self.Y = self.Y * 2 - np.ones(len(self.Y)) # shift label form [0, 1] to [-1, 1]

		# if the user wants to write the data set to the
		if verbose:
			print(f"\nLoading DATA ({path}) .. \n")
			for i in range(len(self.Y)):
				print("X = {}, Y = {:.2f}".format(self.X[i], self.Y[i]))

	""" method used to initialize the VQC circuit architecture, unitary weights. """
	def initialize_circuit(self, circuit = None, bit_correlation = False):

		## TODO :: prevent this method for being called if there is no data associated with the object

		# initialize the circuit architecture
		# if none was specified, load the default ansatz
		if circuit is None:
			circuit = default_VQC_circuit

		# initialize the anstaz object 
		if circuit == 'VC':
			self.circuit = VariationalClassifier(self.qubits, self.state_prep)
		elif circuit == 'TTN':
			self.circuit = TreeTensorNetwork(self.qubits, self.state_prep)
		else:
			print(f"ERROR :: {circuit} circuit ansatz not implemented yet.")
			exit()

		# initialize unitary weights and their upper and lower bounds
		# according to the number of qubits and circuit architecture
		self.W, self.B = self.circuit.initial_weights()

	""" method used to calculate the error of a give set of unitary weights 
		in predicting the class of a given set of bit strings. """
	def cost_function(self, W):

		# make predictions for all X values
		Y_pred = []
		Y_class = []
		for i in range(len(self.SV)):

			# state vector
			sv = self.SV[i]

			# make a prediction with the weights passed to the function
			y = self.circuit.classify(W, state_vector = sv)

			# add the prediction and its known value to the list
			Y_pred.append(y)
			Y_class.append(self.Y[i])

		# calculate the cost and accuracy of the weights
		norm, acc, ce = error(Y_pred, Y_class)
		self.n_it += 1

		# report the status of the model predictions to the user
		print("Iteration: {:5d} | Cost: {:0.5f} | Cross-Entropy: {:0.5f} | Accuracy : {:0.5f}"\
			.format(self.n_it, norm, ce, acc))

		# return the value to the user
		return ce

	""" method used to write bit strings and classification to external file."""
	def write_data (self, path):
		pass


