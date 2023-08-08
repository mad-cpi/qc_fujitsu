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

# default thresholding status
default_threshold_status = False

# default thresholding amount, if none is specified
# when thresholding is turned on
default_threshold_increase_size = 0.02

# default initial threshold, uased when threshold is off or
# has just been turned on
default_initial_threshold = 0.02

# default number of iterations that the threshold is increase
default_threshold_increase_step = 50

# maximum threshold, after which the threshold will not increase any more
# if thresholding is off, this is the constant value used for thresholding
# calssifications from the quantum circuit
default_threshold_max = 0.75

# default batching status
default_batch_status = False

# default batch size, as a precentage, when batching
# is turned on, if none is specified
default_batch_size = 0.8


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

	print(f"\nOptimizing VQC circuit ..")

	# initialize the iteration count for the circuit
	vqc.initialize_optimization_iterations()

	# optimize the weights associated with the circuit
	W_init = vqc.W
	opt = minimize (vqc.cost_function, vqc.W, method = 'Powell', bounds = vqc.B)

	# assign the optimal weights to the classification circuit
	vqc.W = opt.x
	print("\nFinal value of error function after optimization: {:0.3f}.".format(opt.fun))

	# TODO :: save weights!!

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

		# initialize hyperparameters as off
		self.set_threshold(status = default_threshold_status, verbose = False)
		self.set_batching(status = default_batch_status, verbose = False)
		self.n_it = 0

		# initialize X and Y data sets as empty
		self.X = None
		self.Y = None

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
	def initialize_qubit_state_prep_method(self, state_prep, default):
		# if no state prep specification was provided by the user
		if state_prep == None:
			# assign the default
			state_prep = default

		# assign the state prep method
		if state_prep == "BasisEmbedding":
			self.state_prep = "BasisEmbedding"
			# for basis embeddeding, the number of classical bits is the same as
			# the number of qubits
			self.classical_bits = self.qubits
		elif state_prep == "AmplitudeEmedding":
			self.state_prep = "AmplitudeEmedding"
			# for amplitude embedding, the number of classical bits is 2 ^ N_qubits
			self.classical_bits = 2 ** self.qubits
		else:
			print(f"TODO :: Implement {state_prep} qubit state preperation method.")
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
		print(f"\nTranslating SMILES to {self.classical_bits}-bit vector with {fp_type} ..")

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
		self.X = np.zeros((len(fp), self.qubits), dtype=int)
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
	def initialize_circuit(self, circuit = None, QFT = False, bit_correlation = False):

		## TODO :: prevent this method for being called if there is no data associated with the object

		# initialize the circuit architecture
		# if none was specified, load the default ansatz
		if circuit == None:
			circuit = default_VQC_circuit

		# initialize the anstaz object 
		if circuit == 'VC':
			self.circuit = VariationalClassifier(self.qubits)
		elif circuit == 'TTN':
			self.circuit = TreeTensorNetwork(self.qubits)
		else:
			print(f"ERROR :: {circuit} circuit ansatz not implemented yet.")
			exit()

		if QFT == True:
			# if the QFT status is true
			# turn QFT embedding for the circuit on
			self.circuit.set_QFT_status(status = QFT)

		# initialize unitary weights and their upper and lower bounds
		# according to the number of qubits and circuit architecture
		self.W, self.B = self.circuit.initial_weights()

	""" method used to calculate the error of a give set of unitary weights 
		in predicting the class of a given set of bit strings. """
	def cost_function(self, W):

		# if thresholding is turned on, establish the threshold
		if self.thresholding_status:
			if (self.n_it % self.threshold_increase_freq == 0 ) and (self.n_it != 0):
				print(f"Threshold increased from {self.threshold} to {(self.threshold + self.threshold_increase_size)}.")
				self.threshold += self.threshold_increase_size

			if self.threshold > self.threshold_max:
				self.threshold = self.threshold_max
				print(f"Threshold set to maximum ({self.threshold})")
		else:
			self.threshold = default_threshold_max

		# if batching is turned on
		# adjust for class imbalance
		if self.batch_status:
			# generate a random list of X and Y
			# TODO :: double check that this works
			n_batch = math.floor(self.batch_size * len(self.X))
			index = np.random.randint(0, high = len(self.X), size = n_batch)
		else:
			# generate training set that is event in active and inactive samples
			index = [x for x in range(len(self.X))]

		# make predictions for all X values
		# if values are above the treshhold
		# assume the predctions are correct
		Y_pred = []
		Y_class = []
		for i in index:
			# get the fingerprint
			x = self.X[i]

			# make a prediction with the weights passed to the function
			y = self.circuit.classify(W, x)
			# if abs(y) > self.threshold:
			# 	y = np.sign(y)

			# add the prediction and its known value to the list
			Y_pred.append(y)
			Y_class.append(self.Y[i])

		# check that the length of the predictions array is the same length
		# as the batch array
		# print (Y_pred)
		if (len(Y_pred) != len(index)):
			exit()
		# calculate the cost and accuracy of the weights
		norm, acc, ce = error(Y_pred, Y_class)
		self.n_it += 1
		# report the status of the model predictions to the user
		print("Iteration: {:5d} | Cost: {:0.5f} | Cross-Entropy: {:0.5f} | Accuracy : {:0.5f}"\
			.format(self.n_it, norm, ce, acc))

		# return the value to the user
		return ce

	""" initialize batching protcol for optimization """
	def set_batching(self, status, batch_size = None, verbose = True):

		if status == True:
			# turn on the batching routine
			self.batch_status = True

			# if the batch size was not passed to the method
			if batch_size == None:
				# assign the default batch size
				self.batch_size = default_batch_size
			else:
				# check that the batch size value passed to the method is correct
				if batch_size <= 1. and batch_size > 0.:
					# if the value is between one and zero
					self.batch_size = batch_size
				else:
					# assign the default value
					self.batch_size = default_batch_size

			if verbose:
				print(f"\nVQC optimization batching was turned on.")
				print("Batch size :: {:4.2f} precent of training set".format(self.batch_size))
		else:
			self.batch_status = False

	""" initialize tresholding protocol for optimization routine """
	def set_threshold(self, status, threshold_initial = None, threshold_increase_size = None, threshold_increase_freq = None, threshold_max = None, verbose = True):

		if status == True:
			# turn on the tresholding routine
			self.thresholding_status = True

			# assign the thresholding parameters
			# if the threshold was no passed to the method
			if threshold_initial == None:
				# assign the default value
				self.threshold = default_initial_threshold
			else:
				# check that the value passed to the method is correct
				if threshold_initial > 0. and threshold_initial < 1.:
					# if the value is between one and zero
					self.threshold = threshold_initial
				else:
					# assign the default
					self.threshold = default_initial_threshold

			# assign the thresholding increase size
			if threshold_increase_size == None:
				# if no value was passed to the method
				# assign the default
				self.threshold_increase_size = default_threshold_increase_size
			else:
				# if a value for the threshold increase size was passed to the method
				# check that the value is okay
				if threshold_increase_size > 0. and threshold_increase_size < 0.1:
					# assign the value
					self.threshold_increase_size = threshold_increase_size
				else:
					# if the value is outside of the allowable range
					# assign the default
					self.threshold_increase_size = default_threshold_increase_size

			# assign the thresholding increase frequency
			if threshold_increase_freq == None:
				# if no value was passed to the method, assign the default
				self.threshold_increase_freq = default_threshold_increase_step
			else:
				# if a value was passed to the method, check that the value meets the constrains
				if threshold_increase_freq > 0:
					# assign the value
					self.threshold_increase_freq = threshold_increase_freq
				else:
					# if the value did not meet the constrains, assign the default
					self.threshold_increase_freq = default_threshold_increase_step

			# assign the maximum threshold for the method
			if threshold_max == None:
				# if a value has not been passed to the method,
				# assign the default
				self.threshold_max = default_threshold_max
			else:
				# if a value has been passed to the method,
				# check that is meets the contrains
				if threshold_max > 0. and threshold_max < 1.0:
					# the value meets the contrains, 
					# assign the max threshold value
					self.threshold_max = threshold_max 
				else:
					# the value does not meet the criteria
					# assign the default
					self.threshold_max = default_threshold_max

			# report to user
			if verbose:
				print(f"\nVQC optimization thresholding turned on.")
				print("Initial Threshold :: {:5.3f}".format(self.threshold))
				print("Threshold Increase Frequency :: {:03d} iterations".format(self.threshold_increase_freq))
				print("Threshold Increase Size :: {:5.3f}".format(self.threshold_increase_size))
				print("Maximum Threshold :: {:5.3f}".format(self.threshold_max))

		else:
			self.thresholding_status = False 

			if verbose:
				print(f"\nVQC optimization thresholding turned off.")
				print("Threshold will remain constant at {:5.3}".format(default_threshold_max))

	""" method used to write bit strings and classification to external file."""
	def write_data (self, path):
		pass


