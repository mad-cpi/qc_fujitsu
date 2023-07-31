import sys, os
import pandas as pd
import numpy as np
# rdkit libraries
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
# VQC circuit architectures
from fujitsu.VQC.circuit_architecture import VC

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

# default thresholding status
default_threshold_status = False

# defauly thresholding amount, if none is specified
# when thresholding is turned on
default_threshold = 0.02

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


## VQC CLASS AND METHODS ##
# variational quantum classifier 
class VQC:
	""" initialization routine for VQC object. """
	def __init__(self, qubits):

		# initialize the number of qubits input 
		# into the classification circuit
		self.initialize_VQC_qubits(qubits)

		# initialize the architecture type used for VQC
		self.circuit = None

		# initialize hyperparameters as off
		self.thresholding_status = default_threshold_status
		self.batch_status = default_batch_status

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

	""" method used to load smile strings and activity classifications
		from csv file. """
	def load_data (self, path, smile_col, class_col, fp_type = default_fp_type, fp_radius = default_fp_radius, BAE = False, verbose = False):

		# check that the path to the specified file exists
		if not os.path.exists(path):
			print(f" PATH ({path}) does not exist. Cannot load dataset.")
			exit()

		# loaded csv
		df = pd.read_csv(path)
		# check that headers are in the dataframe, load data
		if not smile_col in df.columns:
			print(f" SMILE COL ({smile_col}) not in FILE ({path}). Unable to load smile strings.")
			exit()
		elif not class_col in df.columns:
			print(f" CLASSIFICATION COL ({class_col}) not in FILE ({path}). Unable to load classification data.")
			exit()

		# load dataset
		smi = df[smile_col].tolist()

		# generate bit vector fingerprints
		if BAE == True:
			# use autoencoder to translate high-dimensional bit vector
			# to low dimensional bit vector
			print(f" TODO :: implement autoencoder.")
			exit()
		else:
			fp = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), \
				radius = fp_radius, nBits = self.qubits, useFeatures = True).ToBitString() for x in smi]

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

		# if the user wants to write the data set to the
		if verbose:
			print(f"\nLoading DATA ({path}) .. \n")
			for i in range(len(self.Y)):
				print("X = {}, Y = {:.2f}".format(self.X[i], self.Y[i]))


	""" method used to initialize the VQC circuit architecture, unitary weights. """
	def initialize_circuit(self, circuit = None, QFT = False, bit_correlation = False):

		## TODO :: prevent this method for being called if there is no data associated with the object

		# initialize the circuit architecture
		if circuit == None:
			self.circuit = VC(self.qubits)
		else:
			# TODO check that architecture is in 
			# list of acceptable archite
			self.circuit = VC(self.qubits)

		# initialize unitary weights and their upper and lower bounds
		# according to the number of qubits and circuit architecture
		self.W, self.B = self.circuit.initial_weights()

	""" method used to calculate the error of a give set of unitary weights 
		in predicting the class of a given set of bit strings. """
	def cost_function(W):

		# if thresholding is turned on, establish the threshold
		if self.thresholding_status:
			print ("TODO :: implement thresholding.")
		else:
			t = 0.99

		# if batching is turned on
		if batch_status:
			# generate a random list of X and Y
			index = np.random.randint(0, high = len(X), size = n_batch)
		else:
			index = [x for x in range(len(self.X))]

		# make predictions for all X values
		# if values are above the treshhold
		# assume the predctions are correct
		Y_pred = []
		for i in index:
			# get the smile strings
			x = self.X[i]

			# make a prediction
			y = self.circuit.classify(W, x)

			if self.thresholding_status:
				print ("TODO :: implement thresholding.")

			# add the prediction and its known value to the list
			Y_pred.append([y, Y[i]])


		# calculate the cost and accuracy of the weights
		cost, acc = error(Y_pred)

		# return the value to the user
		return cost

	""" method used to optimize the current weights of a variational circuit according
		to the dataset sotored within the variational classification circuit
		object. """
	def optimize(self):

		opt = minimize (cost_function, self.W, method = 'Powell', bounds = self.B)

		# assign the optimal weights to the classification circuit
		self.W = opt.x
		print("\nFinal value of error function after optimization: {:0.3f}.".format(opt.fun))

		# TODO :: save weights!!

	""" initialize batching protcol for optimization """
	def set_batching(self, status, batch_size = None):

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

	""" initialize tresholding protocol for optimization routine """
	def set_threshold(self, status, threshold = None):

		if status == True:
			# turn on the tresholding routine
			self.thresholding_status = True

			# if the threshold was no passed to the method
			if threshold == None:
				# assign the default value
				self.threshold = default_threshold
			else:
				# check that the value passed to the method is correct
				if threshold > 0. and threshold < 1.:
					# if the value is between one and zero
					self.threshold = threshold
				else:
					# assign the default
					threshold = default_threshold

	""" method used to write bit strings and classification to external file."""
	def write_data (self, path):
		pass


