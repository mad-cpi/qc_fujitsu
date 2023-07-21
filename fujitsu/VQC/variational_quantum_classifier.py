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
			self.circuit = default_VQC_circuit
		else:
			# TODO check that architecture is in 
			# list of acceptable archite
			self.circuit = circuit

		# initialize unitary weights and their upper and lower bounds
		# according to the number of qubits and circuit architecture
		self.W, self.B = self.circuit.initial_weights(self.qubits)


	""" method used to write bit strings and classification to external file."""
	def write_data (self, path):
		pass


