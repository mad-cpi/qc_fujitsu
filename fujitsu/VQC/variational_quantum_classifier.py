import sys, os
import pandas as pd
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
# used for learning
from scipy.optimize import minimize
import warnings
import time
# used for model saving and loading
import yaml
# rdkit libraries
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
# used to calculate model stats
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef, \
	precision_score, recall_score, roc_curve, auc 
# VQC circuit architectures
from VQC.circuit_architecture import VariationalClassifier, TreeTensorNetwork

## PARAMETERS ## 

## VQC class default parameters used for initialization ##
# TODO :: add list of acceptable circuit architectures

# max number of qubits that can be assigned
# to the classification circuit (for safety,
# on any desktop this number shouldnt be greater than 20)
# max_qubits = 26

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


class TookTooLong(Warning):
    pass

## VQC OPTIMIZATION METHODS ##

""" method used to optimize the current weights of a variational circuit according
	to the dataset sotored within the variational classification circuit
	object. """
def optimize(vqc, save_dir = None, title = None, max_opt_steps = None, max_opt_hours = None, tol = None, minimize_method = 'Powell'):

	# check that data has been loaded into the circiut
	if vqc.X is None or vqc.Y is None:
		print(f"Error :: Must load data into circuit before optimization.")
		exit()

	# initialize the iteration count for the circuit
	opts_dict = {}
	vqc.initialize_circuit_optimization(steps = max_opt_steps, hours = max_opt_hours)
	if vqc.max_opt_steps is not None:
		if minimize_method == 'TNC':
			opts_dict['maxfun'] = vqc.max_opt_steps
		elif minimize_method == 'Powell':
			opts_dict['maxfev'] = vqc.max_opt_steps
		else:
			opts_dict['maxiter'] = vqc.max_opt_steps

	if tol is not None:
		if tol > 0.:
			opts_dict['ftol'] = tol

	# translate bit strings to state vectors before processing
	print(f"\nTranslating fingerprints to quantum state vectors ..")
	# vqc.SV = vqc.circuit.batch_state_prep(vqc.X)

	# optimize the weights associated with the circuit
	W_init = vqc.W
	print(f"Optimizing VQC circuit {opts_dict}..")
	opt = minimize (vqc.cost_function, x0 = vqc.W, method = minimize_method, bounds = vqc.B, options = opts_dict, callback = vqc.minimizer_callback)

	# assign the optimal weights to the classification circuit
	vqc.W = opt.x
	print("\nFinal value of error function after optimization: {:0.3f}.".format(opt.fun))

	# if the user has provided a save path, save the stats to a file
	if save_dir is not None:

		# check that the path exists
		if not os.path.exists(save_dir):
			os.mkdir(save_dir)

		# create dataframe
		df = pd.DataFrame(data = {'cost': vqc.opt_cost, 'cross-entropy': vqc.opt_ce, 'accuracy': vqc.opt_acc})

		# save file
		if title is None:
			title = 'VQC'
		save_file = save_dir + title + '_optimiazation.csv'
		df.to_csv(save_file, index = False)
	return None

""" method used to define the error associated with a set of circuit weights,
	calculated by find the norm between a set of predict classifications and
	their real values. """
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
			p = 1. - tol

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
	def __init__(self, qubits, state_prep = None, fp_radius = None, fp_type = None):

		# initialize the number of qubits input 
		# into the classification circuit
		self.set_VQC_qubits(qubits)

		# initialize the stacking state of the function
		self.set_function		

		# initialize the number of classical bits, according to 
		# how the classical information are encoded in the qubits
		self.set_qubit_state_prep_method(state_prep_method = state_prep, \
			default = default_state_prep_method)

		# assign fingerprinting technique
		self.set_fingerprint(fp_type = fp_type, fp_radius = fp_radius)

		# initialize the architecture type used for VQC
		self.circuit = None

		# initialize X and Y data sets as empty
		self.X = None # bit strings / vectors
		self.Y = None # classification of X, as one of two catagories
		self.P = None # predictions make by circuit that use X to determine Y
		self.SV = None
		# array that contains state vectors used to initialize  
		# quantum state during circuit calculations

		# array containing weights of unitary operations
		# and their upper and lower boundaries
		self.W = None
		self.B = None

	""" method used to wrap qubit initialization. """
	def set_VQC_qubits(self, qubits):

		# check that the number of qubits assigned to
		# the VQC is acceptable
		if (qubits >= min_qubits):
			# assign the qubits
			self.qubits = qubits
		else:
			# the number is outside of the range
			# throw an error
			print(f" Error assigning qubits. Number passed to method outside of the allowable range.")
			print(f" Q ({qubits}) is outside of Q_MIN ({min_qubits}).")
			exit()

	""" method used to initialize method for qubit state preperation. 
		The state preperation method determines the number of classical
		bits that should be used to load the dataset. """
	def set_qubit_state_prep_method(self, state_prep_method, default):
		# if no state prep specification was provided by the user
		if state_prep_method is None:
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

	""" method for assigning the techniques that are used to fingerprint
		SMILE strings. """
	def set_fingerprint(self, fp_type = None, fp_radius = None):

		# assign the fingerprinting method
		if fp_type is None:
			fp_type = default_fp_type

		if fp_type == 'rdkfp':
			self.fp_type = 'rdkfp'
		else:
			print(f"TODO :: Implement {fp_type} fingerprinting technique. Assigning default ({default_fp_type}).")
			self.fp_type = default_fp_type

		# assign the fingerprinting radius
		if fp_radius is None or fp_radius < 3:
			fp_radius = default_fp_radius

		self.fp_radius = fp_radius

	""" method used to the set the number used to enumerate the number
		of times that a circuit has been optimized to zero """
	def initialize_circuit_optimization(self, steps = None, hours = None):
		# set the counter to zero
		self.n_it = 0
		# set the elapsed time to the clock time before the minimizer starts optimization process
		self.opt_start_time = time.time()
		# assign the maximum number of steps that the optimizer can for the call back function
		if steps is not None:
			if steps > 0:
				self.max_opt_steps = steps
		else:
			self.max_opt_steps = None
		# assign the maximum length of time the optimizer can run
		if hours is not None:
			if hours > 0.:
				self.max_opt_hours = hours
		else:
			self.max_opt_hours = None
		# initialize the arrays that contain the optimization stats
		self.opt_cost = [] # cumulation of cost scores during model optimization
		self.opt_ce = [] # cumulation of cross entropy scores during optimization
		self.opt_acc = [] # cumulation of accuracy scores during optimization

	""" method used to load smile strings and activity classifications
		from csv file. """
	def load_data (self, load_path, smile_col, class_col, BAE = False, verbose = False):

		# check that the path to the specified file exists
		if not os.path.exists(load_path):
			print(f" PATH ({load_path}) does not exist. Cannot load dataset.")
			exit()

		# inform user, loaded csv
		print(f"\nLoading SMILES from ({load_path}) ..")
		df = pd.read_csv(load_path)
		# check that headers are in the dataframe, load data
		if not smile_col in df.columns:
			print(f" SMILE COL ({smile_col}) not in FILE ({load_path}). Unable to load smile strings.")
			exit()
		elif not class_col in df.columns:
			print(f" CLASSIFICATION COL ({class_col}) not in FILE ({load_path}). Unable to load classification data.")
			exit()

		# load dataset, inform user
		smi = df[smile_col].tolist()
		print(f"Translating SMILES to {self.classical_bits}-bit vector with {self.fp_type} ..")

		# generate bit vector fingerprints
		if BAE == True:
			# use autoencoder to translate high-dimensional bit vector
			# to low dimensional bit vector
			print(f" TODO :: implement autoencoder.")
			exit()
		elif self.fp_type == 'rdkfp':
			# use the rdkfp
			fp = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), \
				radius = self.fp_radius, nBits = self.classical_bits, useFeatures = True).ToBitString() for x in smi]
		else:
			print(f"TODO :: Implement {self.fp_type} fingerprinting technique.")
			exit()

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
			print(f"\nLoading DATA ({load_path}) .. \n")
			for i in range(len(self.Y)):
				print("X = {}, Y = {:.2f}".format(self.X[i], self.Y[i]))

	""" method used to initialize the VQC circuit architecture, unitary weights. """
	def initialize_circuit(self, circuit = None, bit_correlation = False, stack = False):

		## TODO :: prevent this method for being called if there is no data associated with the object

		# initialize the circuit architecture
		# if none was specified, load the default ansatz
		if circuit is None:
			circuit = default_VQC_circuit

		# initialize the anstaz object 
		if circuit == 'VC':
			self.circuit = VariationalClassifier(self.qubits, self.state_prep, stack)
		elif circuit == 'TTN':
			self.circuit = TreeTensorNetwork(self.qubits, self.state_prep, stack)
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
		# Y_pred = self.circuit.classifications(W, state_vectors = self.SV)
		# Y_class = self.Y

		Y_pred = []
		Y_class = []
		# for i in range(len(self.SV)):
		for i in range(len(self.X)):

			# state vector
			# sv = self.SV[i]
			x = self.X[i]

			# make a prediction with the weights passed to the function
			# y = self.circuit.classify(W, state_vector = sv)
			y = self.circuit.classify(W, bit_string = x)

			# add the prediction and its known value to the list
			Y_pred.append(y)
			Y_class.append(self.Y[i])

		# calculate the cost and accuracy of the weights
		norm, acc, ce = error(Y_pred, Y_class)
		self.n_it += 1 # increment iteration
		self.opt_cost.append(norm)
		self.opt_ce.append(ce)
		self.opt_acc.append(acc)

		# report the status of the model predictions to the user
		print("Iteration: {:5d} | Cost: {:0.5f} | Cross-Entropy: {:0.5f} | Accuracy: {:0.5f}"\
			.format(self.n_it, norm, ce, acc))

		# return the value to the user
		return ce

	""" method used to stop the minimizer after the maximum number of iterations
		has been reached. """
	def minimizer_callback (self, xk):

		# get the elapsed time since 
		opt_time = time.time() - self.opt_start_time
		print(opt_time)

		# if too much elapsed time has passed
		if opt_time > self.max_opt_hours:
			warnings.warn("Terminating optimization: time limit reached",
                          TookTooLong)
			# inform the user and stop optimization
			print(f"Optimization time ({opt_time}) has surpassed threshold ({self.max_opt_hours})")
			return True
		elif self.n_it > self.max_opt_steps:
			# if the optimizer has surpased the maximum number of iterations
			# inform the user and stop optimization
			warnings.warn("Terminating optimization: optimization steps reached",
                          TookTooLong)
			print(f"Optimization steps ({self.n_it}) has surpased the maximum allowable amount ({self.max_opt_steps})")
			return True
		else:
			print("Elapsed minimization time {:.3f}".format(opt_time))

	""" method that makes predictions for the data set that is loaded into 
		the variational qunatum circuit. returns array with with "probability-like"
		predictions for each compounds, scaled from 0 - 1 """
	def predict(self):

		# inform the user
		print(f"\nMaking predictions for the data stored within the circuit ..")

		# initialize the predictions, which are stored within the objecty
		self.P = []

		for i in range(len(self.X)):

			 # bit vector
			 x = self.X[i]

			 # make a prediction with the weights that are stored within the circuit
			 y = self.circuit.classify(self.W, bit_string = x)

			 # add the prediction to the list
			 self.P.append(y)

		# return the scaled predictions to the user
		return (self.P + np.ones(len(self.P))) / 2.

	""" use the predictions made by the circuit to generate statistics about the
		performance of the circuit as a model.

		NOTE: circuit does not check it the values that the circuit made predictions
		for were utilized by the model for training. """
	def get_stats (self, save_dir = None, title = None):

		# check that the circuit already has made predictions for a set
		if self.P is None:
			# circuit must already have made predictions
			print(f"Error :: circuit has not made any predictions.")
			exit()

		# if a path was specified by the user,
		# check that it exists
		if save_dir is not None:
			# check that the path exists
			if not os.path.exists(save_dir):
				# if it does not, make the path
				os.mkdir(save_dir)

		# generate file names
		if title is not None:
			title = 'VQC'
		file_stats = title + '_stats.csv'
		file_pred = title + '_true_pred.csv'
		file_roc = title + '_roc.csv'

		# calculate class label and predictions according to values stored with circuit
		Y_prob = [((p + 1.) / 2.) for p in self.P]
		Y_pred = [1 if p >= 0. else 0 for p in self.P]
		Y_true = [1 if c >= 0. else 0 for c in self.Y]
		df_pred = pd.DataFrame(data = {'y_true': Y_true, 'y_pred': Y_pred, 'y_prob': Y_prob})
		if save_dir is not None:
			df_pred.to_csv(save_dir + file_pred, index = False)

		# get stats
		stat_dict = {}
		stat_dict['acc'] = accuracy_score(Y_true, Y_pred)
		stat_dict['f1s'] = f1_score(Y_true, Y_pred)
		stat_dict['cks'] = cohen_kappa_score(Y_true, Y_pred)
		stat_dict['mcc'] = matthews_corrcoef(Y_true, Y_pred)
		stat_dict['pre'] = precision_score(Y_true, Y_pred)
		stat_dict['rec'] = recall_score(Y_true, Y_pred) 

		try:
			fpr, tpr, __ = roc_curve(Y_true, Y_prob)
			roc_auc = auc(fpr, tpr)
			stat_dict['roc_auc'] = roc_auc
			df_roc = pd.DataFrame(data = {'fpr': fpr, 'tpr': tpr})
			if save_dir is not None:
				df_roc.to_csv(save_dir + file_roc, index = False)
		except ValueError as ex:
			log.error(ex)

		if save_dir is not None:
			df_stat = pd.DataFrame(stat_dict, index = [0])
			df_stat.to_csv(save_dir + file_stats, index = False)

		return stat_dict

	""" method that stores the current state of the circuit as a dictionary. """
	def gen_dict(self):

		# initialize empty dictionary
		vqc_dict = {}

		# add circuit parameters to dictionary
		vqc_dict['qubits'] = self.qubits
		vqc_dict['qubit_state_prep'] = self.state_prep
		if type(self.circuit) is VariationalClassifier:
			vqc_dict['ansatz'] = 'VC'
		elif type(self.circuit) is TreeTensorNetwork:
			vqc_dict['ansatz'] = 'TTN'
		else:
			vqc_dict['ansatz'] = 'NA'
		vqc_dict['fp_type'] = self.fp_type
		vqc_dict['fp_radius'] = self.fp_radius
		vqc_dict['n_weights'] = len(self.W)
		vqc_dict['n_bounds'] = len(self.B)

		# save individual weights
		for i in range(len(self.W)):
			vqc_dict['weights_{:03d}'.format(i)] = str(self.W[i])

		# save upper and lower boundaries for each weight
		for i in range(len(self.B)):
			vqc_dict['bounds_{:03d}_0'.format(i)] = str(self.B[i][0])
			vqc_dict['bounds_{:03d}_1'.format(i)] = str(self.B[i][1])


		# return the dictionary to the user
		return vqc_dict

	""" method that uses a dictionary to assign the vqc state according to the
		data stored in the dictionary. """
	def load_dict(self, vqc_dict):

		# assign values to circuit according to data stored in dictionary
		self.sqt_VQC_qubits(vqc_dict['qubits'])
		self.set_qubit_state_prep_method(state_prep_method = vqc_dict['qubit_state_prep'], \
			default = default_state_prep_method)
		if vqc_dict['ansatz'] == 'VC':
			self.circuit = VariationalClassifier(self.qubits, self.state_prep)
		elif vqc_dict['ansatz'] == 'TTN':
			self.circuit = TreeTensorNetwork(self.qubits, self.state_prep)
		else:
			print(f"Unable to load circuit architecture {vqc_dict['ansatz']}")
		self.set_fingerprint(fp_type = vqc_dict['fp_type'], fp_radius = vqc_dict['fp_radius'])

		# create an empty array that is the weights length specified in the dictionary
		self.W = np.empty((vqc_dict['n_weights']))
		for i in range(len(self.W)):
			self.W[i] = float(vqc_dict['weights_{:03d}'.format(i)])

		# create and empty array this is the boundaries length specified in the dictionary
		self.B = [[[], []] for i in range(vqc_dict['n_bounds'])]
		for i in range(len(self.B)):
			self.B[i][0] = float(vqc_dict['bounds_{:03d}_0'.format(i)])
			self.B[i][1] = float(vqc_dict['bounds_{:03d}_1'.format(i)])

	""" method used to save the state of a circuit in a way that the exact same
		circuit can be reloaded. This includes the circuits weights (esp. after
		optimization), state prep, etc. """
	def save_circuit(self, save_to = None, save_as = None):

		if save_to is None:
			# must pass path to method
			print(f"ERROR :: must specify save directory as save_to.")

		# check that the path exists
		# if it does not, make the path
		if not os.path.exists(save_to):
			os.mkdir(save_to)

		vqc_dict = self.gen_dict()
		save_file = save_to + save_as + '.yaml'
		f = open(save_file, 'w')
		yaml.dump(vqc_dict, f)
		f.close()

	""" method that loads yaml file, which specifies circuit state. """
	def load_circuit (self, load_path):

		# check that the path to the yaml file exists
		if not os.path.exists(load_path):
			print(f"ERROR :: Unable to load circuit. Path ({load_path}) does not exist.")

		with open(load_path, 'r') as file:
			vqc_dict = yaml.safe_load(file)
		self.load_dict(vqc_dict)

	""" method used to write bit strings and classification to external file."""
	def write_data (self, path):
		pass


