# Program to run the QFT Classifier and evaluate the results 
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
import os
from rdkit import Chem
from rdkit.Chem import AllChem, rdFingerprintGenerator
import textwrap
from QFT_classifier import QFT_classifier
import random

# parameters to change 

#  training data points usually starts from the 1st point on the dataset
training_data = 450
# Add data points correponding to the position in dataset to check validity of the model

check_start = 1020
check_stop = 2822

#######################
## Reading file and converting to desired fingerprint 
###
path = "malaria_IC50_curated.csv"
smile_col = 'SMILES'
class_col = 'single-class-label'
# default fingerprint type used for encoding 
# smile strings to bit vectors
fp_type = "rdkfp"

# default connectivity radius used for encoding 
# smile strings to bit vectors
fp_radius = 3

qubits = 2047

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
fp = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(x), radius = fp_radius, nBits = qubits, useFeatures = True).ToBitString() for x in smi]	

Y = np.array(df[class_col])
X = np.zeros((len(fp), qubits), dtype=int)

# translate fingerprint vectors to binary strings
for i in range(len(fp)):
	# print(fp[i])
	for j in range(qubits):
		if fp[i][j] == "1":
			# replace 0 with 1
		    X[i,j] = 1

######################################

# converting to position values 
def pos_conv(X):
	count = 0
	X_pos = []
	for i in X:
		count +=1
		if i == 1:
			X_pos.append(count)
	return X_pos

# Converting bits to positional vectors

Qubits = 11

qft_classifier = QFT_classifier(Qubits)

# Debug file for evaluating the training and theta
traindata_file = open('trainingdata.txt','w')
expect_file = open('Optimized_Expectation.txt', 'w')
opt_theta = open('Optimized_theta.txt', 'w')

# Training data set
X_1 = X[0:training_data]
Y_1 = Y[0:training_data]
len_dataset = training_data

print("Length of training data set = ", len_dataset)

#initialize arrays 
X_train = []
Y_train = []
expectation = []

class_list = [0,300,600,900,1200,1500,1800,2047]
X_split = [[],[],[],[],[],[],[]]
Y_split = [[],[],[],[],[],[],[]]

for j in range(len_dataset):
	SX = pos_conv(X[j])
	for kk in range(len(SX)):
		for mm in range(len(class_list)-1):
			if SX[kk] > class_list[mm] and SX[kk] < class_list[mm+1]:
				X_split[mm].append(SX[kk])
				Y_split[mm].append(Y[j])
				


print(X_split[0])

for mn in range(len(class_list)-1):
	traindata_file.write(str(X_split[mn]))  
	traindata_file.write('\n')
	traindata_file.write(str(Y_split[mn]))
	traindata_file.write('\n')	

print("Training model running ...")
# Running the training model
theta_opt =[]
expect_opt = []
for jj in range(len(class_list)-1):
	theta, expect = qft_classifier.classifier_train(X_split[jj],Y_split[jj])
	theta_opt.append(theta)
	expect_opt.append(expect)

print("Training done")

# Printing optimized theta values	
print("Optimized Theta =",theta)
print("Average Expectation =",expect)
opt_theta.write(str(len_dataset))
opt_theta.write('\n')
opt_theta.write(str(theta_opt))
expect_file.write(str(expect_opt))
expect_file.write('\n')


#######################
# Checking trained model data with different data points 
#######################

print("Running Predictor....")

# Debug file 

file_result = open('output.txt','w')
file_classifier = open('classified.txt','w')
Result_data = open('result.csv','w')

X_pred = X[check_start:check_stop]
Y_act = Y[check_start:check_stop]

print("From values",check_start, "to values", check_stop)

Predicted_value = np.zeros(len(X_pred))
Predicted_value_tmp = np.zeros(len(class_list))
Expectation_arr = [[],[],[],[],[],[],[]]

Result_data.write("Data_number,")
Result_data.write("Count matched,")
Result_data.write("Predicted Value,")
Result_data.write("Actual Value")

for i in range(len(X_pred)):
	SX = pos_conv(X_pred[i])
	expectation_mean = []
	
	file_classifier.write('\n')
	file_result.write(str(i))
	file_result.write('\n')

	Result_data.write(str(check_start+i))
	Result_data.write(',')

	for mk in SX:
		for mm in range(len(class_list)-1):
			if mk > class_list[mm] and mk < class_list[mm+1]:
				expectation = qft_classifier.QFT_predictor(theta_opt[mm], mk)
				Expectation_arr[mm].append(expectation)
				file_result.write(str(expectation))
				expectation_mean.append(expectation)
				file_result.write('\n') 
					
	file_classifier.write(str(np.mean(expectation_mean)))
	cnt =0	
	positive = 0
	for kl in range(len(class_list)-1):
		try:
			mean_exp = np.mean(Expectation_arr[kl])
			if mean_exp > expect_opt[kl]:
				positive +=1
			cnt +=1
		except:
			print("missing data")
			
        
		#print("mean",mean_exp)
		#print("expect",expect_opt[kl])
		#print(cnt-positive)
	if positive >= 3:
			Predicted_value[i] =1

	Result_data.write(str(positive))
	Result_data.write(',')
	Result_data.write(str(Predicted_value[i]))
	Result_data.write(',')
	Result_data.write(str(Y[i]))
	Result_data.write('\n')

# Checking the prediction accuracy of the model
count =0
print(Predicted_value)
print(Y_act)
for i in range(len(Y_act)):
	if Predicted_value[i] == Y[i]:
		count +=1
print("Model accuracy for predicting output is")
print((len(X_pred)-count)/len(X_pred)*100)	



	



