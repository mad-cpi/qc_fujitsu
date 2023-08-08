
## Matthew Dorsey 
## 2023.08.07
## Program for splitting training datasets into training and validation sets

import sys, os, math
import pandas as pd
import numpy as np
import random
from sklearn.model_selection import ShuffleSplit

## PARAMETERS
# fraction of dataset that should be assigned the testing catagory
test_set_size = 0.8
# number of times to randomly split testing and training sets
num_set_split = 5
# path to save testing and training datasets to
save_dir = "./fujitsu/data/"


## ARGUMENTS
# path to dataset that should be split
path = sys.argv[1]
# title of SMILE col
smile_col = sys.argv[2]
# title of class col
class_col = sys.argv[3]
# save name after loading and splitting datasets
save_title = sys.argv[4]


## SCRIPT 
# generate the save directory
save_path = save_dir + f"{save_title}/"
if not os.path.exists(save_path):
	os.mkdir(save_path)

# load the data
df = pd.read_csv(path)
smiles = df[smile_col].tolist()
classifications = df[class_col].tolist()

# load data into np array
X = np.array([])
Y = np.array([])
for i in range(len(smiles)):
	X = np.append(X, np.array(smiles[i]))
	Y = np.append(Y, np.array(classifications[i]))

# split data set into 5 random testing and training sets
rs = ShuffleSplit(n_splits = num_set_split, test_size = test_set_size, random_state = 42)
# loop through split sets
for i, (train_index, test_index) in enumerate(rs.split(X)):
	print(f"\nFold {i}: ")
	# print(f"Train ({len(train_index)}): index={train_index}")
	# print(f"Test {len(test_index)}: index={test_index}")

	# determine the number of active and inactives in the training set
	actives = 0
	inactives = 0
	for j in range(len(train_index)):
		if Y[train_index[j]] == 1:
			actives += 1
		else:
			inactives += 1

	print(f"Train set contains {len(train_index)} items ({actives} actives, {inactives} inactives)")
	print(f"Test set contains {len(test_index)} items")

	# randomly sample testing set, until the number of inactives
	# and actives in the training set are equal
	count = 0
	while actives != inactives:
		# while the classes are imbalanced
		# select a random index from the testing set
		j = random.randint(0, len(test_index))
		if (Y[test_index[j]] == 0 and actives > inactives) or \
			(Y[test_index[j]] == 1 and inactives > actives):
			# if the compound is inavtive and there are more actives than inactives
			# or the comoound is active and there are more inactives than actives

			# increment the appropriate class
			if Y[test_index[j]] == 0:
				inactives += 1
			elif Y[test_index[j]] == 1:
				actives += 1

			# add the index in the testing array to the training array
			train_index = np.append(train_index, np.array(test_index[j]))

			# remove the element at the specified index from the testing array
			test_index = np.delete(test_index, j)

			# report to the user
			# print(f"actives: {actives}, inactives: {inactives}")
			count += 1

	# check the number of inactives and actives in the training set
	actives = 0
	inactives = 0
	for j in range(len(train_index)):
		if Y[train_index[j]] == 1:
			actives += 1
		else:
			inactives += 1

	print(f"{count} items were added to the training set.")
	print(f"Train set contains {len(train_index)} items ({actives} actives, {inactives} inactives)")
	print(f"Test set contains {len(test_index)} items")

	# create testing array, save to file
	save_path = save_dir + f"{save_title}/set{i}/"
	if not os.path.exists(save_path):
		# if the save path does not exist, create it
		os.mkdir(save_path)

	save_file = save_path + "test.csv"
	f = open(save_file, 'w')
	f.write("n,SMILE,class\n") # write header
	for l in range(len(test_index)):
		# write each entry in the testing set
		f.write(f"{l},{X[test_index[l]]},{Y[test_index[l]]}\n")
	f.close() # close the file


	# create training array, save to file
	save_file = save_dir + f"{save_title}/set{i}/train.csv"
	f = open(save_file, 'w')
	f.write("n,SMILE,class\n") # write the header
	for l in range(len(train_index)):
		# write each entry in the training set
		f.write(f"{l},{X[test_index[l]]},{Y[test_index[l]]}\n")
	f.close() # close the file





# save the data sets 