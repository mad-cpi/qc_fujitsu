import os, sys
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, matthews_corrcoef, \
	precision_score, recall_score, roc_curve, auc 


# Matthew Dorsey
# 2023.11.16
# program for testing predictions for a binary set made
# by a machine learning algorithm

## PARAMETERS
# default file name
default_filename = 'binary_model_stats.csv'

def get_binary_model_performace(Y_true, Y_pre, filename = None):

	# Y_true 	:: list of known binary classifications for a data set
	# 				(only zero and one are acceptable)
	# Y_pred	:: list of predictions for a set of binary classifications,
	# 				made by a model (can be any number between zero and one,
	#				the program assumes that number less than 0.5 are 0, etc.)
	# filename 	:: (optional) path to directory where stats should be saved,
	#				and name of file, (e.g. 'results/VQC_1024.csv')

	# if a path was specified by the user,
	# check that it exists
	if filename is not None:
		# check that the path exists
		if not os.path.exists(save_dir):
			# if it does not, make the path
			os.mkdir(save_dir)
		else:
			# if the path does not exist, use the default
			filename = default_filename
	else:
		# if a file name was not passed to the method, assign the default
		filename = default_filename

	# generate file names
	if title is not None:
		title = 'VQC'
	file_stats = title + '_stats.csv'
	file_pred = title + '_true_pred.csv'
	file_roc = title + '_roc.csv'

	# transform values to binary
	Y_pred = [1 if p >= 0.5 else 0 for p in Y_pred]

	# get stats
	stat_dict = {}
	stat_dict['acc'] = accuracy_score(Y_true, Y_pred)
	stat_dict['f1s'] = f1_score(Y_true, Y_pred)
	stat_dict['cks'] = cohen_kappa_score(Y_true, Y_pred)
	stat_dict['mcc'] = matthews_corrcoef(Y_true, Y_pred)
	stat_dict['pre'] = precision_score(Y_true, Y_pred)
	stat_dict['rec'] = recall_score(Y_true, Y_pred) 

	# print stats
	df_stat = pd.DataFrame(stat_dict, index = [0])
	df_stat.to_csv(filename, index = False)
