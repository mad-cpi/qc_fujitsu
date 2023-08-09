import sys, os
from fujitsu.VQC.variational_quantum_classifier import VQC, optimize


for i in range(5):
	
	path = f"./fujitsu/data/AChE/set{i}/"
	title = 'test_VQC'
	train_set = path + 'train.csv'
	test_set = path + 'test.csv'

	## create and test VQC object
	# initialize object
	vqc = VQC(qubits = 12, state_prep = "AmplitudeEmbedding", fp_radius = 6)

	# load training data
	vqc.load_data(path = train_data, smile_col = 'SMILE', class_col = 'class', verbose = False)
	vqc.initialize_circuit(circuit = 'VC', bit_correlation = False)

	# train circuit, save circuit weights before and after training
	vqc.save_circuit(save_to = path, save_as = title + 'init')
	optimize(vqc, save_dir = path, title = 'test')
	vqc.save_circuit(save_to = path, save_as = title + 'opt')

	# load testing set, make predictions and score model performance
	vqc.load_data(path = test_set, smile_col = 'SMILE', class_col = 'class', verbose = False)
	vqc.predict()
	stats = vqc.get_stats(save_dir = path, title = title)

	# inform user
	print(f"Mode stats after optimization for testing set({test_set}):\n{stats}")
