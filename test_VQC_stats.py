import sys, os
from fujitsu.VQC.variational_quantum_classifier import VQC, optimize

opt_circuit_load_path = './fujitsu/data/QS/test_VQC_opt.yaml'
test_load_path = "./fujitsu/data/AChE/set0/train.csv"
save_path = './fujitsu/data/QS/'
save_title = 'test'

# load circuit
vqc = VQC(qubits = 11, state_prep = "AmplitudeEmbedding", fp_radius = 3)
vqc.load_circuit(load_path = opt_circuit_load_path)

# make predictions for testing set, get stats
vqc.load_data(load_path = test_load_path, smile_col = 'SMILE', class_col = 'class', verbose = False)
predictions = vqc.predict(save_dir = './fujitsu/data/QS/')
stats = vqc.get_stats(save_dir = save_path, title = save_title)