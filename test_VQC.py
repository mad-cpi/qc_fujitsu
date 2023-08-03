import sys, os
from fujitsu.VQC.variational_quantum_classifier import VQC, optimize


path = "./data/test_activity_set.csv"

# create and test VQC object
vqc = VQC(qubits = 16)
vqc.load_data(path = path, smile_col = 'SMILES', class_col = 'single-class-label', verbose = True)
vqc.initialize_circuit(circuit = 'TTN', QFT = True, bit_correlation = False)
vqc.set_threshold(status = True, verbose = True, threshold_initial = 0.5, threshold_max = 0.90, \
	threshold_increase_size = 0.01, threshold_increase_freq = 100)
vqc.set_batching(status = False)
optimize(vqc)