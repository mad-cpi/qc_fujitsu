import sys, os
from fujitsu.VQC.variational_quantum_classifier import VQC, optimize


path = "./fujitsu/data/AChE/set0/train.csv"

# create and test VQC object
vqc = VQC(qubits = 12, state_prep = "AmplitudeEmbedding")
vqc.load_data(path = path, smile_col = 'SMILE', class_col = 'class', verbose = False)
vqc.initialize_circuit(circuit = 'VC', bit_correlation = False)
vqc.set_threshold(status = True, verbose = True, threshold_initial = 0.02, threshold_max = 0.90, \
	threshold_increase_size = 0.01, threshold_increase_freq = 100)
vqc.set_batching(status = False)
optimize(vqc)