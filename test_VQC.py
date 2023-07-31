import sys, os
from fujitsu.VQC.variational_quantum_classifier import VQC

path = "./data/test_activity_set.csv"

# create and test VQC object
vqc = VQC(qubits = 16)
vqc.load_data(path = path, smile_col = 'SMILES', class_col = 'single-class-label', verbose = True)
vqc.initialize_circuit(circuit = 'VC', QFT = False, bit_correlation = False)
# vqc.set_threshold(status = True)
# vqc.set_batching(status = True)
vqc.optimize()