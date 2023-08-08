import sys, os
from fujitsu.VQC.variational_quantum_classifier import VQC, optimize


path = "./fujitsu/data/AChE/set0/train.csv"

# create and test VQC object
vqc = VQC(qubits = 12 , state_prep = "AmplitudeEmbedding")
vqc.load_data(path = path, smile_col = 'SMILE', class_col = 'class', verbose = False)
vqc.initialize_circuit(circuit = 'VC', bit_correlation = False)
optimize(vqc)