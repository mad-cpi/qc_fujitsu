import sys, os
from fujitsu.VQC.variational_quantum_classifier import VQC, optimize

## create and test VQC object
# initialize object
vqc = VQC(qubits = 28)

# load the LABE data
vqc.load_vector_data(load_path = './fujitsu/data/LBAE/plasmodium_malaria_subsampled_encoded_28_train.csv',\
	class_col = 'binary', vec_col = 'Encoded_Bit_Vector', verbose = True)
# TODO :: load bit vector from column data