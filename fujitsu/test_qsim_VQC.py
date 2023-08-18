from qulacs import QuantumCircuit, QuantumState
from VQC.variational_quantum_classifier import VQC, optimize

path = "./fujitsu/data/QS/MPI_test/"
train_set = "./fujitsu/data/AChE/set0/train.csv"
test_set = "./fujitsu/data/AChE/set0/test.csv"
title = "testMPI"

# set up MPI
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# print (rank)

# create vqc object
vqc = VQC(qubits = 28, state_prep = "BasisEmbedding", fp_radius = 4)

# load training data
vqc.load_data(load_path = train_set, smile_col = 'SMILE', class_col = 'class', verbose = False)

# initialize the circuit
vqc.initialize_circuit(circuit = 'VC', bit_correlation = False)

# save circuit weights
vqc.save_circuit(save_to = path, save_as = title + '_init')
optimize(vqc, save_dir = path, title = 'test', max_opt_steps = 50)
vqc.save_circuit(save_to = path, save_as = title + '_opt')

# load testing set, make predictions and score model performance
vqc.load_data(load_path = test_set, smile_col = 'SMILE', class_col = 'class', verbose = False)
vqc.predict()
stats = vqc.get_stats(save_dir = path, title = title)

# inform user
print(f"Mode stats after optimization for testing set({test_set}):\n{stats}")