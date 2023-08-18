import sys, os
from fujitsu.VQC.circuit_architecture import TreeTensorNetwork

# create TTN circuit for 3 qubits
ttn = TreeTensorNetwork(16)

# test the get qubits layer function
print (f"\nThe qubits which are initialized for each TTN layer are:")
for i in range(4):
	print(f"LAYER {i} :: QUBTIS {ttn.get_layer_qubits(i)}")

# test the get layer weights function
W, __ = ttn.initial_weights()
W = W[:-1] # remove bias
print (f"\nTTN circuit with {16} qubits contains total of {len(W)} unitary operations.")
for i in range(4):
	# check the length of each of the weights returned from the function
	print(len(ttn.get_layer_weights(W, i)))