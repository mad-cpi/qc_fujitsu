import sys, os
import pandas as pd
import numpy as np
import qulacs 
from qulacs import QuantumState
from qulacs import QuantumCircuit
from qulacs.gate import H
import matplotlib.pyplot as plt
import seaborn as sns

""" Goal of this program is to generate a uniform
	superposition of n-quibits by apply Hadamar gates to
	each qubit independently. Then, verify that the 
	superposition is uniform by sampling the distribution of
	measured states m-times. """

## parameters ## 
verbose = False


## arguments ##
# number of qubits
n = int(sys.argv[1])
# number of times to sample quantum state
m = int(sys.argv[2])

print(f"\nGenerating {n}-qubit state.")

# initialize state of qubits
state = QuantumState(n)
state.set_zero_state()

# build quantum circuit
print(f"Applying Hadamar Gate to each qubit.")
circuit = QuantumCircuit(n)
for i in range(n):
	# add Hadamar gate to each wire
	gate = H(i)
	circuit.add_gate(gate)
	if verbose:
		print(gate)
if verbose:
	print(circuit)

# apply hadamar gate to all bits
# generating a uniform super position
circuit.update_quantum_state(state)

# measure the distribution by sampling m-times
# is it even?
print(f"Sampling quantum state {m}-times.")
data = state.sampling(m)
hist = [format(value, "b").zfill(n) for value in data]
# hist = [0] * ((n ** 2) - 1) # empty array containing all possible states
# for i in data:
# 	hist[i] += 1
# # generate histogram label
# label = [""] * len(hist)
# for i in range(len(label)):
# 	label[i] = format(i, "b").zfill(n)


state_dist = pd.DataFrame(dict(State=hist))
g = sns.displot(data=state_dist, x="State", palette='pastel', discrete=True)
plt.suptitle(f"({n} Qubits Sampled {m} Times)")
plt.title("Distribution of Quantum Measured Quantum States")
plt.xlabel("Unique Quantum State")
plt.ylabel("Number of Times Measured")
plt.show()