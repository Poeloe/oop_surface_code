import sys
sys.path.insert(1, '/Users/Paul/Documents/TU/Master/Afstuderen/forked_surface_code/oop_surface_code/')
from circuit_simulation.circuit_simulator import *


def monolithic(operation):
    qc = QuantumCircuit(8, 2, noise=True, pg=0.009, pm=0.009)
    qc.add_top_qubit(ket_p)
    qc.apply_2_qubit_gate(operation, 0, 1)
    qc.apply_2_qubit_gate(operation, 0, 3)
    qc.apply_2_qubit_gate(operation, 0, 5)
    qc.apply_2_qubit_gate(operation, 0, 7)
    qc.measure_first_N_qubits(1)

    qc.draw_circuit()
    qc.get_superoperator([0, 2, 4, 6], gate_name(operation))


def expedient(operation):
    qc = QuantumCircuit(8, 2, noise=True, pg=0.006, pm=0.006, pn=0.1)

    # Noisy ancilla Bell pair is now between are now 0 and 1
    qc.create_bell_pairs_top(1, new_qubit=True)
    qc.double_selection(Z, new_qubit=True)
    qc.double_selection(X)

    # New noisy ancilla Bell pair is now between 0 and 1, old ancilla Bell pair now between 2 and 3
    qc.create_bell_pairs_top(1, new_qubit=True)
    qc.double_selection(Z, new_qubit=True)
    qc.double_selection(X)

    # Now entanglement between ancilla 0 and 3 is made
    qc.single_dot(Z, 2, 5)
    qc.single_dot(Z, 2, 5)

    # And finally the entanglement between ancilla 1 and 1 is made, now all ancilla's are entangled
    qc.single_dot(Z, 3, 4)
    qc.single_dot(Z, 3, 4)

    qc.apply_2_qubit_gate(operation, 0, 4)
    qc.apply_2_qubit_gate(operation, 1, 6)
    qc.apply_2_qubit_gate(operation, 2, 8)
    qc.apply_2_qubit_gate(operation, 3, 10)

    qc.measure_first_N_qubits(4)

    qc.draw_circuit()
    qc.get_superoperator([0, 2, 4, 6], gate_name(operation))


def stringent(operation):
    qc = QuantumCircuit(8, 2, noise=True, pg=0.0075, pm=0.0075, pn=0.1)

    # Noisy ancilla Bell pair between 0 and 1
    qc.create_bell_pairs_top(1, new_qubit=True)
    qc.double_selection(Z, new_qubit=True)
    qc.double_selection(X)
    qc.double_dot(Z, 2, 3)
    qc.double_dot(X, 2, 3)

    # New noisy ancilla Bell pair is now between 0 and 1, old ancilla Bell pair now between 2 and 3
    qc.create_bell_pairs_top(1, new_qubit=True)
    qc.double_selection(Z, new_qubit=True)
    qc.double_selection(X)
    qc.double_dot(X, 2, 3)
    qc.double_dot(X, 2, 3)

    # Now entanglement between ancilla 0 and 3 is made
    qc.double_dot(Z, 2, 5)
    qc.double_dot(Z, 2, 5)

    # And finally the entanglement between ancilla 1 and 1 is made, now all ancilla's are entangled
    qc.double_dot(Z, 3, 4)
    qc.double_dot(Z, 3, 4)

    qc.apply_2_qubit_gate(operation, 3, 10)
    qc.apply_2_qubit_gate(operation, 2, 8)
    qc.apply_2_qubit_gate(operation, 1, 6)
    qc.apply_2_qubit_gate(operation, 0, 4)

    qc.measure_first_N_qubits(4)

    qc.draw_circuit()
    qc.get_superoperator([0, 2, 4, 6], gate_name(operation))


if __name__ == "__main__":
    # monolithic(Z)
    # monolithic(X)
    expedient(Z)
    # expedient(X)
    # stringent(Z)
    # stringent(X)
