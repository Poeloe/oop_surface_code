from circuit_simulation.circuit_simulator import *


def smallest_circuit(operation, measure):
    qc = QuantumCircuit(8, 2)
    qc.add_top_qubit(ket_p)
    for i in range(1, qc.num_qubits, 2):
        qc.apply_2_qubit_gate(operation, 0, i)

    qc.measure_first_N_qubits(1, measure=measure)

    qc.save_density_matrix()


if __name__ == "__main__":

    smallest_circuit(X, 1)
