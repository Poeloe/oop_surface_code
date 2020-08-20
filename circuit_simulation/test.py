from circuit_simulation.circuit_simulator import *


def smallest_circuit(operation, measure):
    qc = QuantumCircuit(8, 2)
    qc.add_top_qubit(ket_p)
    for i in range(1, qc.num_qubits, 2):
        qc.apply_2_qubit_gate(operation, 0, i)

    qc.measure_first_N_qubits(1, measure=measure)

    qc.save_density_matrix()

def two_qubit_gate_tryout():
    qc = QuantumCircuit(2, 0)
    qc.Ry(1, np.pi/2)
    qc.two_qubit_gate_NV(0, 1)
    qc.draw_circuit()
    print(qc)

if __name__ == "__main__":

    two_qubit_gate_tryout()
    # smallest_circuit(X_gate, 1)
