from circuit_simulation.circuit_simulator import *


def smallest_circuit():
    qc = QuantumCircuit(2, 2, noise=False)

    qc.add_top_qubit(ket_0)
    qc.CNOT(0, 1)

    qc.measure_first_N_qubits(1, noise=False)
    qc.draw_circuit()
    qc.get_kraus_operator()


def easy_circuit():
    qc = QuantumCircuit(2, 2, noise=False)

    qc.X(0)
    qc.X(1)

    qc.draw_circuit()
    qc.get_kraus_operator()


if __name__ == "__main__":

    smallest_circuit()
