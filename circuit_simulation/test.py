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


def show_gate():
    qc = QuantumCircuit(2, 0)
    theta = np.pi/2
    Ry_half_pi = SingleQubitGate("half pi y rotation",
                                 np.array([[np.cos(theta/2), -1 * np.sin(theta/2)],
                                           [1 * np.sin(theta/2), np.cos(theta/2)]]),
                                 "Ry(pi/2)")
    Rx_half_pi = SingleQubitGate("half pi x rotation",
                                 np.array([[np.cos(theta/2), -1j * np.sin(theta/2)],
                                           [-1j * np.sin(theta/2), np.cos(theta/2)]]),
                                 "Rx(pi/2)")
    Rx_minus_half_pi = SingleQubitGate("minus half pi x rotation",
                                       np.array([[np.cos(theta/2), 1j * np.sin(theta/2)],
                                                 [1j * np.sin(theta/2), np.cos(theta/2)]]),
                                       "Rx(-pi/2)")

    Z_gate_U = Z_gate.get_circuit_dimension_matrix(2, 0)
    S_gate_U = S_gate.get_circuit_dimension_matrix(2, 0)
    S_gate_U_dagger = S_gate.get_circuit_dimension_matrix(2, 0).conj().T
    Ry_half_pi_gate = Ry_half_pi.get_circuit_dimension_matrix(2, 1)
    Rx_minus_half_pi_gate = Rx_minus_half_pi.get_circuit_dimension_matrix(2, 1)
    Rx_half_pi_gate = Rx_half_pi.get_circuit_dimension_matrix(2, 1)
    NV_two_qubit_gate_U = NV_two_qubit_gate.get_circuit_dimension_matrix(2, 0, 1)
    S_gate_U_2 = S_gate.get_circuit_dimension_matrix(2, 1)
    S_gate_U_2_dagger = S_gate.get_circuit_dimension_matrix(2, 1).conj().T

    CY_gate = qc._create_2_qubit_gate(Y_gate, 0, 1)
    print(NV_two_qubit_gate)

    print("\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n{}\n\n".format(np.round(Z_gate_U.toarray(), 3),
                                                        np.round(S_gate_U.toarray(), 3),
                                                        np.round(Ry_half_pi_gate.toarray(), 3),
                                                        np.round(Rx_minus_half_pi_gate.toarray(), 3),
                                                        np.round(NV_two_qubit_gate_U.toarray(), 3),
                                                        np.round(Rx_half_pi_gate.toarray(), 3),
                                                        np.round(S_gate_U_2.toarray(), 3),
                                                        np.round(S_gate_U_2_dagger.toarray(), 3)))

    matrix = []
    for gate in [Z_gate_U, S_gate_U, Ry_half_pi_gate, NV_two_qubit_gate_U]:
        if matrix == []:
            matrix = gate
        matrix *= gate
    print(np.round(matrix.toarray(), 3))
    print("Equal to CY gate: {}\n\n".format(np.array_equal(np.round(matrix.toarray(), 3),
                                                           CY_gate.toarray())))
    matrix = []
    for gate in [Z_gate_U, S_gate_U, Ry_half_pi_gate, Rx_half_pi_gate, NV_two_qubit_gate_U, Rx_minus_half_pi_gate]:
        if matrix == []:
            matrix = gate
        matrix *= gate
    print(np.round(matrix.toarray(), 3))
    print("Equal to CZ gate: {}\n\n".format(np.array_equal(np.round(matrix.toarray(), 3),
                                                           CZ_gate.matrix)))
    matrix = []
    for gate in [Z_gate_U, S_gate_U, Ry_half_pi_gate, S_gate_U_2, NV_two_qubit_gate_U, S_gate_U_2_dagger]:
        if matrix == []:
            matrix = gate
        matrix *= gate
    print(np.round(matrix.toarray(), 3))
    print("Equal to CNOT gate: {}\n\n".format(np.array_equal(np.round(matrix.toarray(), 3),
                                                             CNOT_gate.matrix)))

    print()
    print(CY_gate.toarray())
    print(CZ_gate.matrix)
    print(CNOT_gate.matrix)


if __name__ == "__main__":

    show_gate()
    # two_qubit_gate_tryout()
    # smallest_circuit(X_gate, 1)
