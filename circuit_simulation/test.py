from circuit_simulation.circuit_simulator import *
from circuit_simulation.basic_operations import *
from copy import copy


def see_rho_structure(amount_qubits):
    def kronecker(*args):
        vector = None
        for array in args:
            if vector is None:
                vector = array
                continue
            temp_array = (len(vector)*2) * [None]
            for i, element in enumerate(vector):
                temp_array[i*2] = element + array[0]
                temp_array[i*2+1] = element + array[1]
            vector = copy(temp_array)

        for k, element in enumerate(vector):
            vector[k] = element.split(".")
        return vector

    def CT(vector_1, vector_2=None):
        if vector_2 is None:
            vector_2 = copy(vector_1)
        result = np.zeros((len(vector_1), len(vector_1)), dtype=list)

        for i, element_1 in enumerate(vector_1):
            for j, element_2 in enumerate(vector_2):
                new_el = copy(element_1)
                new_el.extend(copy(element_2))
                newer_el = copy(new_el)
                for item in new_el:
                    if new_el.count(item) < 2:
                        newer_el.remove(item)
                result[i, j] = sorted(set(newer_el))

        return result

    def get_states():
        abc = list('abcdefghijklmnop')
        states = []
        if amount_qubits >= len(abc):
            raise ValueError("Amount of qubits to large")

        for i in range(amount_qubits):
            if i == amount_qubits-1:
                states.append([abc[i] + "1", abc[i] + "2"])
                continue
            states.append([abc[i]+"1.", abc[i]+"2."])
        return states

    result = kronecker(*get_states())

    return CT(result)


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

    quantumCircuit = QuantumCircuit(8, 1)
    qc2 = QuantumCircuit(7, 1)
    print(qc2)
    print(quantumCircuit)
    prob, dens = quantumCircuit._get_measurement_outcome_probability(1, 0)

    print(prob)
    print(dens)


    # qubits = 3
    # element_length = 0
    # for line in see_rho_structure(qubits):
    #     for content in line:
    #         print(str(content) + (element_length-len(str(content))) * " ", end=" | ")
    #         if element_length == 0:
    #             element_length = len(str(content))
    #     print()

    # show_gate()
    # two_qubit_gate_tryout()
    # smallest_circuit(X_gate, 1)
