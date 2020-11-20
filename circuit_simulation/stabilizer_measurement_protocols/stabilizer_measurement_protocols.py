from circuit_simulation.circuit_simulator import *
import pickle
PBAR = None


def create_quantum_circuit(protocol, pbar, **kwargs):
    """
        Initialises a QuantumCircuit object corresponding to the protocol requested.

        Parameters
        ----------
        protocol : str
            Name of the protocol for which the QuantumCircuit object should be initialised

        For other parameters, please see QuantumCircuit class for more information

    """
    global PBAR
    PBAR = pbar

    if protocol == 'monolithic':
        kwargs.pop('basis_transformation_noise')
        kwargs.pop('no_single_qubit_error')
        qc = QuantumCircuit(9, 2, basis_transformation_noise=True, no_single_qubit_error=False, **kwargs)

    elif protocol == 'duo_structure':
        qc = QuantumCircuit(14, 2, **kwargs)

        qc.define_node("A", qubits=[0, 1, 2, 6, 8], electron_qubits=2, data_qubits=[6, 8])
        qc.define_node("B", qubits=[3, 4, 5, 10, 12], electron_qubits=5, data_qubits=[10, 12])

        qc.define_sub_circuit("AB")

    elif protocol == 'duo_structure_2':
        qc = QuantumCircuit(32, 5, **kwargs)

        qc.define_node("A", qubits=[30, 28, 15, 14, 13, 12], electron_qubits=12, data_qubits=[30, 28])
        qc.define_node("B", qubits=[26, 24, 11, 10, 9, 8], electron_qubits=8, data_qubits=[26, 24])
        qc.define_node("C", qubits=[22, 20, 7, 6, 5, 4], electron_qubits=4, data_qubits=[22, 10])
        qc.define_node("D", qubits=[18, 16, 3, 2, 1, 0], electron_qubits=0, data_qubits=[18, 16])

        qc.define_sub_circuit("AB")
        qc.define_sub_circuit("CD", concurrent_sub_circuits="AB")
        qc.define_sub_circuit("AC")
        qc.define_sub_circuit("BD", concurrent_sub_circuits="AC")
        qc.define_sub_circuit("A")
        qc.define_sub_circuit("B")
        qc.define_sub_circuit("C")
        qc.define_sub_circuit("D", concurrent_sub_circuits=["A", "B", "C"])

    elif protocol == 'dyn_prot_14_1':
        qc = QuantumCircuit(22, 2, **kwargs)

        qc.define_node("A", qubits=[20, 13, 12, 11, 10], electron_qubits=10, data_qubits=20, ghz_qubits=13)
        qc.define_node("B", qubits=[18, 9, 8, 7], electron_qubits=7, data_qubits=18, ghz_qubits=9)
        qc.define_node("C", qubits=[16, 6, 5, 4, 3], electron_qubits=3, data_qubits=16, ghz_qubits=6)
        qc.define_node("D", qubits=[14, 2, 1, 0], electron_qubits=0, data_qubits=14, ghz_qubits=2)

        qc.define_sub_circuit("AB")
        qc.define_sub_circuit("CD", concurrent_sub_circuits="AB")
        qc.define_sub_circuit("AC")
        qc.define_sub_circuit("BD", concurrent_sub_circuits="AC")
        qc.define_sub_circuit("A")
        qc.define_sub_circuit("B")
        qc.define_sub_circuit("C")
        qc.define_sub_circuit("D", concurrent_sub_circuits=["A", "B", "C"])

    elif protocol == 'dyn_prot_22_1':
        qc = QuantumCircuit(24, 2, **kwargs)

        qc.define_node("A", qubits=[22, 15, 14, 13, 12], electron_qubits=12, data_qubits=22, ghz_qubits=15)
        qc.define_node("B", qubits=[20, 11, 10, 9, 8], electron_qubits=8, data_qubits=20, ghz_qubits=11)
        qc.define_node("C", qubits=[18, 7, 6, 5, 4], electron_qubits=4, data_qubits=18, ghz_qubits=7)
        qc.define_node("D", qubits=[16, 3, 2, 1, 0], electron_qubits=0, data_qubits=16, ghz_qubits=3)

        qc.define_sub_circuit("AB")
        qc.define_sub_circuit("CD", concurrent_sub_circuits="AB")
        qc.define_sub_circuit("AC")
        qc.define_sub_circuit("BD", concurrent_sub_circuits="AC")
        qc.define_sub_circuit("A")
        qc.define_sub_circuit("B")
        qc.define_sub_circuit("C")
        qc.define_sub_circuit("D", concurrent_sub_circuits=["A", "B", "C"])

    elif protocol == 'dyn_prot_42_1':
        qc = QuantumCircuit(28, 2, **kwargs)

        qc.define_node("A", qubits=[26, 19, 18, 17, 16, 15], electron_qubits=15, data_qubits=26, ghz_qubits=19)
        qc.define_node("B", qubits=[24, 14, 13, 12, 11, 10], electron_qubits=10, data_qubits=24, ghz_qubits=14)
        qc.define_node("C", qubits=[22, 9, 8, 7, 6, 5], electron_qubits=5, data_qubits=22, ghz_qubits=9)
        qc.define_node("D", qubits=[20, 4, 3, 2, 1, 0], electron_qubits=0, data_qubits=20, ghz_qubits=4)

        qc.define_sub_circuit("AB")
        qc.define_sub_circuit("CD", concurrent_sub_circuits="AB")
        qc.define_sub_circuit("AC")
        qc.define_sub_circuit("BD", concurrent_sub_circuits="AC")
        qc.define_sub_circuit("A")
        qc.define_sub_circuit("B")
        qc.define_sub_circuit("C")
        qc.define_sub_circuit("D", concurrent_sub_circuits=["A", "B", "C"])

    elif protocol in ['bipartite_4', 'bipartite_5', 'bipartite_6', 'bipartite_7', 'bipartite_8', 'bipartite_9',
                      'bipartite_10', 'bipartite_11', 'bipartite_12']:
        qc = QuantumCircuit(28, 6, **kwargs)

        qc.define_node("A", qubits=[26, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12], electron_qubits=12,
                       data_qubits=26, ghz_qubits=12)
        qc.define_node("B", qubits=[24, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0], electron_qubits=0, data_qubits=24,
                       ghz_qubits=0)

        qc.define_sub_circuit("AB")

        qc.define_sub_circuit("A")
        qc.define_sub_circuit("B", concurrent_sub_circuits=["A"])

    else:
        qc = QuantumCircuit(20, 2, **kwargs)

        qc.define_node("A", qubits=[18, 11, 10, 9], electron_qubits=9, data_qubits=18, ghz_qubits=11)
        qc.define_node("B", qubits=[16, 8, 7, 6], electron_qubits=6, data_qubits=16, ghz_qubits=8)
        qc.define_node("C", qubits=[14, 5, 4, 3], electron_qubits=3, data_qubits=14, ghz_qubits=5)
        qc.define_node("D", qubits=[12, 2, 1, 0], electron_qubits=0, data_qubits=12, ghz_qubits=2)

        qc.define_sub_circuit("AB")
        qc.define_sub_circuit("CD", concurrent_sub_circuits="AB")
        qc.define_sub_circuit("AC")
        qc.define_sub_circuit("BD", concurrent_sub_circuits="AC")
        qc.define_sub_circuit("A")
        qc.define_sub_circuit("B")
        qc.define_sub_circuit("C")
        qc.define_sub_circuit("D", concurrent_sub_circuits=["A", "B", "C"])

    return qc


def monolithic(qc: QuantumCircuit, *, operation):
    qc.set_qubit_states({0: ket_p})
    qc.apply_gate(operation, cqubit=0, tqubit=1)
    qc.apply_gate(operation, cqubit=0, tqubit=3)
    qc.apply_gate(operation, cqubit=0, tqubit=5)
    qc.apply_gate(operation, cqubit=0, tqubit=7)
    qc.measure(0, probabilistic=False)

    PBAR.update(50) if PBAR is not None else None

    return [1, 3, 5, 7]


def expedient(qc: QuantumCircuit, *, operation):
    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None

        # Step 1-2 from Table D.1 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("AB")
        success_ab = False
        while not success_ab:
            qc.create_bell_pair(11, 8)
            success_ab = qc.double_selection(CZ_gate, 10, 7, retry=False)
            if not success_ab:
                continue
            success_ab = qc.double_selection(CNOT_gate, 10, 7, retry=False)

        PBAR.update(20) if PBAR is not None else None

        # Step 1-2 from Table D.1 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("CD")
        success_cd = False
        while not success_cd:
            qc.create_bell_pair(5, 2)
            success_cd = qc.double_selection(CZ_gate, 4, 1, retry=False)
            if not success_cd:
                continue
            success_cd = qc.double_selection(CNOT_gate, 4, 1, retry=False)

        PBAR.update(20) if PBAR is not None else None

        # Step 3-5 from Table D.1 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("AC")
        # Return success (even parity measurement). If False (uneven), X-gate must be drawn at second single dot
        success_1 = qc.single_dot(CZ_gate, 10, 4, parity_check=False)
        qc.start_sub_circuit("BD")
        ghz_success = qc.single_dot(CZ_gate, 7, 1, draw_X_gate=not success_1, retry=False)
        if not ghz_success:
            continue

        PBAR.update(20) if PBAR is not None else None

        # Step 6-8 from Table D.1 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("AC", forced_level=True)
        ghz_success_1 = qc.single_dot(CZ_gate, 10, 4, retry=False)
        qc.start_sub_circuit("BD")
        ghz_success_2 = qc.single_dot(CZ_gate, 7, 1, retry=False)
        if any([not ghz_success_1, not ghz_success_2]):
            ghz_success = False
            continue

        PBAR.update(20) if PBAR is not None else None

    qc.get_state_fidelity()

    # Step 9 from Table D.1 (Thesis Naomi Nickerson)
    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    qc.start_sub_circuit("B")
    qc.apply_gate(operation, cqubit=8, tqubit=16)
    qc.measure(8, probabilistic=False)

    qc.start_sub_circuit("A")
    qc.apply_gate(operation, cqubit=11, tqubit=18)
    qc.measure(11, probabilistic=False)

    qc.start_sub_circuit("D")
    qc.apply_gate(operation, cqubit=2, tqubit=12)
    qc.measure(2, probabilistic=False)

    qc.start_sub_circuit("C")
    qc.apply_gate(operation, cqubit=5, tqubit=14)
    qc.measure(5, probabilistic=False)

    qc.end_current_sub_circuit(total=True)

    PBAR.update(10) if PBAR is not None else None


def bipartite_4(qc: QuantumCircuit, *, operation):
    # ['CNOT32', 'CNOT30', 'CNOT21', 'H2', 'CNOT02', 'H2', 'H1']
    qc.start_sub_circuit("AB")
    qc.create_bell_pair(3, 15)
    qc.create_bell_pair(2, 14)
    qc.create_bell_pair(1, 13)
    qc.create_bell_pair(0, 12)
    qc.apply_gate(CNOT_gate, cqubit=3, tqubit=2)    # 15, 3, 14, 2
    qc.apply_gate(CNOT_gate, cqubit=15, tqubit=14)
    qc.apply_gate(CNOT_gate, cqubit=3, tqubit=0)    # 15, 3, 14, 2, 12, 0
    qc.apply_gate(CNOT_gate, cqubit=15, tqubit=12)
    qc.apply_gate(CNOT_gate, cqubit=2, tqubit=1, reverse=True)    # 13, 1, 15, 3, 14, 2, 12, 0
    qc.apply_gate(CNOT_gate, cqubit=14, tqubit=13)
    qc.apply_gate(H_gate, 2)
    qc.apply_gate(H_gate, 14)
    qc.apply_gate(CNOT_gate, cqubit=0, tqubit=2)
    qc.apply_gate(CNOT_gate, cqubit=12, tqubit=14)
    qc.apply_gate(H_gate, 2)
    qc.apply_gate(H_gate, 14)
    qc.apply_gate(H_gate, 1)
    qc.apply_gate(H_gate, 13)

    qc.measure([13, 1, 15, 3, 14, 2], probabilistic=False)

    qc.get_state_fidelity()

    qc.start_sub_circuit("A")
    qc.apply_gate(operation, cqubit=12, tqubit=26)
    qc.measure(12, probabilistic=False)

    qc.start_sub_circuit("B")
    qc.apply_gate(operation, cqubit=0, tqubit=24)
    qc.measure(0, probabilistic=False)

    qc.end_current_sub_circuit(total=True)


def bipartite_5(qc: QuantumCircuit, *, operation):
    # ['CNOT30', 'CNOT32', 'CNOT40', 'CZ20', 'CZ21', 'CZ41', 'CZ42', 'CZ43']
    qc.start_sub_circuit("AB")
    qc.create_bell_pair(4, 16)
    qc.create_bell_pair(3, 15)
    qc.create_bell_pair(2, 14)
    qc.create_bell_pair(1, 13)
    qc.create_bell_pair(0, 12)
    qc.apply_gate(CNOT_gate, cqubit=3, tqubit=0)    # 15, 3, 12, 0
    qc.apply_gate(CNOT_gate, cqubit=15, tqubit=12)
    qc.apply_gate(CNOT_gate, cqubit=3, tqubit=2, reverse=True)    # 14, 2, 15, 3, 12, 0
    qc.apply_gate(CNOT_gate, cqubit=15, tqubit=14)
    qc.apply_gate(CNOT_gate, cqubit=4, tqubit=0)    # 16, 4, 14, 2, 15, 3, 12, 0
    qc.apply_gate(CNOT_gate, cqubit=16, tqubit=12)
    qc.apply_gate(CZ_gate, cqubit=2, tqubit=0)
    qc.apply_gate(CZ_gate, cqubit=14, tqubit=12)
    qc.apply_gate(CZ_gate, cqubit=2, tqubit=1, reverse=True)    # 13, 1, 16, 4, 14, 2, 15, 3, 12, 0
    qc.apply_gate(CZ_gate, cqubit=14, tqubit=13)
    qc.apply_gate(CZ_gate, cqubit=4, tqubit=1)
    qc.apply_gate(CZ_gate, cqubit=16, tqubit=13)
    qc.apply_gate(CZ_gate, cqubit=4, tqubit=2)
    qc.apply_gate(CZ_gate, cqubit=16, tqubit=14)
    qc.apply_gate(CZ_gate, cqubit=4, tqubit=3)
    qc.apply_gate(CZ_gate, cqubit=16, tqubit=15)

    qc.measure([13, 1, 16, 4, 14, 2, 15, 3], probabilistic=False)

    qc.get_state_fidelity()

    qc.start_sub_circuit("A")
    qc.apply_gate(operation, cqubit=12, tqubit=26)
    qc.measure(12, probabilistic=False)

    qc.start_sub_circuit("B")
    qc.apply_gate(operation, cqubit=0, tqubit=24)
    qc.measure(0, probabilistic=False)

    qc.end_current_sub_circuit(total=True)


def bipartite_6(qc: QuantumCircuit, *, operation):

    # T = (1 / math.sqrt(2)) * sp.csr_matrix([[1, 0, 0, 1], [0, 1, 1, 0], [1, 0, 0, -1], [0, 1, -1, 0]])
    # ['CNOT20', 'CNOT21', 'CNOT40', 'CNOT32', 'CZ45', 'CZ21', 'CZ02', 'CZ41', 'H5', 'H4', 'H3', 'H2', 'H1']

    qc.start_sub_circuit("AB")
    # qc.create_bell_pair(11, 23)
    # qc.create_bell_pair(10, 22)
    # qc.create_bell_pair(9, 21)
    # qc.create_bell_pair(8, 20)
    # qc.create_bell_pair(7, 19)
    # qc.create_bell_pair(6, 18)
    qc.create_bell_pair(5, 17)
    qc.create_bell_pair(4, 16)
    qc.create_bell_pair(3, 15)
    qc.create_bell_pair(2, 14)
    qc.create_bell_pair(1, 13)
    qc.create_bell_pair(0, 12)
    qc.apply_gate(CNOT_gate, cqubit=2, tqubit=0)    # 14, 2, 12, 0
    qc.apply_gate(CNOT_gate, cqubit=14, tqubit=12)
    qc.apply_gate(CNOT_gate, cqubit=2, tqubit=1, reverse=True)    # 13, 1, 14, 2, 12, 0
    qc.apply_gate(CNOT_gate, cqubit=14, tqubit=13)
    qc.apply_gate(CNOT_gate, cqubit=4, tqubit=0)    # 16, 4, 13, 1, 14, 2, 12, 0
    qc.apply_gate(CNOT_gate, cqubit=16, tqubit=12)
    qc.apply_gate(CNOT_gate, cqubit=3, tqubit=2)    # 15, 3, 16, 4, 13, 1, 14, 2, 12, 0
    qc.apply_gate(CNOT_gate, cqubit=15, tqubit=14)
    qc.apply_gate(CZ_gate, cqubit=4, tqubit=5, reverse=True)    # 17, 5, 15, 3, 16, 4, 13, 1, 14, 2, 12, 0
    qc.apply_gate(CZ_gate, cqubit=16, tqubit=17)
    qc.apply_gate(CZ_gate, cqubit=2, tqubit=1)
    qc.apply_gate(CZ_gate, cqubit=14, tqubit=13)
    qc.apply_gate(CZ_gate, cqubit=0, tqubit=2)
    qc.apply_gate(CZ_gate, cqubit=12, tqubit=14)
    qc.apply_gate(CZ_gate, cqubit=4, tqubit=1)
    qc.apply_gate(CZ_gate, cqubit=16, tqubit=13)

    measurement_outcomes = qc.measure([17, 5, 15, 3, 16, 4, 13, 1, 14, 2], probabilistic=False)
    # qc.draw_circuit()
    # qc.append_print_lines(T*(qc.get_combined_density_matrix([0, 8])[0])*T.transpose())
    # qc.append_print_lines("\n ")
    # qc.append_print_lines("Measurement outcomes are {}".format(measurement_outcomes))

    qc.get_state_fidelity()

    qc.start_sub_circuit("A")
    qc.apply_gate(operation, cqubit=12, tqubit=26)
    qc.measure(12, probabilistic=False)

    qc.start_sub_circuit("B")
    qc.apply_gate(operation, cqubit=0, tqubit=24)
    qc.measure(0, probabilistic=False)

    qc.end_current_sub_circuit(total=True)


def bipartite_7(qc: QuantumCircuit, *, operation):
    # ['CNOT43', 'CNOT20', 'CNOT42', 'CNOT10', 'CZ56', 'CZ51', 'CZ20', 'CZ31', 'CZ62', 'CZ32', 'CZ54']
    qc.start_sub_circuit("AB")
    qc.create_bell_pair(6, 18)
    qc.create_bell_pair(5, 17)
    qc.create_bell_pair(4, 16)
    qc.create_bell_pair(3, 15)
    qc.create_bell_pair(2, 14)
    qc.create_bell_pair(1, 13)
    qc.create_bell_pair(0, 12)
    qc.apply_gate(CNOT_gate, cqubit=4, tqubit=3)    # 16, 4, 15, 3
    qc.apply_gate(CNOT_gate, cqubit=16, tqubit=15)
    qc.apply_gate(CNOT_gate, cqubit=2, tqubit=0)    # 14, 2, 12, 0
    qc.apply_gate(CNOT_gate, cqubit=14, tqubit=12)
    qc.apply_gate(CNOT_gate, cqubit=4, tqubit=2)    # 16, 4, 15, 3, 14, 2, 12, 0
    qc.apply_gate(CNOT_gate, cqubit=16, tqubit=14)
    qc.apply_gate(CNOT_gate, cqubit=1, tqubit=0)    # 13, 1, 16, 4, 15, 3, 14, 2, 12, 0
    qc.apply_gate(CNOT_gate, cqubit=13, tqubit=12)
    qc.apply_gate(CZ_gate, cqubit=5, tqubit=6)      # 17, 5, 18, 6
    qc.apply_gate(CZ_gate, cqubit=17, tqubit=18)
    qc.apply_gate(CZ_gate, cqubit=5, tqubit=1)      # 17, 5, 18, 6, 13, 1, 16, 4, 15, 3, 14, 2, 12, 0
    qc.apply_gate(CZ_gate, cqubit=17, tqubit=13)
    qc.apply_gate(CZ_gate, cqubit=2, tqubit=0)
    qc.apply_gate(CZ_gate, cqubit=14, tqubit=12)
    qc.apply_gate(CZ_gate, cqubit=3, tqubit=1)
    qc.apply_gate(CZ_gate, cqubit=15, tqubit=13)
    qc.apply_gate(CZ_gate, cqubit=6, tqubit=2)
    qc.apply_gate(CZ_gate, cqubit=18, tqubit=14)
    qc.apply_gate(CZ_gate, cqubit=3, tqubit=2)
    qc.apply_gate(CZ_gate, cqubit=15, tqubit=14)
    qc.apply_gate(CZ_gate, cqubit=5, tqubit=4)
    qc.apply_gate(CZ_gate, cqubit=17, tqubit=16)

    qc.measure([17, 5, 18, 6, 13, 1, 16, 4, 15, 3, 14, 2], probabilistic=False)

    qc.get_state_fidelity()     # [0.9452874234023928, 0.015611321855850608, 0.015611321855850608, 0.023489932885906045]

    qc.start_sub_circuit("A")
    qc.apply_gate(operation, cqubit=12, tqubit=26)
    qc.measure(12, probabilistic=False)

    qc.start_sub_circuit("B")
    qc.apply_gate(operation, cqubit=0, tqubit=24)
    qc.measure(0, probabilistic=False)

    qc.end_current_sub_circuit(total=True)


def bipartite_8(qc: QuantumCircuit, *, operation):
    # ['CNOT72', 'CNOT21', 'CNOT73', 'CNOT30', 'CNOT65', 'CNOT76', 'CNOT53', 'CZ45', 'CZ24', 'CZ62', 'CZ63', 'CZ60', 'CZ13']
    qc.start_sub_circuit("AB")
    qc.create_bell_pair(7, 19)
    qc.create_bell_pair(6, 18)
    qc.create_bell_pair(5, 17)
    qc.create_bell_pair(4, 16)
    qc.create_bell_pair(3, 15)
    qc.create_bell_pair(2, 14)
    qc.create_bell_pair(1, 13)
    qc.create_bell_pair(0, 12)
    qc.apply_gate(CNOT_gate, cqubit=7, tqubit=2)    # 19, 7, 14, 2
    qc.apply_gate(CNOT_gate, cqubit=19, tqubit=14)
    qc.apply_gate(CNOT_gate, cqubit=2, tqubit=1)    # 19, 7, 14, 2, 13, 1
    qc.apply_gate(CNOT_gate, cqubit=14, tqubit=13)
    qc.apply_gate(CNOT_gate, cqubit=7, tqubit=3)    # 19, 7, 14, 2, 13, 1, 15, 3
    qc.apply_gate(CNOT_gate, cqubit=19, tqubit=15)
    qc.apply_gate(CNOT_gate, cqubit=3, tqubit=0)    # 19, 7, 14, 2, 13, 1, 15, 3, 12, 0
    qc.apply_gate(CNOT_gate, cqubit=15, tqubit=12)
    qc.apply_gate(CNOT_gate, cqubit=6, tqubit=5)    # 18, 6, 17, 5
    qc.apply_gate(CNOT_gate, cqubit=18, tqubit=17)
    qc.apply_gate(CNOT_gate, cqubit=7, tqubit=6, reverse=True)    # 18, 6, 17, 5, 19, 7, 14, 2, 13, 1, 15, 3, 12, 0
    qc.apply_gate(CNOT_gate, cqubit=19, tqubit=18)
    qc.apply_gate(CNOT_gate, cqubit=5, tqubit=3)
    qc.apply_gate(CNOT_gate, cqubit=17, tqubit=15)
    qc.apply_gate(CZ_gate, cqubit=4, tqubit=5)      # 16, 4, 18, 6, 17, 5, 19, 7, 14, 2, 13, 1, 15, 3, 12, 0
    qc.apply_gate(CZ_gate, cqubit=16, tqubit=17)
    qc.apply_gate(CZ_gate, cqubit=2, tqubit=4)
    qc.apply_gate(CZ_gate, cqubit=14, tqubit=16)
    qc.apply_gate(CZ_gate, cqubit=6, tqubit=2)
    qc.apply_gate(CZ_gate, cqubit=18, tqubit=14)
    qc.apply_gate(CZ_gate, cqubit=6, tqubit=3)
    qc.apply_gate(CZ_gate, cqubit=18, tqubit=15)
    qc.apply_gate(CZ_gate, cqubit=6, tqubit=0)
    qc.apply_gate(CZ_gate, cqubit=18, tqubit=12)
    qc.apply_gate(CZ_gate, cqubit=1, tqubit=3)
    qc.apply_gate(CZ_gate, cqubit=13, tqubit=15)

    qc.measure([16, 4, 18, 6, 17, 5, 19, 7, 14, 2, 13, 1, 15, 3], probabilistic=False)

    qc.get_state_fidelity()     # [0.9564483457123565, 0.012997974341661047, 0.012997974341661049, 0.017555705604321417]

    qc.start_sub_circuit("A")
    qc.apply_gate(operation, cqubit=12, tqubit=26)
    qc.measure(12, probabilistic=False)

    qc.start_sub_circuit("B")
    qc.apply_gate(operation, cqubit=0, tqubit=24)
    qc.measure(0, probabilistic=False)

    qc.end_current_sub_circuit(total=True)


def dyn_prot_14_1(qc: QuantumCircuit, *, operation):
    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None

        qc.start_sub_circuit("AB")
        success_ab = False
        while not success_ab:
            qc.create_bell_pair(13, 9)
            success_ab = qc.single_selection(CZ_gate, 12, 8, retry=False)
            if not success_ab:
                continue
            success_ab2 = False
            while not success_ab2:
                qc.create_bell_pair(12, 8)
                success_ab2 = qc.single_selection(CNOT_gate, 11, 7, retry=False)
            success_ab = qc.single_selection_var(CY_gate, CminY_gate, 12, 8, create_bell_pair=False, retry=False)

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit("CD")
        success_cd = False
        while not success_cd:
            qc.create_bell_pair(6, 2)
            success_cd = qc.single_selection(CZ_gate, 5, 1, retry=False)
            if not success_cd:
                continue
            success_cd2 = False
            while not success_cd2:
                qc.create_bell_pair(5, 1)
                success_cd2 = qc.single_selection(CZ_gate, 4, 0, retry=False)
            success_cd = qc.single_selection_var(CNOT_gate, CNOT_gate, 5, 1, create_bell_pair=False, retry=False)

        PBAR.update(30) if PBAR is not None else None

        qc.start_sub_circuit("AC")
        success_ac = False
        while not success_ac:
            success_ac2 = False
            qc.create_bell_pair(12, 5)
            while not success_ac2:
                qc.create_bell_pair(11, 4)
                success_ac2 = qc.single_selection(CNOT_gate, 10, 3, retry=False)
            success_ac = qc.single_selection_var(CY_gate, CminY_gate, 11, 4, create_bell_pair=False, retry=False)
            if not success_ac:
                continue
            success_ac = qc.single_selection(CZ_gate, 11, 4, retry=False)

        qc.start_sub_circuit("BD")
        success_bd = False
        while not success_bd:
            qc.create_bell_pair(8, 1)
            success_bd = qc.single_selection(CZ_gate, 7, 0, retry=False)

        qc.start_sub_circuit("AC", forced_level=True)
        qc.apply_gate(CNOT_gate, cqubit=13, tqubit=12, reverse=True)    # 5, 12, 9, 13
        # qc.start_sub_circuit("C")
        qc.apply_gate(CNOT_gate, cqubit=6, tqubit=5, reverse=True)      # 5, 12, 9, 13, 2, 6
        # qc.start_sub_circuit("AC")
        # qc._thread_safe_printing = False
        # qc.draw_circuit()
        measurement_outcomes = qc.measure([5, 12], basis="Z")           # 9, 13, 2, 6
        success = measurement_outcomes[0] == measurement_outcomes[1]
        qc.start_sub_circuit("AB")
        if not success:
            qc.X(13)
            qc.X(9)
        qc.start_sub_circuit("BD")
        qc.apply_gate(CZ_gate, cqubit=9, tqubit=8, reverse=True)        # 1, 8, 9, 13, 2, 6
        # qc.start_sub_circuit("D")
        qc.apply_gate(CZ_gate, cqubit=2, tqubit=1, reverse=True)        # 1, 8, 9, 13, 2, 6
        # qc.start_sub_circuit("BD")
        measurement_outcomes2 = qc.measure([1, 8])
        ghz_success = measurement_outcomes2[0] == measurement_outcomes2[1]
        PBAR.update(30) if PBAR is not None else None

    qc.get_state_fidelity()

    qc.start_sub_circuit("B")
    qc.apply_gate(operation, cqubit=9, tqubit=18)
    qc.measure(9, probabilistic=False)

    qc.start_sub_circuit("A")
    qc.apply_gate(operation, cqubit=13, tqubit=20)
    qc.measure(13, probabilistic=False)

    qc.start_sub_circuit("D")
    qc.apply_gate(operation, cqubit=2, tqubit=14)
    qc.measure(2, probabilistic=False)

    qc.start_sub_circuit("C")
    qc.apply_gate(operation, cqubit=6, tqubit=16)
    qc.measure(6, probabilistic=False)

    qc.end_current_sub_circuit(total=True)

    PBAR.update(10) if PBAR is not None else None

    # qc.append_print_lines("\nGHZ fidelity: {}\n".format(qc.ghz_fidelity))


def stringent(qc, *, operation):
    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None

        # Step 1-8 from Table D.2 (Thesis Naomi Nickerson)
        success_ab = False
        qc.start_sub_circuit("AB")
        while not success_ab:
            qc.create_bell_pair(11, 8)
            success_ab = qc.double_selection(CZ_gate, 10, 7, retry=False)
            if not success_ab:
                continue
            success_ab = qc.double_selection(CNOT_gate, 10, 7, retry=False)
            if not success_ab:
                continue

            success_ab = qc.double_dot(CZ_gate, 10, 7, retry=False)
            if not success_ab:
                continue
            success_ab = qc.double_dot(CNOT_gate, 10, 7, retry=False)
            if not success_ab:
                continue

        PBAR.update(20) if PBAR is not None else None

        # Step 1-8 from Table D.2 (Thesis Naomi Nickerson)
        success_cd = False
        qc.start_sub_circuit("CD")
        while not success_cd:
            qc.create_bell_pair(5, 2)
            success_cd = qc.double_selection(CZ_gate, 4, 1, retry=False)
            if not success_cd:
                continue
            success_cd = qc.double_selection(CNOT_gate, 4, 1, retry=False)
            if not success_cd:
                continue

            success_cd = qc.double_dot(CZ_gate, 4, 1, retry=False)
            if not success_cd:
                continue
            success_cd = qc.double_dot(CNOT_gate, 4, 1, retry=False)
            if not success_cd:
                continue

        PBAR.update(20) if PBAR is not None else None

        # Step 9-11 from Table D.2 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("AC")
        # Return success (even parity measurement). If False (uneven), X-gate must be drawn at second single dot
        success, single_selection_success = qc.double_dot(CZ_gate, 10, 4, parity_check=False)
        qc.start_sub_circuit("BD")
        ghz_success = qc.double_dot(CZ_gate, 7, 1, draw_X_gate=not success, retry=False)
        if not ghz_success or not single_selection_success:
            ghz_success = False
            continue

        PBAR.update(20) if PBAR is not None else None

        # Step 12-14 from Table D.2 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("AC", forced_level=True)
        ghz_success_1 = qc.double_dot(CZ_gate, 10, 4, retry=False)
        qc.start_sub_circuit("BD")
        ghz_success_2 = qc.double_dot(CZ_gate, 7, 1, retry=False)
        if any([not ghz_success_1, not ghz_success_2]):
            ghz_success = False
            continue

        PBAR.update(20) if PBAR is not None else None

    # Step 15 from Table D.2 (Thesis Naomi Nickerson)
    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    qc.start_sub_circuit("B")
    qc.apply_gate(operation, cqubit=8, tqubit=16)
    qc.measure(8, probabilistic=False)

    qc.start_sub_circuit("A")
    qc.apply_gate(operation, cqubit=11, tqubit=18)
    qc.measure(11, probabilistic=False)

    qc.start_sub_circuit("D")
    qc.apply_gate(operation, cqubit=2, tqubit=12)
    qc.measure(2, probabilistic=False)

    qc.start_sub_circuit("C")
    qc.apply_gate(operation, cqubit=5, tqubit=14)
    qc.measure(5, probabilistic=False)

    qc.end_current_sub_circuit(total=True)

    PBAR.update(10) if PBAR is not None else None


def expedient_swap(qc, *, operation):
    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None

        qc.start_sub_circuit("AB")
        success_ab = False
        while not success_ab:
            qc.create_bell_pair(9, 6)
            qc.SWAP(9, 10, efficient=True)
            qc.SWAP(6, 7, efficient=True)
            success_ab = qc.double_selection_swap(CZ_gate, 9, 6)
            if not success_ab:
                continue
            success_ab = qc.double_selection_swap(CNOT_gate, 9, 6)

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit("CD")
        success_cd = False
        while not success_cd:
            qc.create_bell_pair(3, 0)
            qc.SWAP(3, 4, efficient=True)
            qc.SWAP(0, 1, efficient=True)
            success_cd = qc.double_selection_swap(CZ_gate, 3, 0)
            if not success_cd:
                continue
            success_cd = qc.double_selection_swap(CNOT_gate, 3, 0)

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit('AC')
        success_1 = qc.single_dot_swap(CZ_gate, 9, 3, parity_check=False)
        qc.start_sub_circuit('BD')
        ghz_success = qc.single_dot_swap(CZ_gate, 6, 0, draw_X_gate=not success_1, retry=False)
        if not ghz_success:
            continue

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit('AC', forced_level=True)
        ghz_success_1 = qc.single_dot_swap(CZ_gate, 9, 3, retry=False)
        qc.start_sub_circuit("BD")
        ghz_success_2 = qc.single_dot_swap(CZ_gate, 6, 0, retry=False)
        if any([not ghz_success_1, not ghz_success_2]):
            ghz_success = False
            continue

    PBAR.update(20) if PBAR is not None else None

    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    qc.start_sub_circuit("B")
    qc.SWAP(6, 7, efficient=True)
    qc.apply_gate(operation, cqubit=6, tqubit=16)
    qc.measure(6, probabilistic=False)

    qc.start_sub_circuit("A")
    qc.SWAP(9, 10, efficient=True)
    qc.apply_gate(operation, cqubit=9, tqubit=18)
    qc.measure(9, probabilistic=False)

    qc.start_sub_circuit("D")
    qc.SWAP(0, 1, efficient=True)
    qc.apply_gate(operation, cqubit=0, tqubit=12)
    qc.measure(0, probabilistic=False)

    qc.start_sub_circuit("C")
    qc.SWAP(3, 4, efficient=True)
    qc.apply_gate(operation, cqubit=3, tqubit=14)
    qc.measure(3, probabilistic=False)

    qc.end_current_sub_circuit(total=True)

    PBAR.update(10) if PBAR is not None else None


def stringent_swap(qc, *, operation):
    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None

        qc.start_sub_circuit("AB")
        success_ab = False
        while not success_ab:
            qc.create_bell_pair(9, 6)
            qc.SWAP(9, 10, efficient=True)
            qc.SWAP(6, 7, efficient=True)
            if not qc.double_selection_swap(CZ_gate, 9, 6):
                continue
            if not qc.double_selection_swap(CNOT_gate, 9, 6):
                continue
            if not qc.double_dot_swap(CZ_gate, 9, 6):
                continue
            success_ab = qc.double_dot_swap(CNOT_gate, 9, 6)

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit("CD")
        success_cd = False
        while not success_cd:
            qc.create_bell_pair(3, 0)
            qc.SWAP(3, 4, efficient=True)
            qc.SWAP(0, 1, efficient=True)
            if not qc.double_selection_swap(CZ_gate, 3, 0):
                continue
            if not qc.double_selection_swap(CNOT_gate, 3, 0):
                continue
            if not qc.double_dot_swap(CZ_gate, 3, 0):
                continue
            success_cd = qc.double_dot_swap(CNOT_gate, 3, 0)

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit("AC")
        success, single_selection_success = qc.double_dot_swap(CZ_gate, 9, 3, parity_check=False)
        qc.start_sub_circuit("BD")
        ghz_success = qc.double_dot_swap(CZ_gate, 6, 0, draw_X_gate=not success, retry=False)
        if not ghz_success or not single_selection_success:
            ghz_success = False
            continue

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit("AC", forced_level=True)
        ghz_success_1 = qc.double_dot_swap(CZ_gate, 9, 3, retry=False)
        qc.start_sub_circuit("BD")
        ghz_success_2 = qc.double_dot_swap(CZ_gate, 6, 0, retry=False)
        if any([not ghz_success_1, not ghz_success_2]):
            ghz_success = False
            continue

        PBAR.update(20) if PBAR is not None else None

    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    qc.start_sub_circuit("B")
    qc.SWAP(6, 7, efficient=True)
    qc.apply_gate(operation, cqubit=6, tqubit=16)
    qc.measure(6, probabilistic=False)

    qc.start_sub_circuit("A")
    qc.SWAP(9, 10, efficient=True)
    qc.apply_gate(operation, cqubit=9, tqubit=18)
    qc.measure(9, probabilistic=False)

    qc.start_sub_circuit("D")
    qc.SWAP(0, 1, efficient=True)
    qc.apply_gate(operation, cqubit=0, tqubit=12)
    qc.measure(0, probabilistic=False)

    qc.start_sub_circuit("C")
    qc.SWAP(3, 4, efficient=True)
    qc.apply_gate(operation, cqubit=3, tqubit=14)
    qc.measure(3, probabilistic=False)

    qc.end_current_sub_circuit(total=True)

    PBAR.update(10) if PBAR is not None else None


def duo_structure(qc: QuantumCircuit, *, operation):
    qc.start_sub_circuit("AB")
    qc.create_bell_pair(2, 5)
    qc.double_selection(CZ_gate, 1, 4)
    qc.double_selection(CNOT_gate, 1, 4)
    qc.apply_gate(operation, cqubit=2, tqubit=6)
    qc.apply_gate(operation, cqubit=2, tqubit=8)
    qc.apply_gate(operation, cqubit=5, tqubit=10)
    qc.apply_gate(operation, cqubit=5, tqubit=12)
    qc.measure([5, 2], probabilistic=False)
    qc.end_current_sub_circuit(total=True)

    PBAR.update(50) if PBAR is not None else None


# noinspection PyUnresolvedReferences
def duo_structure_2(qc: QuantumCircuit, *, operation):
    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None

        # Step 1-2 from Table D.1 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("AB")
        success_ab = False
        while not success_ab:
            qc.create_bell_pair(15, 11)
            success_ab = qc.double_selection(CZ_gate, 14, 10, retry=False)
            if not success_ab:
                continue
            success_ab = qc.double_selection(CNOT_gate, 14, 10, retry=False)

        PBAR.update(20) if PBAR is not None else None

        # Step 1-2 from Table D.1 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("CD")
        success_cd = False
        while not success_cd:
            qc.create_bell_pair(7, 3)
            success_cd = qc.double_selection(CZ_gate, 6, 2, retry=False)
            if not success_cd:
                continue
            success_cd = qc.double_selection(CNOT_gate, 6, 2, retry=False)

        PBAR.update(20) if PBAR is not None else None

        # Step 3-5 from Table D.1 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("AC")
        # Return success (even parity measurement). If False (uneven), X-gate must be drawn at second single dot
        success_1 = qc.single_dot(CZ_gate, 14, 6, parity_check=False)
        qc.start_sub_circuit("BD")
        ghz_success = qc.single_dot(CZ_gate, 10, 2, draw_X_gate=not success_1, retry=False)
        if not ghz_success:
            continue

        PBAR.update(20) if PBAR is not None else None

        # Step 6-8 from Table D.1 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("AC", forced_level=True)
        ghz_success_1 = qc.single_dot(CZ_gate, 14, 6, retry=False)
        qc.start_sub_circuit("BD")
        ghz_success_2 = qc.single_dot(CZ_gate, 10, 2, retry=False)
        if any([not ghz_success_1, not ghz_success_2]):
            ghz_success = False
            continue

        PBAR.update(20) if PBAR is not None else None

    # Step 9 from Table D.1 (Thesis Naomi Nickerson)
    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    qc.start_sub_circuit("B")
    qc.apply_gate(operation, cqubit=11, tqubit=26)
    qc.measure(11, probabilistic=False)

    qc.start_sub_circuit("A")
    qc.apply_gate(operation, cqubit=15, tqubit=30)
    qc.measure(15, probabilistic=False)

    qc.start_sub_circuit("D")
    qc.apply_gate(operation, cqubit=3, tqubit=18)
    qc.measure(3, probabilistic=False)

    qc.start_sub_circuit("C")
    qc.apply_gate(operation, cqubit=7, tqubit=22)
    qc.measure(7, probabilistic=False)

    qc.end_current_sub_circuit(total=True)

    PBAR.update(10) if PBAR is not None else None

    return [[30, 26, 22, 18], [28, 24, 20, 16]]
