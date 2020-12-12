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
        qc.define_node("A", qubits=[1, 3, 5, 7, 0], data_qubits=[1, 3, 5, 7], ghz_qubit=0)
        qc.define_sub_circuit("A")

    elif protocol == 'duo_structure':
        qc = QuantumCircuit(14, 2, **kwargs)

        qc.define_node("A", qubits=[0, 1, 2, 6, 8], data_qubits=[6, 8], ghz_qubit=2)
        qc.define_node("B", qubits=[3, 4, 5, 10, 12], data_qubits=[10, 12], ghz_qubit=5)

        qc.define_sub_circuit("AB")
        qc.define_sub_circuit("A")
        qc.define_sub_circuit("B", concurrent_sub_circuits="A")

    elif protocol == 'plain':
        qc = QuantumCircuit(16, 2, **kwargs)

        qc.define_node("A", qubits=[14, 7, 6], data_qubits=14, ghz_qubit=7)
        qc.define_node("B", qubits=[12, 5, 4], data_qubits=12, ghz_qubit=5)
        qc.define_node("C", qubits=[10, 3, 2], data_qubits=10, ghz_qubit=3)
        qc.define_node("D", qubits=[8, 0, 1], data_qubits=8, ghz_qubit=1)

    elif protocol == 'plain_swap':
        qc = QuantumCircuit(16, 2, **kwargs)

        qc.define_node("A", qubits=[14, 7, 6], data_qubits=14, electron_qubits=6, ghz_qubit=7)
        qc.define_node("B", qubits=[12, 5, 4], data_qubits=12, electron_qubits=4, ghz_qubit=5)
        qc.define_node("C", qubits=[10, 3, 2], data_qubits=10, electron_qubits=2, ghz_qubit=3)
        qc.define_node("D", qubits=[8, 0, 1], data_qubits=8, electron_qubits=0, ghz_qubit=1)

    elif protocol == 'duo_structure_2':
        qc = QuantumCircuit(32, 5, **kwargs)

        qc.define_node("A", qubits=[30, 28, 15, 14, 13, 12], data_qubits=[30, 28], ghz_qubit=15)
        qc.define_node("B", qubits=[26, 24, 11, 10, 9, 8], data_qubits=[26, 24], ghz_qubit=11)
        qc.define_node("C", qubits=[22, 20, 7, 6, 5, 4], data_qubits=[22, 10], ghz_qubit=7)
        qc.define_node("D", qubits=[18, 16, 3, 2, 1, 0], data_qubits=[18, 16], ghz_qubit=3)

    elif protocol in ['expedient_swap', 'stringent_swap']:
        qc = QuantumCircuit(20, 2, **kwargs)

        qc.define_node("A", qubits=[18, 11, 10, 9], electron_qubits=9, data_qubits=18, ghz_qubit=10)
        qc.define_node("B", qubits=[16, 8, 7, 6], electron_qubits=6, data_qubits=16, ghz_qubit=7)
        qc.define_node("C", qubits=[14, 5, 4, 3], electron_qubits=3, data_qubits=14, ghz_qubit=4)
        qc.define_node("D", qubits=[12, 2, 1, 0], electron_qubits=0, data_qubits=12, ghz_qubit=1)

    else:
        qc = QuantumCircuit(20, 2, **kwargs)

        qc.define_node("A", qubits=[18, 11, 10, 9], data_qubits=18, ghz_qubit=11)
        qc.define_node("B", qubits=[16, 8, 7, 6], data_qubits=16, ghz_qubit=8)
        qc.define_node("C", qubits=[14, 5, 4, 3], data_qubits=14, ghz_qubit=5)
        qc.define_node("D", qubits=[12, 2, 1, 0], data_qubits=12, ghz_qubit=2)

    # Common sub circuit defining handled here
    if protocol in ['plain', 'plain_swap', 'duo_structure_2', 'expedient', 'expedient', 'expedient_swap', 'stringent',
                    'stringent_swap']:
        qc.define_sub_circuit("AB")
        qc.define_sub_circuit("CD", concurrent_sub_circuits="AB")
        qc.define_sub_circuit("AC")
        if 'plain' not in protocol:
            qc.define_sub_circuit("BD", concurrent_sub_circuits="AC")
        qc.define_sub_circuit("A")
        qc.define_sub_circuit("B")
        qc.define_sub_circuit("C")
        qc.define_sub_circuit("D", concurrent_sub_circuits=["A", "B", "C"])

    return qc


def monolithic(qc: QuantumCircuit, *, operation):
    qc.set_qubit_states({0: ket_p})
    qc.stabilizer_measurement(operation, nodes=["A"])

    PBAR.update(90) if PBAR is not None else None


def plain(qc: QuantumCircuit, *, operation):
    qc.start_sub_circuit("AB")
    qc.create_bell_pair(7, 5)
    qc.start_sub_circuit("CD")
    qc.create_bell_pair(3, 1)
    qc.start_sub_circuit("AC")
    success = qc.single_selection(operation, 6, 2, retry=False)
    if not success:
        qc.start_sub_circuit("A")
        qc.X(7)
        qc.start_sub_circuit("B")
        qc.X(5)

    qc.get_state_fidelity()

    qc.stabilizer_measurement(operation, nodes=["B", "A", "D", "C"])

    PBAR.update(90) if PBAR is not None else None


def plain_swap(qc: QuantumCircuit, *, operation):
    qc.start_sub_circuit("AB")
    qc.create_bell_pair(6, 4)
    qc.SWAP(6, 7)
    qc.SWAP(4, 5)
    qc.start_sub_circuit("CD")
    qc.create_bell_pair(2, 0)
    qc.SWAP(2, 3)
    qc.SWAP(0, 1)
    qc.start_sub_circuit("AC")
    success = qc.single_selection_swap(operation, 6, 2)
    if not success:
        qc.start_sub_circuit("A")
        qc.X(7)
        qc.start_sub_circuit("B")
        qc.X(5)

    qc.get_state_fidelity()

    qc.stabilizer_measurement(operation, nodes=["B", "A", "D", "C"], swap=True)

    PBAR.update(90) if PBAR is not None else None


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
    qc.stabilizer_measurement(operation, nodes=["B", "A", "D", "C"])

    PBAR.update(10) if PBAR is not None else None


def stringent(qc: QuantumCircuit, *, operation):
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
    qc.get_state_fidelity()

    # Step 15 from Table D.2 (Thesis Naomi Nickerson)
    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    qc.stabilizer_measurement(operation, nodes=["B", "A", "D", "C"])

    PBAR.update(10) if PBAR is not None else None


def expedient_swap(qc: QuantumCircuit, *, operation):
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
    qc.get_state_fidelity()

    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    qc.stabilizer_measurement(operation, nodes=["B", "A", "D", "C"], swap=True)

    PBAR.update(10) if PBAR is not None else None


def stringent_swap(qc: QuantumCircuit, *, operation):
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
    qc.get_state_fidelity()

    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    qc.stabilizer_measurement(operation, nodes=["B", "A", "D", "C"], swap=True)

    PBAR.update(10) if PBAR is not None else None


def duo_structure(qc: QuantumCircuit, *, operation):
    qc.start_sub_circuit("AB")
    qc.create_bell_pair(2, 5)
    qc.double_selection(CZ_gate, 1, 4)
    qc.double_selection(CNOT_gate, 1, 4)
    qc.get_state_fidelity()

    qc.stabilizer_measurement(operation, nodes=["A", "B"])

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
    qc.get_state_fidelity()

    # Step 9 from Table D.1 (Thesis Naomi Nickerson)
    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    qc.stabilizer_measurement(operation, nodes=["B", "A", "D", "C"], tqubit=[26, 30, 18, 22])

    PBAR.update(10) if PBAR is not None else None

    return [[30, 26, 22, 18], [28, 24, 20, 16]]
