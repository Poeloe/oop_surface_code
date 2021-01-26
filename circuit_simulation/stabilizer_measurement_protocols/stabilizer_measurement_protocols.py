from circuit_simulation.circuit_simulator import *
from tqdm import tqdm
PBAR: tqdm = None


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
        qc.define_node("A", qubits=[1, 3, 5, 7, 0], amount_data_qubits=4)
        qc.define_sub_circuit("A")

    elif 'plain' in protocol:
        qc = QuantumCircuit(16, 2, **kwargs)

        qc.define_node("A", qubits=[14, 7, 6])
        qc.define_node("B", qubits=[12, 5, 4])
        qc.define_node("C", qubits=[10, 3, 2])
        qc.define_node("D", qubits=[8, 1, 0])

    elif protocol in ['weight_2_4_swap', 'weight_3_swap']:
        qc = QuantumCircuit(32, 16, **kwargs)

        qc.define_node("A", qubits=[30, 28, 15, 14, 13, 12], amount_data_qubits=2)
        qc.define_node("B", qubits=[26, 24, 11, 10, 9, 8], amount_data_qubits=2)
        qc.define_node("C", qubits=[22, 20, 7, 6, 5, 4], amount_data_qubits=2)
        qc.define_node("D", qubits=[18, 16, 3, 2, 1, 0], amount_data_qubits=2)

    elif protocol in ['dyn_prot_3_4_1_swap']:
        qc = QuantumCircuit(16, 2, **kwargs)

        qc.define_node("A", qubits=[12, 14, 7, 6], amount_data_qubits=2)
        qc.define_node("B", qubits=[10, 4, 5, 3])
        qc.define_node("C", qubits=[8, 1, 2, 0])

    elif protocol in ['dyn_prot_3_8_1_swap']:
        qc = QuantumCircuit(19, 2, **kwargs)

        qc.define_node("A", qubits=[15, 17, 10, 9, 8], amount_data_qubits=2)
        qc.define_node("B", qubits=[13, 7, 6, 5, 4])
        qc.define_node("C", qubits=[11, 3, 2, 1, 0])

    elif protocol in ['dyn_prot_4_4_1_swap']:
        qc = QuantumCircuit(16, 2, **kwargs)

        qc.define_node("A", qubits=[14, 7, 6])
        qc.define_node("B", qubits=[12, 5, 4])
        qc.define_node("C", qubits=[10, 3, 2])
        qc.define_node("D", qubits=[8, 1, 0])

    elif 'dyn_prot_4_6_sym_1' in protocol:
        qc = QuantumCircuit(18, 2, **kwargs)

        qc.define_node("A", qubits=[16, 9, 8])
        qc.define_node("B", qubits=[14, 7, 6, 5])
        qc.define_node("C", qubits=[12, 4, 3])
        qc.define_node("D", qubits=[10, 2, 1, 0])

    elif 'dyn_prot_4_14_1' in protocol:
        qc = QuantumCircuit(22, 2, **kwargs)

        qc.define_node("A", qubits=[20, 13, 12, 11, 10])
        qc.define_node("B", qubits=[18, 9, 8, 7])
        qc.define_node("C", qubits=[16, 6, 5, 4, 3])
        qc.define_node("D", qubits=[14, 2, 1, 0])

    elif protocol == 'dyn_prot_4_22_1':
        qc = QuantumCircuit(24, 2, **kwargs)

        qc.define_node("A", qubits=[22, 15, 14, 13, 12])
        qc.define_node("B", qubits=[20, 11, 10, 9, 8])
        qc.define_node("C", qubits=[18, 7, 6, 5, 4])
        qc.define_node("D", qubits=[16, 3, 2, 1, 0])

    elif protocol == 'dyn_prot_4_42_1':
        qc = QuantumCircuit(28, 2, **kwargs)

        qc.define_node("A", qubits=[26, 19, 18, 17, 16, 15])
        qc.define_node("B", qubits=[24, 14, 13, 12, 11, 10])
        qc.define_node("C", qubits=[22, 9, 8, 7, 6, 5])
        qc.define_node("D", qubits=[20, 4, 3, 2, 1, 0])

    elif protocol in ['dejmps_2_4_1_swap', 'dejmps_2_6_1_swap', 'dejmps_2_8_1_swap', 'bipartite_4_swap',
                      'bipartite_6_swap']:
        qc = QuantumCircuit(28, 4, **kwargs)

        # If you don't specify which qubits are the data-qubits and electron-qubits, it is assumed that the first
        # qubit(s) in the list is (are) the data-qubit(s) and the last one is the electron_qubit.

        qc.define_node("A", qubits=[26, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 12], amount_data_qubits=2)
        qc.define_node("B", qubits=[24, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 0], amount_data_qubits=2)

        qc.define_sub_circuit("AB")

        qc.define_sub_circuit("A")
        qc.define_sub_circuit("B", concurrent_sub_circuits=["A"])

    else:
        qc = QuantumCircuit(20, 2, **kwargs)

        qc.define_node("A", qubits=[18, 11, 10, 9])
        qc.define_node("B", qubits=[16, 8, 7, 6])
        qc.define_node("C", qubits=[14, 5, 4, 3])
        qc.define_node("D", qubits=[12, 2, 1, 0])

    # Common sub circuit defining handled here
    if protocol in ['plain', 'plain_swap', 'weight_2_4_swap', 'expedient', 'expedient_swap', 'stringent',
                    'stringent_swap', 'dyn_prot_4_6_sym_1', 'dyn_prot_4_6_sym_1_swap', 'dyn_prot_4_14_1',
                    'dyn_prot_4_4_1_swap', 'dyn_prot_4_14_1_swap', 'dyn_prot_4_22_1', 'dyn_prot_4_42_1']:
        qc.define_sub_circuit("AB")
        qc.define_sub_circuit("CD", concurrent_sub_circuits="AB")
        qc.define_sub_circuit("AC")
        if 'plain' not in protocol:
            qc.define_sub_circuit("BD", concurrent_sub_circuits="AC")
        qc.define_sub_circuit("A")
        qc.define_sub_circuit("B")
        qc.define_sub_circuit("C")
        qc.define_sub_circuit("D", concurrent_sub_circuits=["A", "B", "C"])

    elif protocol in ['dyn_prot_3_8_1_swap', 'dyn_prot_3_4_1_swap', 'weight_3_swap']:
        qc.define_sub_circuit("ABC")

        qc.define_sub_circuit("A")
        qc.define_sub_circuit("B")
        qc.define_sub_circuit("C", concurrent_sub_circuits=["A", "B"])

        qc.define_sub_circuit("AB")
        qc.define_sub_circuit("BC")
        qc.define_sub_circuit("AC")
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
    success = qc.single_selection(operation, 6, 2)
    if not success:
        qc.start_sub_circuit("A")
        qc.X(7)
        qc.start_sub_circuit("B")
        qc.X("B-e")

    qc.stabilizer_measurement(operation, nodes=["B", "A", "D", "C"])

    PBAR.update(90) if PBAR is not None else None


def plain_swap(qc: QuantumCircuit, *, operation):
    qc.start_sub_circuit("AB")
    qc.create_bell_pair("B-e", "A-e")
    qc.SWAP("A-e", "A-e+1")
    qc.start_sub_circuit("CD")
    qc.create_bell_pair("D-e", "C-e")
    qc.SWAP("C-e", "C-e+1")
    qc.start_sub_circuit("AC")
    success = qc.single_selection(operation, "C-e", "A-e", swap=True)
    if not success:
        qc.start_sub_circuit("A")
        qc.X("A-e+1")
        qc.start_sub_circuit("B")
        qc.X("B-e")

    qc.stabilizer_measurement(operation, nodes=["C", "D", "A", "B"], swap=True)

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
            success_ab = qc.double_selection(CZ_gate, 10, 7)
            if not success_ab:
                continue
            success_ab = qc.double_selection(CNOT_gate, 10, 7)

        PBAR.update(20) if PBAR is not None else None

        # Step 1-2 from Table D.1 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("CD")
        success_cd = False
        while not success_cd:
            qc.create_bell_pair(5, 2)
            success_cd = qc.double_selection(CZ_gate, 4, 1)
            if not success_cd:
                continue
            success_cd = qc.double_selection(CNOT_gate, 4, 1)

        PBAR.update(20) if PBAR is not None else None

        # Step 3-5 from Table D.1 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("AC")
        # Return success (even parity measurement). If False (uneven), X-gate must be drawn at second single dot
        success_1 = qc.single_dot(CZ_gate, 10, 4, parity_check=False)
        qc.start_sub_circuit("BD")
        ghz_success = qc.single_dot(CZ_gate, 7, 1, draw_X_gate=not success_1)
        if not ghz_success:
            continue

        PBAR.update(20) if PBAR is not None else None

        # Step 6-8 from Table D.1 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("AC", forced_level=True)
        ghz_success_1 = qc.single_dot(CZ_gate, 10, 4)
        qc.start_sub_circuit("BD")
        ghz_success_2 = qc.single_dot(CZ_gate, 7, 1)
        if any([not ghz_success_1, not ghz_success_2]):
            ghz_success = False
            continue

        PBAR.update(20) if PBAR is not None else None

    # Step 9 from Table D.1 (Thesis Naomi Nickerson)
    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    qc.stabilizer_measurement(operation, nodes=["B", "A", "D", "C"])

    PBAR.update(10) if PBAR is not None else None


def dejmps_2_4_1_swap(qc: QuantumCircuit, *, operation):

    level_1 = False
    while not level_1:
        PBAR.reset() if PBAR is not None else None
        qc.start_sub_circuit("AB", forced_level=True)
        qc.create_bell_pair("B-e", "A-e")
        qc.SWAP("B-e", "B-e+1", efficient=True)
        qc.SWAP("A-e", "A-e+1", efficient=True)
        success_level_1 = qc.single_selection(CZ_gate, "B-e", "A-e")
        if not success_level_1:
            continue

        PBAR.update(35) if PBAR is not None else None

        level_2 = False
        while not level_2:
            qc.create_bell_pair("B-e", "A-e")
            qc.SWAP("B-e", "B-e+2", efficient=True)
            qc.SWAP("A-e", "A-e+2", efficient=True)
            level_2 = qc.single_selection(CZ_gate, "B-e", "A-e", "B-e+2", "A-e+2")

        PBAR.update(35) if PBAR is not None else None

        qc.SWAP("B-e+2", "B-e", efficient=True)
        qc.SWAP("A-e+2", "A-e", efficient=True)
        level_1 = qc.single_selection(CNOT_gate, "B-e", "A-e", "B-e+1", "A-e+1", create_bell_pair=False)

        PBAR.update(20) if PBAR is not None else None

    qc.stabilizer_measurement(operation, nodes=["A", "B"])

    PBAR.update(10) if PBAR is not None else None


def dejmps_2_6_1_swap(qc: QuantumCircuit, *, operation):

    level_1 = False
    while not level_1:
        PBAR.reset() if PBAR is not None else None
        qc.start_sub_circuit("AB", forced_level=True)
        qc.create_bell_pair("B-e", "A-e")
        qc.SWAP("B-e", "B-e+1", efficient=True)
        qc.SWAP("A-e", "A-e+1", efficient=True)
        int_level_1 = qc.single_selection(CZ_gate, "B-e", "A-e")
        if not int_level_1:
            continue

        PBAR.update(20) if PBAR is not None else None

        level_2 = False
        while not level_2:
            qc.create_bell_pair("B-e", "A-e")
            qc.SWAP("B-e", "B-e+2", efficient=True)
            qc.SWAP("A-e", "A-e+2", efficient=True)
            level_2 = qc.single_selection(CZ_gate, "B-e", "A-e", "B-e+2", "A-e+2")

        PBAR.update(20) if PBAR is not None else None

        qc.SWAP("B-e+2", "B-e", efficient=True)
        qc.SWAP("A-e+2", "A-e", efficient=True)
        int_level_3 = qc.single_selection(CNOT_gate, "B-e", "A-e", "B-e+1", "A-e+1", create_bell_pair=False)
        if not int_level_3:
            continue

        PBAR.update(20) if PBAR is not None else None

        level_4 = False
        while not level_4:
            qc.create_bell_pair("B-e", "A-e")
            qc.SWAP("B-e", "B-e+2", efficient=True)
            qc.SWAP("A-e", "A-e+2", efficient=True)
            level_4 = qc.single_selection(CZ_gate, "B-e", "A-e", "B-e+2", "A-e+2")

        qc.SWAP("B-e+2", "B-e", efficient=True)
        qc.SWAP("A-e+2", "A-e", efficient=True)
        level_1 = qc.single_selection(CiY_gate, "B-e", "A-e", "B-e+1", "A-e+1", create_bell_pair=False,)

        PBAR.update(20) if PBAR is not None else None

    qc.stabilizer_measurement(operation, nodes=["A", "B"])

    PBAR.update(20) if PBAR is not None else None


def dejmps_2_8_1_swap(qc: QuantumCircuit, *, operation):

    level_1 = False
    while not level_1:
        PBAR.reset() if PBAR is not None else None
        qc.start_sub_circuit("AB", forced_level=True)
        qc.create_bell_pair("B-e", "A-e")
        qc.SWAP("B-e", "B-e+1", efficient=True)
        qc.SWAP("A-e", "A-e+1", efficient=True)
        int_level_1 = qc.single_selection(CZ_gate, "B-e", "A-e")
        if not int_level_1:
            continue

        PBAR.update(15) if PBAR is not None else None

        level_2 = False
        while not level_2:
            qc.create_bell_pair("B-e", "A-e")
            qc.SWAP("B-e", "B-e+2", efficient=True)
            qc.SWAP("A-e", "A-e+2", efficient=True)
            level_2 = qc.single_selection(CZ_gate, "B-e", "A-e", "B-e+2", "A-e+2")

        PBAR.update(15) if PBAR is not None else None

        qc.SWAP("B-e+2", "B-e", efficient=True)
        qc.SWAP("A-e+2", "A-e", efficient=True)
        int_level_3 = qc.single_selection(CNOT_gate, "B-e", "A-e", "B-e+1", "A-e+1", create_bell_pair=False)
        if not int_level_3:
            continue

        PBAR.update(15) if PBAR is not None else None

        level_4 = False
        while not level_4:
            qc.create_bell_pair("B-e", "A-e")
            qc.SWAP("B-e", "B-e+2", efficient=True)
            qc.SWAP("A-e", "A-e+2", efficient=True)
            int_level_4 = qc.single_selection(CZ_gate, "B-e", "A-e", "B-e+2", "A-e+2")
            if not int_level_4:
                continue

            PBAR.update(15) if PBAR is not None else None

            level_5 = False
            while not level_5:
                qc.create_bell_pair("B-e", "A-e")
                qc.SWAP("B-e", "B-e+3", efficient=True)
                qc.SWAP("A-e", "A-e+3", efficient=True)
                level_5 = qc.single_selection(CZ_gate, "B-e", "A-e", "B-e+3", "A-e+3")

            PBAR.update(15) if PBAR is not None else None

            qc.SWAP("B-e+3", "B-e", efficient=True)
            qc.SWAP("A-e+3", "A-e", efficient=True)
            level_4 = qc.single_selection(CNOT_gate, "B-e", "A-e", "B-e+2", "A-e+2", create_bell_pair=False)

        qc.SWAP("B-e+2", "B-e", efficient=True)
        qc.SWAP("A-e+2", "A-e", efficient=True)
        level_1 = qc.single_selection(CiY_gate, "B-e", "A-e", "B-e+1", "A-e+1", create_bell_pair=False)

        PBAR.update(15) if PBAR is not None else None

    qc.stabilizer_measurement(operation, nodes=["A", "B"])

    PBAR.update(10) if PBAR is not None else None


def bipartite_4_swap(qc: QuantumCircuit, *, operation):
    level_1 = False
    while not level_1:
        PBAR.reset() if PBAR is not None else None
        qc.start_sub_circuit("AB", forced_level=True)
        qc.create_bell_pair("B-e", "A-e")       # [12, 0]
        qc.SWAP("B-e", "B-e+1", efficient=True)
        qc.SWAP("A-e", "A-e+1", efficient=True)   # [13, 1]
        qc.create_bell_pair("B-e", "A-e")
        qc.SWAP("B-e", "B-e+2", efficient=True)
        qc.SWAP("A-e", "A-e+2", efficient=True)   # [14, 2]
        qc.single_selection(CNOT_gate, "B-e", "A-e", "B-e+1", "A-e+1", measure=False)   # [12, 0, 13, 1]
        int_level_1 = qc.single_selection(CNOT_gate, "B-e", "A-e", "B-e+2", "A-e+2", create_bell_pair=False)    # [13, 1, 14, 2]
        if not int_level_1:
            continue

        PBAR.update(30) if PBAR is not None else None

        int_level_2 = qc.single_selection(CZ_gate, "B-e", "A-e", "B-e+2", "A-e+2")
        if not int_level_2:
            continue

        PBAR.update(30) if PBAR is not None else None

        qc.SWAP("B-e+2", "B-e", efficient=True)
        qc.SWAP("A-e+2", "A-e", efficient=True)       # [13, 1, 12, 0]
        level_1 = qc.single_selection(CZ_gate, "B-e", "A-e", "B-e+1", "A-e+1", create_bell_pair=False)
        PBAR.update(30) if PBAR is not None else None

    qc.stabilizer_measurement(operation, nodes=["A", "B"])

    PBAR.update(10) if PBAR is not None else None


def bipartite_6_swap(qc: QuantumCircuit, *, operation):
    level_1 = False
    while not level_1:
        PBAR.reset() if PBAR is not None else None
        qc.start_sub_circuit("AB", forced_level=True)
        qc.create_bell_pair("B-e", "A-e")           # [12, 0]
        qc.SWAP("B-e", "B-e+1", efficient=True)
        qc.SWAP("A-e", "A-e+1", efficient=True)       # [13, 1]
        qc.create_bell_pair("B-e", "A-e")
        qc.SWAP("B-e", "B-e+2", efficient=True)
        qc.SWAP("A-e", "A-e+2", efficient=True)
        qc.single_selection(CNOT_gate, "B-e", "A-e", "B-e+1", "A-e+1", measure=False)   # [12, 0, 13, 1]
        qc.single_selection(CNOT_gate, "B-e", "A-e", "B-e+2", "A-e+2", create_bell_pair=False, measure=False, reverse_den_mat_add=True)   # [14, 2, 12, 0, 13, 1]
        qc.SWAP("B-e", "B-e+3", efficient=True)
        qc.SWAP("A-e", "A-e+3", efficient=True)   # [14, 2, 15, 3, 13, 1]
        qc.single_selection(CNOT_gate, "B-e", "A-e", "B-e+1", "A-e+1", measure=False, reverse_den_mat_add=True)   # [14, 2, 15, 3, 13, 1, 12, 0]
        qc.SWAP("B-e", "B-e+4", efficient=True)
        qc.SWAP("A-e", "A-e+4", efficient=True)   # [14, 2, 15, 3, 13, 1, 16, 4]
        int_level_1 = qc.single_selection(CZ_gate, "B-e", "A-e", "B-e+4", "A-e+4")
        if not int_level_1:
            continue

        PBAR.update(15) if PBAR is not None else None

        int_level_2 = qc.single_selection(CNOT_gate, "B-e", "A-e", "B-e+3", "A-e+3")
        if not int_level_2:
            continue

        PBAR.update(15) if PBAR is not None else None

        qc.SWAP("B-e+3", "B-e", efficient=True)
        qc.SWAP("A-e+3", "A-e", efficient=True)   # [14, 2, 12, 0, 13, 1, 16, 4]
        qc.single_selection(CZ_gate, "B-e", "A-e", "B-e+2", "A-e+2", create_bell_pair=False, measure=False)
        int_level_3 = qc.single_selection(CZ_gate, "B-e", "A-e", "B-e+1", "A-e+1", create_bell_pair=False)      # [14, 2, 13, 1, 16, 4]
        if not int_level_3:
            continue

        PBAR.update(15) if PBAR is not None else None

        qc.SWAP("B-e+2", "B-e", efficient=True)
        qc.SWAP("A-e+2", "A-e", efficient=True)       # [12, 0, 13, 1, 16, 4]
        int_level_4 = qc.single_selection(CZ_gate, "B-e", "A-e", "B-e+4", "A-e+4", create_bell_pair=False)      # [13, 1, 16, 4]
        if not int_level_4:
            continue

        PBAR.update(15) if PBAR is not None else None

        qc.SWAP("B-e+4", "B-e", efficient=True)
        qc.SWAP("A-e+4", "A-e", efficient=True)
        meas_outc = qc.measure(["A-e", "B-e"])
        level_1 = meas_outc[0] == meas_outc[1]

        PBAR.update(15) if PBAR is not None else None

    qc.stabilizer_measurement(operation, nodes=["A", "B"])

    PBAR.update(25) if PBAR is not None else None


def dyn_prot_3_4_1_swap(qc: QuantumCircuit, *, operation, tqubit=None):

    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None
        qc.start_sub_circuit("AB")
        qc.create_bell_pair("A-e", "B-e")       # 3, 6
        qc.SWAP("A-e", "A-e+2", efficient=True)
        qc.SWAP("B-e", "B-e+1", efficient=True)   # 4, 7

        PBAR.update(30) if PBAR is not None else None

        qc.start_sub_circuit("AC")
        qc.create_bell_pair("C-e", "A-e")       # 6, 0
        qc.SWAP("C-e", "C-e+1", efficient=True)   # 6, 1
        qc.apply_gate(CNOT_gate, cqubit="A-e+2", tqubit="A-e", electron_is_target=True, reverse=True)   # 6, 1, 4, 7
        measurement_outcome = qc.measure(["A-e"], basis="Z")    # 1, 4, 7
        # BEGIN FUSION CORRECTION:
        time_in_A = qc.nodes["A"].sub_circuit_time
        time_in_C = qc.nodes["C"].sub_circuit_time
        if time_in_C < time_in_A:
            qc._increase_duration(time_in_A - time_in_C, [], involved_nodes=["C"])
        if measurement_outcome[0] == 1:
            qc.X("C-e+1")
        # END FUSION CORRECTION

        PBAR.update(30) if PBAR is not None else None

        level_2 = False
        while not level_2:
            qc.start_sub_circuit("BC")
            qc.create_bell_pair("B-e", "C-e")  # 0, 3
            qc.SWAP("B-e", "B-e+2", efficient=True)
            qc.SWAP("C-e", "C-e+2", efficient=True)  # 2, 5
            level_2 = qc.single_selection(CiY_gate, "B-e", "C-e", "B-e+2", "C-e+2")
        qc.SWAP("B-e+2", "B-e", efficient=True)
        qc.SWAP("C-e+2", "C-e", efficient=True)
        ghz_success = qc.single_selection(CZ_gate, "B-e", "C-e", "B-e+1", "C-e+1", create_bell_pair=False)

        PBAR.update(30) if PBAR is not None else None

    qc.stabilizer_measurement(operation, nodes=["C", "B", "A"], tqubit=tqubit)

    PBAR.update(10) if PBAR is not None else None


def dyn_prot_3_8_1_swap(qc: QuantumCircuit, *, operation, tqubit=None):

    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None
        qc.start_sub_circuit("AB")
        qc.create_bell_pair("A-e", "B-e")       # 4, 8
        qc.SWAP("A-e", "A-e+2", efficient=True)
        qc.SWAP("B-e", "B-e+3", efficient=True)   # 5, 9
        int_level_1 = qc.single_selection(CNOT_gate, "A-e", "B-e", "A-e+2", "B-e+3")
        if not int_level_1:
            continue

        PBAR.update(30) if PBAR is not None else None

        qc.start_sub_circuit("AC")
        level_2 = False
        while not level_2:
            qc.create_bell_pair("C-e", "A-e")       # 8, 0
            qc.SWAP("A-e", "A-e+1", efficient=True)
            qc.SWAP("C-e", "C-e+3", efficient=True)   # 10, 1
            level_2 = qc.single_selection(CiY_gate, "A-e", "C-e", "A-e+1", "C-e+3")
        qc.SWAP("A-e+1", "A-e", efficient=True)       # 8, 1
        qc.apply_gate(CNOT_gate, cqubit="A-e+2", tqubit="A-e", electron_is_target=True, reverse=True)   # 8, 1, 5, 9
        measurement_outcome = qc.measure(["A-e"], basis="Z")    # 1, 5, 9
        # BEGIN FUSION CORRECTION:
        time_in_A = qc.nodes["A"].sub_circuit_time
        time_in_C = qc.nodes["C"].sub_circuit_time
        if time_in_C < time_in_A:
            qc._increase_duration(time_in_A - time_in_C, [], involved_nodes=["C"])
        if measurement_outcome[0] == 1:
            qc.X("C-e+3")
        # END FUSION CORRECTION

        PBAR.update(30) if PBAR is not None else None

        qc.start_sub_circuit("BC")
        level_3 = False
        while not level_3:
            qc.create_bell_pair("B-e", "C-e")  # 0, 4
            qc.SWAP("B-e", "B-e+2", efficient=True)
            qc.SWAP("C-e", "C-e+2", efficient=True)  # 2, 6
            int_level_3 = qc.single_selection(CNOT_gate, "B-e", "C-e", "B-e+2", "C-e+2")
            if not int_level_3:
                continue
            level_4 = False
            while not level_4:
                qc.create_bell_pair("B-e", "C-e")  # 0, 4
                qc.SWAP("B-e", "B-e+1", efficient=True)
                qc.SWAP("C-e", "C-e+1", efficient=True)  # 3, 7
                level_4 = qc.single_selection(CNOT_gate, "B-e", "C-e", "B-e+1", "C-e+1")
            qc.SWAP("B-e+1", "B-e", efficient=True)
            qc.SWAP("C-e+1", "C-e", efficient=True)
            level_3 = qc.single_selection(CiY_gate, "B-e", "C-e", "B-e+2", "C-e+2", create_bell_pair=False)

        qc.SWAP("B-e+2", "B-e", efficient=True)
        qc.SWAP("C-e+2", "C-e", efficient=True)
        ghz_success = qc.single_selection(CZ_gate, "B-e", "C-e", "B-e+3", "C-e+3", create_bell_pair=False)

        PBAR.update(30) if PBAR is not None else None

    qc.stabilizer_measurement(operation, nodes=["C", "B", "A"], tqubit=tqubit)

    PBAR.update(10) if PBAR is not None else None


def dyn_prot_4_4_1_swap(qc: QuantumCircuit, *, operation):

    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None
        qc.start_sub_circuit("AC")
        qc.create_bell_pair("A-e", "C-e")       # 2, 6;     2 is "C-e" and 6 is "A-e"
        qc.SWAP("A-e", "A-e+1", efficient=True)
        qc.SWAP("C-e", "C-e+1", efficient=True)   # 3, 7

        qc.start_sub_circuit("BD")
        qc.create_bell_pair("D-e", "B-e")       # 4, 0
        qc.SWAP("B-e", "B-e+1", efficient=True)
        qc.SWAP("D-e", "D-e+1", efficient=True)   # 5, 1

        PBAR.update(40) if PBAR is not None else None

        qc.start_sub_circuit("AB")
        qc.create_bell_pair("B-e", "A-e")       # 6, 4
        # qc.append_print_lines(qc.get_combined_density_matrix([3]))
        qc.apply_gate(CNOT_gate, cqubit="A-e", tqubit="A-e+1")   # 6, 4, 3, 7;        6 is "A-e" and 7 is "A-e+1"
        qc.apply_gate(CNOT_gate, cqubit="B-e", tqubit="B-e+1")   # 6, 4, 3, 7, 5, 1;  4 is "B-e" and 5 is "B-e+1"
        qc.SWAP("A-e", "A-e+1", efficient=False)
        qc.SWAP("B-e", "B-e+1", efficient=False)
        measurement_outcomes = qc.measure(["A-e", "B-e"], basis="Z")    # 3, 7, 5, 1
        # BEGIN FUSION CORRECTION:
        # The correction in nodes C and D can only be applied after both measurements in A and B
        time_after_meas = max(qc.nodes["A"].sub_circuit_time, qc.nodes["B"].sub_circuit_time)
        # END FUSION CORRECTION

        qc.start_sub_circuit("CD")
        qc.create_bell_pair("C-e", "D-e")
        # BEGIN FUSION CORRECTION
        time_in_C = qc.nodes["C"].sub_circuit_time
        time_in_D = qc.nodes["D"].sub_circuit_time
        if time_in_C < time_after_meas:
            qc._increase_duration(time_after_meas - time_in_C, [], involved_nodes=["C"])
        if time_in_D < time_after_meas:
            qc._increase_duration(time_after_meas - time_in_D, [], involved_nodes=["D"])
        if measurement_outcomes != SKIP() and measurement_outcomes[0] == 1:
            qc.X("C-e+1")
        if measurement_outcomes != SKIP() and measurement_outcomes[1] == 1:
            qc.X("D-e+1")
        # END FUSION CORRECTION
        ghz_success = qc.single_selection(CZ_gate, "C-e", "D-e", "C-e+1", "D-e+1", create_bell_pair=False)

        PBAR.update(40) if PBAR is not None else None

    qc.stabilizer_measurement(operation, nodes=["C", "A", "B", "D"], swap=True)

    PBAR.update(20) if PBAR is not None else None


def dyn_prot_4_6_sym_1(qc: QuantumCircuit, *, operation):

    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None
        qc.start_sub_circuit("AB")
        qc.create_bell_pair(9, 7)

        PBAR.update(25) if PBAR is not None else None
        qc.start_sub_circuit("CD")
        qc.create_bell_pair(2, 3)

        PBAR.update(25) if PBAR is not None else None



        qc.start_sub_circuit("AC", forced_level=True)
        qc.create_bell_pair(4, 8)
        qc.apply_gate(CNOT_gate, cqubit=9, tqubit=8, reverse=True)    # 8, 4, 7, 9
        qc.apply_gate(CNOT_gate, cqubit=4, tqubit=3, reverse=True)      # 3, 2, 8, 4, 7, 9
        measurement_outcomes = qc.measure([3, 8], basis="Z")           # 2, 4, 7, 9
        # BEGIN FUSION CORRECTION: The correction in node C can only be applied AFTER the measurement in A
        time_between_meas = qc.nodes["A"].sub_circuit_time - qc.nodes["C"].sub_circuit_time
        if time_between_meas > 0:
            qc._increase_duration(time_between_meas, [], involved_nodes=["C"])
        if measurement_outcomes[0] == 1:
            qc.X(4)
        # The correction in node D can only be applied after both measurements in A and C
        time_after_both_meas = max(qc.nodes["A"].sub_circuit_time, qc.nodes["C"].sub_circuit_time)
        # END FUSION CORRECTION
        qc.create_bell_pair(3, 8)
        qc.apply_gate(CZ_gate, cqubit=8, tqubit=9)        # 8, 3, 2, 4, 7, 9
        qc.apply_gate(CZ_gate, cqubit=3, tqubit=4)
        measurement_outcomes_1 = qc.measure([8, 3])       # 2, 4, 7, 9
        ghz_success_1 = measurement_outcomes_1[0] == measurement_outcomes_1[1]

        PBAR.update(25) if PBAR is not None else None

        qc.start_sub_circuit("BD")
        success_bd = False
        while not success_bd:
            qc.create_bell_pair(1, 6)
            qc.create_bell_pair(5, 0)
            success_bd = qc.single_selection(CiY_gate, 5, 0, 6, 1, create_bell_pair=False)
        # BEGIN FUSION CORRECTION
        time_diff_with_meas = time_after_both_meas - qc.nodes["D"].sub_circuit_time
        if time_diff_with_meas > 0:
            qc._increase_duration(time_diff_with_meas, [], involved_nodes=["D"])
        if measurement_outcomes in [[0, 1], [1, 0]]:
            qc.X(2)
        # END FUSION CORRECTION
        qc.apply_gate(CZ_gate, cqubit=6, tqubit=7)        # 6, 1, 2, 4, 7, 9
        qc.apply_gate(CZ_gate, cqubit=1, tqubit=2)
        measurement_outcomes_2 = qc.measure([6, 1])       # 2, 4, 7, 9
        ghz_success_2 = measurement_outcomes_2[0] == measurement_outcomes_2[1]
        if ghz_success_1 and ghz_success_2:
            ghz_success = True
        else:
            ghz_success = False

        PBAR.update(25) if PBAR is not None else None

    qc.stabilizer_measurement(operation, nodes=["D", "C", "B", "A"], swap=False)

    PBAR.update(10) if PBAR is not None else None


def dyn_prot_4_6_sym_1_swap(qc: QuantumCircuit, *, operation):

    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None
        qc.start_sub_circuit("AB")
        qc.create_bell_pair(8, 5)
        qc.SWAP(5, 7, efficient=True)
        qc.SWAP(8, 9, efficient=True)

        PBAR.update(25) if PBAR is not None else None
        qc.start_sub_circuit("CD")
        qc.create_bell_pair(0, 3)
        qc.SWAP(3, 4, efficient=True)
        qc.SWAP(0, 2, efficient=True)

        PBAR.update(25) if PBAR is not None else None

        qc.start_sub_circuit("AC", forced_level=True)
        qc.create_bell_pair(3, 8)
        qc.apply_gate(CNOT_gate, cqubit=9, tqubit=8, electron_is_target=True, reverse=True)    # 8, 3, 7, 9
        qc.apply_gate(CNOT_gate, cqubit=3, tqubit=4)      # 8, 3, 7, 9, 4, 2
        qc.SWAP(3, 4, efficient=False)
        measurement_outcomes = qc.measure([8, 3], basis="Z")           # 7, 9, 4, 2
        # BEGIN FUSION CORRECTION: The correction in node C can only be applied AFTER the measurement in A
        time_between_meas = qc.nodes["A"].sub_circuit_time - qc.nodes["C"].sub_circuit_time
        if time_between_meas > 0:
            qc._increase_duration(time_between_meas, [], involved_nodes=["C"])
        if measurement_outcomes[0] == 1:
            qc.X(4)
        # The correction in node D can only be applied after both measurements in A and C
        time_after_both_meas = max(qc.nodes["A"].sub_circuit_time, qc.nodes["C"].sub_circuit_time)
        # END FUSION CORRECTION
        qc.create_bell_pair(3, 8)
        qc.apply_gate(CZ_gate, cqubit=8, tqubit=9)        # 8, 3, 7, 9, 4, 2
        qc.apply_gate(CZ_gate, cqubit=3, tqubit=4)
        measurement_outcomes_1 = qc.measure([8, 3])       # 7, 9, 4, 2
        ghz_success_1 = measurement_outcomes_1[0] == measurement_outcomes_1[1]

        PBAR.update(25) if PBAR is not None else None

        qc.start_sub_circuit("BD")
        success_bd = False
        while not success_bd:
            qc.create_bell_pair(5, 0)
            qc.SWAP(5, 6, efficient=True)
            qc.SWAP(0, 1, efficient=True)
            qc.create_bell_pair(5, 0)
            success_bd = qc.single_selection(CiY_gate, 5, 0, 6, 1, create_bell_pair=False)
        qc.SWAP(5, 6, efficient=True)
        qc.SWAP(0, 1, efficient=True)
        # BEGIN FUSION CORRECTION
        time_diff_with_meas = time_after_both_meas - qc.nodes["D"].sub_circuit_time
        if time_diff_with_meas > 0:
            qc._increase_duration(time_diff_with_meas, [], involved_nodes=["D"])
        if measurement_outcomes in [[0, 1], [1, 0]]:
            qc.X(2)
        # END FUSION CORRECTION
        qc.apply_gate(CZ_gate, cqubit=5, tqubit=7)        # 0, 5, 7, 9, 4, 2
        qc.apply_gate(CZ_gate, cqubit=0, tqubit=2)
        measurement_outcomes_2 = qc.measure([0, 5])       # 7, 9, 4, 2
        ghz_success_2 = measurement_outcomes_2[0] == measurement_outcomes_2[1]
        if ghz_success_1 and ghz_success_2:
            ghz_success = True
        else:
            ghz_success = False

        PBAR.update(25) if PBAR is not None else None

    qc.stabilizer_measurement(operation, nodes=["B", "A", "C", "D"], swap=True)

    PBAR.update(10) if PBAR is not None else None


def dyn_prot_4_14_1(qc: QuantumCircuit, *, operation):
    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None

        qc.start_sub_circuit("AB")
        success_ab = False
        while not success_ab:
            qc.create_bell_pair(13, 9)
            success_ab = qc.single_selection(CZ_gate, 12, 8)
            if not success_ab:
                continue
            success_ab2 = False
            while not success_ab2:
                qc.create_bell_pair(12, 8)
                success_ab2 = qc.single_selection(CNOT_gate, 11, 7)
            success_ab = qc.single_selection(CiY_gate, 12, 8, 13, 9, create_bell_pair=False)

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit("CD")
        success_cd = False
        while not success_cd:
            qc.create_bell_pair(6, 2)
            success_cd = qc.single_selection(CZ_gate, 5, 1)
            if not success_cd:
                continue
            success_cd2 = False
            while not success_cd2:
                qc.create_bell_pair(5, 1)
                success_cd2 = qc.single_selection(CZ_gate, 4, 0)
            success_cd = qc.single_selection(CNOT_gate, 5, 1, 6, 2, create_bell_pair=False)

        PBAR.update(30) if PBAR is not None else None

        qc.start_sub_circuit("AC")
        success_ac = False
        while not success_ac:
            qc.create_bell_pair(12, 5)
            success_ac = qc.single_selection(CZ_gate, 11, 4)
            if not success_ac:
                continue
            success_ac2 = False
            while not success_ac2:
                qc.create_bell_pair(11, 4)
                success_ac2 = qc.single_selection(CNOT_gate, 10, 3)
            success_ac = qc.single_selection(CiY_gate, 11, 4, 12, 5, create_bell_pair=False)

        qc.start_sub_circuit("BD")
        success_bd = False
        while not success_bd:
            qc.create_bell_pair(8, 1)
            success_bd = qc.single_selection(CZ_gate, 7, 0)

        qc.start_sub_circuit("AC", forced_level=True)
        qc.apply_gate(CNOT_gate, cqubit=13, tqubit=12, reverse=True)    # 5, 12, 9, 13
        qc.apply_gate(CNOT_gate, cqubit=6, tqubit=5, reverse=True)      # 5, 12, 9, 13, 2, 6
        measurement_outcomes = qc.measure([5, 12], basis="Z")           # 9, 13, 2, 6
        success = measurement_outcomes[0] == measurement_outcomes[1]
        qc.start_sub_circuit("AB")
        if not success:
            qc.X(13)
            qc.X(9)
        qc.start_sub_circuit("BD")
        qc.apply_gate(CZ_gate, cqubit=9, tqubit=8, reverse=True)        # 1, 8, 9, 13, 2, 6
        qc.apply_gate(CZ_gate, cqubit=2, tqubit=1, reverse=True)        # 1, 8, 9, 13, 2, 6
        measurement_outcomes2 = qc.measure([1, 8])
        ghz_success = measurement_outcomes2[0] == measurement_outcomes2[1]
        PBAR.update(30) if PBAR is not None else None

    qc.stabilizer_measurement(operation, nodes=["B", "A", "D", "C"], swap=False)

    PBAR.update(10) if PBAR is not None else None


def dyn_prot_4_14_1_swap(qc: QuantumCircuit, *, operation):
    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None

        qc.start_sub_circuit("AB")
        success_ab = False
        while not success_ab:
            qc.create_bell_pair("A-e", "B-e")
            qc.SWAP("A-e", "A-e+1", efficient=True)
            qc.SWAP("B-e", "B-e+1", efficient=True)
            success_ab = qc.single_selection(CZ_gate, "A-e", "B-e", swap=True)
            if not success_ab:
                continue
            success_ab2 = False
            while not success_ab2:
                qc.create_bell_pair("A-e", "B-e")
                qc.SWAP("A-e", "A-e+2", efficient=True)
                qc.SWAP("B-e", "B-e+2", efficient=True)
                success_ab2 = qc.single_selection(CNOT_gate, "A-e", "B-e", "A-e+2", "B-e+2", swap=True)
            qc.SWAP("A-e", "A-e+2", efficient=True)
            qc.SWAP("B-e", "B-e+2", efficient=True)
            success_ab = qc.single_selection(CiY_gate, "A-e", "B-e", create_bell_pair=False, swap=True)

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit("CD")
        success_cd = False
        while not success_cd:
            qc.create_bell_pair("C-e", "D-e")
            qc.SWAP("C-e", "C-e+1", efficient=True)
            qc.SWAP("D-e", "D-e+1", efficient=True)
            success_cd = qc.single_selection(CZ_gate, "C-e", "D-e", swap=True)
            if not success_cd:
                continue
            success_cd2 = False
            while not success_cd2:
                qc.create_bell_pair("C-e", "D-e")
                qc.SWAP("C-e", "C-e+2", efficient=True)
                qc.SWAP("D-e", "D-e+2", efficient=True)
                success_cd2 = qc.single_selection(CZ_gate, "C-e", "D-e", "C-e+2", "D-e+2", swap=True)
            qc.SWAP("C-e", "C-e+2", efficient=True)
            qc.SWAP("D-e", "D-e+2", efficient=True)
            success_cd = qc.single_selection(CNOT_gate, "C-e", "D-e", create_bell_pair=False, swap=True)

        PBAR.update(30) if PBAR is not None else None

        qc.start_sub_circuit("AC")
        success_ac = False
        while not success_ac:
            qc.create_bell_pair("A-e", "C-e")
            qc.SWAP("C-e", "C-e+2", efficient=True)
            qc.SWAP("A-e", "A-e+2", efficient=True)
            success_ac = qc.single_selection(CZ_gate, "A-e", "C-e", "A-e+2", "C-e+2", swap=True)
            if not success_ac:
                continue
            success_ac2 = False
            while not success_ac2:
                qc.create_bell_pair("A-e", "C-e")
                qc.SWAP("C-e", "C-e+3", efficient=True)
                qc.SWAP("A-e", "A-e+3", efficient=True)
                success_ac2 = qc.single_selection(CNOT_gate, "A-e", "C-e", "A-e+3", "C-e+3", swap=True)
            qc.SWAP("A-e", "A-e+3", efficient=True)
            qc.SWAP("C-e", "C-e+3", efficient=True)
            success_ac = qc.single_selection(CiY_gate, "A-e", "C-e", "A-e+2", "C-e+2", create_bell_pair=False,
                                             swap=True)

        qc.start_sub_circuit("BD")
        success_bd = False
        while not success_bd:
            qc.create_bell_pair("B-e", "D-e")
            qc.SWAP("B-e", "B-e+2", efficient=True)
            qc.SWAP("D-e", "D-e+2", efficient=True)
            success_bd = qc.single_selection(CZ_gate, "B-e", "D-e", "B-e+2", "D-e+2", swap=True)

        qc.start_sub_circuit("AC", forced_level=True)
        qc.SWAP("A-e", "A-e+1", efficient=True)
        qc.SWAP("C-e", "C-e+1", efficient=True)
        qc.apply_gate(CNOT_gate, cqubit="A-e", tqubit="A-e+2", reverse=True)    # 5, 12, 9, 13
        qc.apply_gate(CNOT_gate, cqubit="C-e", tqubit="C-e+2", reverse=True)      # 5, 12, 9, 13, 2, 6
        qc.SWAP("A-e", "A-e+2", efficient=True)
        qc.SWAP("C-e", "C-e+2", efficient=True)
        measurement_outcomes = qc.measure(["C-e", "A-e"], basis="Z")           # 9, 13, 2, 6
        success = type(measurement_outcomes) == SKIP or measurement_outcomes[0] == measurement_outcomes[1]
        if not success:
            qc.start_sub_circuit("AB")
            qc.X("A-e+2")
            qc.X("B-e+1")
        qc.start_sub_circuit("BD")
        qc.SWAP("B-e", "B-e+1", efficient=True)
        qc.SWAP("D-e", "D-e+1", efficient=True)
        qc.apply_gate(CZ_gate, cqubit="B-e", tqubit="B-e+2", reverse=True)  # 5, 12, 9, 13
        qc.apply_gate(CZ_gate, cqubit="D-e", tqubit="D-e+2", reverse=True)  # 5, 12, 9, 13, 2, 6
        qc.SWAP("B-e", "B-e+2", efficient=True)
        qc.SWAP("D-e", "D-e+2", efficient=True)
        measurement_outcomes2 = qc.measure(["D-e", "B-e"], basis="X")
        ghz_success = type(measurement_outcomes2) == SKIP or measurement_outcomes2[0] == measurement_outcomes2[1]
        PBAR.update(30) if PBAR is not None else None

    qc.stabilizer_measurement(operation, nodes=["B", "A", "D", "C"], swap=True)

    PBAR.update(10) if PBAR is not None else None


def dyn_prot_4_22_1(qc: QuantumCircuit, *, operation):
    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None

        qc.start_sub_circuit("AB")
        success_ab = False
        while not success_ab:
            qc.create_bell_pair(15, 11)
            success_ab = qc.single_selection(CiY_gate, 14, 10, 15, 11)
            if not success_ab:
                continue
            success_ab2 = False
            while not success_ab2:
                qc.create_bell_pair(14, 10)
                success_ab2 = qc.single_selection(CZ_gate, 13, 9)
            success_ab = qc.single_selection(CNOT_gate, 14, 10, 15, 11, create_bell_pair=False)

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit("CD")
        success_cd = False
        while not success_cd:
            qc.create_bell_pair(6, 2)
            success_cd = qc.single_selection(CiY_gate, 7, 3, 6, 2)
            if not success_cd:
                continue
            success_cd = qc.single_selection(CZ_gate, 7, 3, 6, 2)
            if not success_cd:
                continue
            success_cd2 = False
            while not success_cd2:
                qc.create_bell_pair(7, 3)
                success_cd2 = qc.single_selection(CiY_gate, 5, 1, 7, 3)
            success_cd = qc.single_selection(CNOT_gate, 7, 3, 6, 2, create_bell_pair=False)
            if not success_cd:
                continue
            success_cd2 = False
            while not success_cd2:
                qc.create_bell_pair(7, 3)
                success_cd2 = qc.single_selection(CNOT_gate, 5, 1, 7, 3)
                if not success_cd2:
                    continue
                success_cd3 = False
                while not success_cd3:
                    qc.create_bell_pair(5, 1)
                    success_cd3 = qc.single_selection(CiY_gate, 4, 0, 5, 1)
                success_cd2 = qc.single_selection(CZ_gate, 5, 1, 7, 3, create_bell_pair=False)
            success_cd = qc.single_selection(CZ_gate, 7, 3, 6, 2, create_bell_pair=False)

        PBAR.update(30) if PBAR is not None else None

        qc.start_sub_circuit("AC")
        success_ac = False
        while not success_ac:
            qc.create_bell_pair(7, 14)
            success_ac = qc.single_selection(CiY_gate, 13, 5, 14, 7)
            if not success_ac:
                continue
            success_ac2 = False
            while not success_ac2:
                qc.create_bell_pair(13, 5)
                success_ac2 = qc.single_selection(CZ_gate, 12, 4)
            success_ac = qc.single_selection(CNOT_gate, 13, 5, 14, 7, create_bell_pair=False)

        qc.start_sub_circuit("BD")
        success_bd = False
        while not success_bd:
            qc.create_bell_pair(3, 10)
            success_bd = qc.single_selection(CNOT_gate, 9, 1, 10, 3)
            if not success_bd:
                continue
            success_bd2 = False
            while not success_bd2:
                qc.create_bell_pair(8, 0)
                success_bd2 = qc.single_selection(CZ_gate, 9, 1, 8, 0)
                if not success_bd2:
                    continue
                qc.create_bell_pair(9, 1)
                success_bd2 = qc.single_selection(CZ_gate, 8, 0, 9, 1, create_bell_pair=False)
            success_bd = qc.single_selection(CiY_gate, 9, 1, 10, 3, create_bell_pair=False)

        qc.start_sub_circuit("AB", forced_level=True)
        qc.apply_gate(CNOT_gate, cqubit=15, tqubit=14, reverse=True)    # 14, 7, 11, 15
        # qc.start_sub_circuit("C")
        qc.apply_gate(CNOT_gate, cqubit=11, tqubit=10, reverse=True)      # 10, 3, 14, 7, 11, 15
        # qc.start_sub_circuit("AC")
        # qc._thread_safe_printing = False
        # qc.draw_circuit()
        measurement_outcomes = qc.measure([10, 14], basis="Z")           # 3, 7, 11, 15
        success = type(measurement_outcomes) == SKIP or measurement_outcomes[0] == measurement_outcomes[1]
        qc.start_sub_circuit("AC")
        if not success:
            qc.X(15)
            qc.X(7)
        qc.start_sub_circuit("CD")
        qc.apply_gate(CZ_gate, cqubit=7, tqubit=6, reverse=True)        # 2, 6, 3, 7, 11, 15
        # qc.start_sub_circuit("D")
        qc.apply_gate(CZ_gate, cqubit=3, tqubit=2)
        # qc.start_sub_circuit("BD")
        measurement_outcomes2 = qc.measure([2, 6])      # 3, 7, 11, 15
        ghz_success = type(measurement_outcomes) == SKIP or measurement_outcomes2[1]
        PBAR.update(30) if PBAR is not None else None

    qc.stabilizer_measurement(operation, nodes=["D", "C", "B", "A"], swap=False)

    PBAR.update(10) if PBAR is not None else None

    qc.append_print_lines("\nGHZ fidelity: {}\n".format(qc.ghz_fidelity))


def stringent(qc: QuantumCircuit, *, operation):
    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None

        # Step 1-8 from Table D.2 (Thesis Naomi Nickerson)
        success_ab = False
        qc.start_sub_circuit("AB")
        while not success_ab:
            qc.create_bell_pair(11, 8)
            success_ab = qc.double_selection(CZ_gate, 10, 7)
            if not success_ab:
                continue
            success_ab = qc.double_selection(CNOT_gate, 10, 7)
            if not success_ab:
                continue

            success_ab = qc.double_dot(CZ_gate, 10, 7)
            if not success_ab:
                continue
            success_ab = qc.double_dot(CNOT_gate, 10, 7)
            if not success_ab:
                continue

        PBAR.update(20) if PBAR is not None else None

        # Step 1-8 from Table D.2 (Thesis Naomi Nickerson)
        success_cd = False
        qc.start_sub_circuit("CD")
        while not success_cd:
            qc.create_bell_pair(5, 2)
            success_cd = qc.double_selection(CZ_gate, 4, 1)
            if not success_cd:
                continue
            success_cd = qc.double_selection(CNOT_gate, 4, 1)
            if not success_cd:
                continue

            success_cd = qc.double_dot(CZ_gate, 4, 1)
            if not success_cd:
                continue
            success_cd = qc.double_dot(CNOT_gate, 4, 1)
            if not success_cd:
                continue

        PBAR.update(20) if PBAR is not None else None

        # Step 9-11 from Table D.2 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("AC")
        # Return success (even parity measurement). If False (uneven), X-gate must be drawn at second single dot
        success, single_selection_success = qc.double_dot(CZ_gate, 10, 4, parity_check=False)
        qc.start_sub_circuit("BD")
        ghz_success = qc.double_dot(CZ_gate, 7, 1, draw_X_gate=not success)
        if not ghz_success or not single_selection_success:
            ghz_success = False
            continue

        PBAR.update(20) if PBAR is not None else None

        # Step 12-14 from Table D.2 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("AC", forced_level=True)
        ghz_success_1 = qc.double_dot(CZ_gate, 10, 4)
        qc.start_sub_circuit("BD")
        ghz_success_2 = qc.double_dot(CZ_gate, 7, 1)
        if any([not ghz_success_1, not ghz_success_2]):
            ghz_success = False
            continue

        PBAR.update(20) if PBAR is not None else None

    # Step 15 from Table D.2 (Thesis Naomi Nickerson)
    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    qc.stabilizer_measurement(operation, nodes=["B", "A", "D", "C"])

    PBAR.update(10) if PBAR is not None else None


def expedient_swap(qc: QuantumCircuit, *, operation, tqubit=None):
    ghz_success = False
    while not ghz_success:
        PBAR.reset() if PBAR is not None else None

        qc.start_sub_circuit("AB")
        success_ab = False
        while not success_ab:
            qc.create_bell_pair("A-e", "B-e")
            qc.SWAP("A-e", "A-e+1", efficient=True)
            qc.SWAP("B-e", "B-e+1", efficient=True)
            success_ab = qc.double_selection(CZ_gate, "A-e", "B-e", swap=True)
            if not success_ab:
                continue
            success_ab = qc.double_selection(CNOT_gate, "A-e", "B-e", swap=True)

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit("CD")
        success_cd = False
        while not success_cd:
            qc.create_bell_pair("C-e", "D-e")
            qc.SWAP("C-e", "C-e+1", efficient=True)
            qc.SWAP("D-e", "D-e+1", efficient=True)
            success_cd = qc.double_selection(CZ_gate, "C-e", "D-e", swap=True)
            if not success_cd:
                continue
            success_cd = qc.double_selection(CNOT_gate, "C-e", "D-e", swap=True)

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit('AC')
        success_1 = qc.single_dot(CZ_gate, "A-e", "C-e", parity_check=False, swap=True)
        qc.start_sub_circuit('BD')
        ghz_success = qc.single_dot(CZ_gate, "B-e", "D-e", draw_X_gate=not success_1, swap=True)
        if not ghz_success:
            continue

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit('AC', forced_level=True)
        ghz_success_1 = qc.single_dot(CZ_gate, "A-e", "C-e", swap=True)
        qc.start_sub_circuit("BD")
        ghz_success_2 = qc.single_dot(CZ_gate, "B-e", "D-e", swap=True)
        if any([not ghz_success_1, not ghz_success_2]):
            ghz_success = False
            continue

    PBAR.update(20) if PBAR is not None else None

    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    qc.stabilizer_measurement(operation, nodes=["B", "A", "D", "C"], swap=True, tqubit=tqubit)

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
            if not qc.double_selection(CZ_gate, 9, 6, swap=True):
                continue
            if not qc.double_selection(CNOT_gate, 9, 6, swap=True):
                continue
            if not qc.double_dot(CZ_gate, 9, 6, swap=True):
                continue
            success_ab = qc.double_dot(CNOT_gate, 9, 6, swap=True)

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit("CD")
        success_cd = False
        while not success_cd:
            qc.create_bell_pair(3, 0)
            qc.SWAP(3, 4, efficient=True)
            qc.SWAP(0, 1, efficient=True)
            if not qc.double_selection(CZ_gate, 3, 0, swap=True):
                continue
            if not qc.double_selection(CNOT_gate, 3, 0, swap=True):
                continue
            if not qc.double_dot(CZ_gate, 3, 0, swap=True):
                continue
            success_cd = qc.double_dot(CNOT_gate, 3, 0, swap=True)

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit("AC")
        success, single_selection_success = qc.double_dot(CZ_gate, 9, 3, parity_check=False, swap=True)
        qc.start_sub_circuit("BD")
        ghz_success = qc.double_dot(CZ_gate, 6, 0, draw_X_gate=not success, swap=True)
        if not ghz_success or not single_selection_success:
            ghz_success = False
            continue

        PBAR.update(20) if PBAR is not None else None

        qc.start_sub_circuit("AC", forced_level=True)
        ghz_success_1 = qc.double_dot(CZ_gate, 9, 3, swap=True)
        qc.start_sub_circuit("BD")
        ghz_success_2 = qc.double_dot(CZ_gate, 6, 0, swap=True)
        if any([not ghz_success_1, not ghz_success_2]):
            ghz_success = False
            continue

        PBAR.update(20) if PBAR is not None else None

    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    qc.stabilizer_measurement(operation, nodes=["B", "A", "D", "C"], swap=True)

    PBAR.update(10) if PBAR is not None else None


def duo_structure(qc: QuantumCircuit, *, operation):
    qc.start_sub_circuit("AB")
    qc.create_bell_pair(2, 5)
    qc.double_selection(CZ_gate, 1, 4)
    qc.double_selection(CNOT_gate, 1, 4)

    qc.stabilizer_measurement(operation, nodes=["A", "B"])

    PBAR.update(50) if PBAR is not None else None


def weight_2_4_swap(qc: QuantumCircuit, *, operation):
    expedient_swap(qc, operation=operation, tqubit=[26, 30, 18, 22])

    return [[30, 26, 22, 18], [28, 24, 20, 16]]


def weight_3_swap(qc: QuantumCircuit, *, operation):
    dyn_prot_3_8_1_swap(qc, operation=operation, tqubit=[30, 28, 26, 22])

    return [[30, 26, 22, 28], [18, 24, 20, 16]]
