from circuit_simulation.circuit_simulator import *
import pickle


def create_quantum_circuit(protocol, *, pg, pm, pm_1, pn, decoherence, bell_pair_creation_success, measurement_duration,
                           bell_pair_creation_duration, pulse_duration, probabilistic, lkt_1q, lkt_2q,
                           fixed_lde_attempts, network_noise_type):
    """
        Initialises a QuantumCircuit object corresponding to the protocol requested.

        Parameters
        ----------
        protocol : str
            Name of the protocol for which the QuantumCircuit object should be initialised

        For other parameters, please see QuantumCircuit class for more information

    """
    if protocol == 'monolithic':
        qc = QuantumCircuit(9, 2, noise=True, pg=pg, pm=pm, pm_1=pm_1, basis_transformation_noise=True,
                            thread_safe_printing=True, bell_creation_duration=bell_pair_creation_duration,
                            measurement_duration=measurement_duration, single_qubit_gate_lookup=lkt_1q,
                            two_qubit_gate_lookup=lkt_2q, decoherence=decoherence, pulse_duration=pulse_duration)

    elif protocol == 'duo_structure':
        qc = QuantumCircuit(14, 2, noise=True, basis_transformation_noise=False, pg=pg, pm=pm, pm_1=pm_1, pn=pn,
                            thread_safe_printing=True, probabilistic=probabilistic, T1_lde=2,
                            decoherence=decoherence, p_bell_success=bell_pair_creation_success, T1_idle=(5 * 60),
                            T2_idle=10, T2_idle_electron=1, T2_lde=2, measurement_duration=measurement_duration,
                            bell_creation_duration=bell_pair_creation_duration, pulse_duration=pulse_duration,
                            single_qubit_gate_lookup=lkt_1q, two_qubit_gate_lookup=lkt_2q, T1_idle_electron=1000,
                            no_single_qubit_error=True, fixed_lde_attempts=fixed_lde_attempts,
                            network_noise_type=network_noise_type)

        qc.define_node("A", qubits=[0, 1, 2, 6, 8], electron_qubits=2, data_qubits=[6, 8])
        qc.define_node("B", qubits=[3, 4, 5, 10, 12], electron_qubits=5, data_qubits=[10, 12])

        qc.define_sub_circuit("AB", waiting_qubits=[6, 8, 10, 12])

    elif protocol == 'duo_structure_2':
        qc = QuantumCircuit(32, 5, noise=True, basis_transformation_noise=False, pg=pg, pm=pm, pm_1=pm_1, pn=pn,
                            thread_safe_printing=False, probabilistic=probabilistic, T1_lde=2,
                            decoherence=decoherence, p_bell_success=bell_pair_creation_success, T1_idle=(5 * 60),
                            T2_idle=10, T2_idle_electron=1, T2_lde=2, measurement_duration=measurement_duration,
                            bell_creation_duration=bell_pair_creation_duration, pulse_duration=pulse_duration,
                            single_qubit_gate_lookup=lkt_1q, two_qubit_gate_lookup=lkt_2q, T1_idle_electron=1000,
                            no_single_qubit_error=True, fixed_lde_attempts=fixed_lde_attempts,
                            network_noise_type=network_noise_type)

        qc.define_node("A", qubits=[30, 28, 15, 14, 13, 12], electron_qubits=12, data_qubits=[30, 28])
        qc.define_node("B", qubits=[26, 24, 11, 10, 9, 8], electron_qubits=8, data_qubits=[26, 24])
        qc.define_node("C", qubits=[22, 20, 7, 6, 5, 4], electron_qubits=4, data_qubits=[22, 10])
        qc.define_node("D", qubits=[18, 16, 3, 2, 1, 0], electron_qubits=0, data_qubits=[18, 16])

        qc.define_sub_circuit("AB", waiting_qubits=[30, 28, 13, 26, 24, 9])
        qc.define_sub_circuit("CD", waiting_qubits=[22, 20, 5, 18, 16, 1], concurrent_sub_circuits="AB")
        qc.define_sub_circuit("AC", waiting_qubits=[30, 28, 12, 22, 20, 5])
        qc.define_sub_circuit("BD", waiting_qubits=[26, 24, 9, 18, 16, 1], concurrent_sub_circuits="AC")
        qc.define_sub_circuit("A")
        qc.define_sub_circuit("B")
        qc.define_sub_circuit("C")
        qc.define_sub_circuit("D", concurrent_sub_circuits=["A", "B", "C"])

    else:
        qc = QuantumCircuit(20, 2, noise=True, basis_transformation_noise=False, pg=pg, pm=pm, pm_1=pm_1, pn=pn,
                            thread_safe_printing=True, probabilistic=probabilistic, T1_lde=2,
                            decoherence=decoherence, p_bell_success=bell_pair_creation_success, T1_idle=(5 * 60),
                            T2_idle=10, T2_idle_electron=1, T2_lde=2, measurement_duration=measurement_duration,
                            bell_creation_duration=bell_pair_creation_duration, pulse_duration=pulse_duration,
                            single_qubit_gate_lookup=lkt_1q, two_qubit_gate_lookup=lkt_2q, T1_idle_electron=1000,
                            no_single_qubit_error=True, fixed_lde_attempts=fixed_lde_attempts,
                            network_noise_type=network_noise_type)

        qc.define_node("A", qubits=[18, 11, 10, 9], electron_qubits=9, data_qubits=18)
        qc.define_node("B", qubits=[16, 8, 7, 6], electron_qubits=6, data_qubits=16)
        qc.define_node("C", qubits=[14, 5, 4, 3], electron_qubits=3, data_qubits=14)
        qc.define_node("D", qubits=[12, 2, 1, 0], electron_qubits=0, data_qubits=12)

        qc.define_sub_circuit("AB", waiting_qubits=[10, 7, 18, 16])
        qc.define_sub_circuit("CD", waiting_qubits=[4, 1, 14, 12], concurrent_sub_circuits="AB")
        qc.define_sub_circuit("AC", waiting_qubits=[10, 4, 18, 14])
        qc.define_sub_circuit("BD", waiting_qubits=[7, 1, 16, 12], concurrent_sub_circuits="AC")
        qc.define_sub_circuit("A")
        qc.define_sub_circuit("B")
        qc.define_sub_circuit("C")
        qc.define_sub_circuit("D", concurrent_sub_circuits=["A", "B", "C"])

    return qc


def monolithic(qc: QuantumCircuit, *, operation, color, save_latex_pdf, pbar, draw_circuit, to_console):
    qc.set_qubit_states({0: ket_p})
    qc.apply_gate(operation, cqubit=0, tqubit=1)
    qc.apply_gate(operation, cqubit=0, tqubit=3)
    qc.apply_gate(operation, cqubit=0, tqubit=5)
    qc.apply_gate(operation, cqubit=0, tqubit=7)
    qc.measure(0, probabilistic=False)

    pbar.update(50) if pbar is not None else None

    if draw_circuit:
        qc.draw_circuit(not color)
    if save_latex_pdf:
        qc.draw_circuit_latex()
    stab_rep = "Z" if operation == CZ_gate else "X"
    _, dataframe = qc.get_superoperator([1, 3, 5, 7], stab_rep, no_color=(not color), stabilizer_protocol=True,
                                        print_to_console=to_console)
    pbar.update(50) if pbar is not None else None

    print_lines = qc.print_lines
    cut_off_reached = qc.cut_off_time_reached
    qc.reset()

    return (dataframe, cut_off_reached), print_lines


def expedient(qc: QuantumCircuit, *, operation, color, save_latex_pdf, pbar, draw_circuit, to_console):
    ghz_success = False
    while not ghz_success:
        pbar.reset() if pbar is not None else None

        # Step 1-2 from Table D.1 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("AB")
        success_ab = False
        while not success_ab:
            qc.create_bell_pair(11, 8)
            success_ab = qc.double_selection(CZ_gate, 10, 7, retry=False)
            if not success_ab:
                continue
            success_ab = qc.double_selection(CNOT_gate, 10, 7, retry=False)

        pbar.update(20) if pbar is not None else None

        # Step 1-2 from Table D.1 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("CD")
        success_cd = False
        while not success_cd:
            qc.create_bell_pair(5, 2)
            success_cd = qc.double_selection(CZ_gate, 4, 1, retry=False)
            if not success_cd:
                continue
            success_cd = qc.double_selection(CNOT_gate, 4, 1, retry=False)

        pbar.update(20) if pbar is not None else None

        # Step 3-5 from Table D.1 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("AC")
        # Return success (even parity measurement). If False (uneven), X-gate must be drawn at second single dot
        success_1 = qc.single_dot(CZ_gate, 10, 4, parity_check=False)
        qc.start_sub_circuit("BD")
        ghz_success = qc.single_dot(CZ_gate, 7, 1, draw_X_gate=not success_1, retry=False)
        if not ghz_success:
            continue

        pbar.update(20) if pbar is not None else None

        # Step 6-8 from Table D.1 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("AC", forced_level=True)
        ghz_success_1 = qc.single_dot(CZ_gate, 10, 4, retry=False)
        qc.start_sub_circuit("BD")
        ghz_success_2 = qc.single_dot(CZ_gate, 7, 1, retry=False)
        if any([not ghz_success_1, not ghz_success_2]):
            ghz_success = False
            continue

        pbar.update(20) if pbar is not None else None

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

    pbar.update(10) if pbar is not None else None

    if draw_circuit:
        qc.draw_circuit(no_color=not color, color_nodes=True)

    if save_latex_pdf:
        qc.draw_circuit_latex()
    stab_rep = "Z" if operation == CZ_gate else "X"
    _, dataframe = qc.get_superoperator([18, 16, 14, 12], stab_rep, no_color=(not color), stabilizer_protocol=True,
                                        print_to_console=to_console, use_exact_path=True)

    pbar.update(10) if pbar is not None else None

    qc.append_print_lines("\nTotal circuit duration: {} seconds".format(qc.total_duration)) if draw_circuit else None
    print_lines = qc.print_lines
    cut_off_reached = qc.cut_off_time_reached
    qc.reset()

    return (dataframe, cut_off_reached), print_lines


def stringent(qc, *, operation, color, save_latex_pdf, pbar, draw_circuit, to_console):
    ghz_success = False
    while not ghz_success:
        pbar.reset() if pbar is not None else None

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

        pbar.update(20) if pbar is not None else None

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

        pbar.update(20) if pbar is not None else None

        # Step 9-11 from Table D.2 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("AC")
        # Return success (even parity measurement). If False (uneven), X-gate must be drawn at second single dot
        success, single_selection_success = qc.double_dot(CZ_gate, 10, 4, parity_check=False)
        qc.start_sub_circuit("BD")
        ghz_success = qc.double_dot(CZ_gate, 7, 1, draw_X_gate=not success, retry=False)
        if not ghz_success or not single_selection_success:
            ghz_success = False
            continue

        pbar.update(20) if pbar is not None else None

        # Step 12-14 from Table D.2 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("AC", forced_level=True)
        ghz_success_1 = qc.double_dot(CZ_gate, 10, 4, retry=False)
        qc.start_sub_circuit("BD")
        ghz_success_2 = qc.double_dot(CZ_gate, 7, 1, retry=False)
        if any([not ghz_success_1, not ghz_success_2]):
            ghz_success = False
            continue

        pbar.update(20) if pbar is not None else None

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

    pbar.update(10) if pbar is not None else None

    if draw_circuit:
        qc.draw_circuit(no_color=not color, color_nodes=True)

    if save_latex_pdf:
        qc.draw_circuit_latex()

    stab_rep = "Z" if operation == CZ_gate else "X"
    _, dataframe = qc.get_superoperator([18, 16, 14, 12], stab_rep, no_color=(not color), stabilizer_protocol=True,
                                        print_to_console=to_console)

    pbar.update(10) if pbar is not None else None

    qc.append_print_lines("\nTotal circuit duration: {} seconds".format(qc.total_duration)) if draw_circuit else None
    print_lines = qc.print_lines
    cut_off_reached = qc.cut_off_time_reached
    qc.reset()

    return (dataframe, cut_off_reached), print_lines


def expedient_swap(qc, *, operation, color, save_latex_pdf, pbar, draw_circuit, to_console):
    ghz_success = False
    while not ghz_success:
        pbar.reset() if pbar is not None else None

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

        pbar.update(20) if pbar is not None else None

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

        pbar.update(20) if pbar is not None else None

        qc.start_sub_circuit('AC')
        success_1 = qc.single_dot_swap(CZ_gate, 9, 3, parity_check=False)
        qc.start_sub_circuit('BD')
        ghz_success = qc.single_dot_swap(CZ_gate, 6, 0, draw_X_gate=not success_1, retry=False)
        if not ghz_success:
            continue

        pbar.update(20) if pbar is not None else None

        qc.start_sub_circuit('AC', forced_level=True)
        ghz_success_1 = qc.single_dot_swap(CZ_gate, 9, 3, retry=False)
        qc.start_sub_circuit("BD")
        ghz_success_2 = qc.single_dot_swap(CZ_gate, 6, 0, retry=False)
        if any([not ghz_success_1, not ghz_success_2]):
            ghz_success = False
            continue

    pbar.update(20) if pbar is not None else None

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

    pbar.update(10) if pbar is not None else None

    if draw_circuit:
        qc.draw_circuit(no_color=not color, color_nodes=True)

    if save_latex_pdf:
        qc.draw_circuit_latex()
    stab_rep = "Z" if operation == CZ_gate else "X"
    _, dataframe = qc.get_superoperator([18, 16, 14, 12], stab_rep, no_color=(not color), stabilizer_protocol=True,
                                        print_to_console=to_console)

    pbar.update(10) if pbar is not None else None

    qc.append_print_lines("\nTotal circuit duration: {} seconds".format(qc.total_duration)) if draw_circuit else None
    print_lines = qc.print_lines
    cut_off_reached = qc.cut_off_time_reached
    qc.reset()

    return (dataframe, cut_off_reached), print_lines


def stringent_swap(qc, *, operation, color, save_latex_pdf, pbar, draw_circuit, to_console):
    ghz_success = False
    while not ghz_success:
        pbar.reset() if pbar is not None else None

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

        pbar.update(20) if pbar is not None else None

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

        pbar.update(20) if pbar is not None else None

        qc.start_sub_circuit("AC")
        success, single_selection_success = qc.double_dot_swap(CZ_gate, 9, 3, parity_check=False)
        qc.start_sub_circuit("BD")
        ghz_success = qc.double_dot_swap(CZ_gate, 6, 0, draw_X_gate=not success, retry=False)
        if not ghz_success or not single_selection_success:
            ghz_success = False
            continue

        pbar.update(20) if pbar is not None else None

        qc.start_sub_circuit("AC", forced_level=True)
        ghz_success_1 = qc.double_dot_swap(CZ_gate, 9, 3, retry=False)
        qc.start_sub_circuit("BD")
        ghz_success_2 = qc.double_dot_swap(CZ_gate, 6, 0, retry=False)
        if any([not ghz_success_1, not ghz_success_2]):
            ghz_success = False
            continue

        pbar.update(20) if pbar is not None else None

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

    pbar.update(10) if pbar is not None else None

    if draw_circuit:
        qc.draw_circuit(no_color=not color, color_nodes=True)

    if save_latex_pdf:
        qc.draw_circuit_latex()

    stab_rep = "Z" if operation == CZ_gate else "X"

    _, dataframe = qc.get_superoperator([18, 16, 14, 12], stab_rep, no_color=(not color), stabilizer_protocol=True,
                                        print_to_console=to_console)

    pbar.update(10) if pbar is not None else None

    qc.append_print_lines("\nTotal circuit duration: {} seconds".format(qc.total_duration)) if draw_circuit else None
    print_lines = qc.print_lines
    cut_off_reached = qc.cut_off_time_reached
    qc.reset()

    return (dataframe, cut_off_reached), print_lines


def duo_structure(qc: QuantumCircuit, *, operation, color, save_latex_pdf, pbar, draw_circuit, to_console):
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

    pbar.update(50) if pbar is not None else None

    if draw_circuit:
        qc.draw_circuit(not color, color_nodes=True)
    if save_latex_pdf:
        qc.draw_circuit_latex()
    stab_rep = "Z" if operation == CZ_gate else "X"
    _, dataframe = qc.get_superoperator([6, 8, 10, 12], stab_rep, no_color=(not color), stabilizer_protocol=True,
                                        print_to_console=to_console)
    pbar.update(50) if pbar is not None else None

    qc.append_print_lines("\nTotal circuit duration: {} seconds".format(qc.total_duration)) if draw_circuit else None
    print_lines = qc.print_lines
    cut_off_reached = qc.cut_off_time_reached
    qc.reset()

    return (None, cut_off_reached), print_lines


def duo_structure_2(qc: QuantumCircuit, *, operation, color, save_latex_pdf, pbar, draw_circuit, to_console):
    ghz_success = False
    while not ghz_success:
        pbar.reset() if pbar is not None else None

        # Step 1-2 from Table D.1 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("AB")
        success_ab = False
        while not success_ab:
            qc.create_bell_pair(15, 11)
            success_ab = qc.double_selection(CZ_gate, 14, 10, retry=False)
            if not success_ab:
                continue
            success_ab = qc.double_selection(CNOT_gate, 14, 10, retry=False)

        pbar.update(20) if pbar is not None else None

        # Step 1-2 from Table D.1 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("CD")
        success_cd = False
        while not success_cd:
            qc.create_bell_pair(7, 3)
            success_cd = qc.double_selection(CZ_gate, 6, 2, retry=False)
            if not success_cd:
                continue
            success_cd = qc.double_selection(CNOT_gate, 6, 2, retry=False)

        pbar.update(20) if pbar is not None else None

        # Step 3-5 from Table D.1 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("AC")
        # Return success (even parity measurement). If False (uneven), X-gate must be drawn at second single dot
        success_1 = qc.single_dot(CZ_gate, 14, 6, parity_check=False)
        qc.start_sub_circuit("BD")
        ghz_success = qc.single_dot(CZ_gate, 10, 2, draw_X_gate=not success_1, retry=False)
        if not ghz_success:
            continue

        pbar.update(20) if pbar is not None else None

        # Step 6-8 from Table D.1 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("AC", forced_level=True)
        ghz_success_1 = qc.single_dot(CZ_gate, 14, 6, retry=False)
        qc.start_sub_circuit("BD")
        ghz_success_2 = qc.single_dot(CZ_gate, 10, 2, retry=False)
        if any([not ghz_success_1, not ghz_success_2]):
            ghz_success = False
            continue

        pbar.update(20) if pbar is not None else None

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

    pbar.update(10) if pbar is not None else None

    if draw_circuit:
        qc.draw_circuit(no_color=not color, color_nodes=True)

    if save_latex_pdf:
        qc.draw_circuit_latex()
    stab_rep = "Z" if operation == CZ_gate else "X"
    _, dataframe = qc.get_superoperator([30, 26, 22, 18], stab_rep, no_color=(not color),
                                        stabilizer_protocol=False, print_to_console=to_console, use_exact_path=True)
    _, dataframe_idle = qc.get_superoperator([28, 24, 20, 16], stab_rep, no_color=(not color),
                                             stabilizer_protocol=True, print_to_console=to_console, use_exact_path=True,
                                             idle_data_qubit=4)

    pbar.update(10) if pbar is not None else None

    qc.append_print_lines("\nTotal circuit duration: {} seconds".format(qc.total_duration)) if draw_circuit else None
    print_lines = qc.print_lines
    cut_off_reached = qc.cut_off_time_reached
    qc.reset()

    return ([dataframe, dataframe_idle], cut_off_reached), print_lines
