from circuit_simulation.circuit_simulator import *
import time


def monolithic(*, operation, pg, pm, pm_1, color, bell_pair_creation_duration, measurement_duration, pulse_duration,
               lkt_1q, lkt_2q, save_latex_pdf, pbar, draw_circuit, to_console, decoherence, **kwargs):
    qc = QuantumCircuit(9, 2, noise=True, pg=pg, pm=pm, pm_1=pm_1, basis_transformation_noise=True,
                        thread_safe_printing=True, bell_creation_duration=bell_pair_creation_duration,
                        measurement_duration=measurement_duration, single_qubit_gate_lookup=lkt_1q,
                        two_qubit_gate_lookup=lkt_2q, decoherence=decoherence, pulse_duration=pulse_duration,)
    qc.set_qubit_states({0: ket_p})
    qc.apply_gate(operation, cqubit=0, tqubit=1)
    qc.apply_gate(operation, cqubit=0, tqubit=3)
    qc.apply_gate(operation, cqubit=0, tqubit=5)
    qc.apply_gate(operation, cqubit=0, tqubit=7)
    qc.measure([0])

    if pbar is not None:
        pbar.update(50)

    if draw_circuit:
        qc.draw_circuit(not color)
    if save_latex_pdf:
        qc.draw_circuit_latex()
    stab_rep = "Z" if operation == CZ_gate else "X"
    _, dataframe = qc.get_superoperator([1, 3, 5, 7], stab_rep, no_color=(not color), stabilizer_protocol=True,
                                        print_to_console=to_console)
    if pbar is not None:
        pbar.update(50)

    return (dataframe, qc.cut_off_time_reached), qc.print_lines


def expedient(*, operation, pg, pm, pm_1, pn, color, decoherence, bell_pair_creation_success, measurement_duration,
              bell_pair_creation_duration, pulse_duration, probabilistic, lkt_1q, lkt_2q, save_latex_pdf, pbar,
              draw_circuit, to_console):
    start = time.time()
    qc = QuantumCircuit(20, 2, noise=True, basis_transformation_noise=False, pg=pg, pm=pm, pm_1=pm_1, pn=pn,
                        network_noise_type=1, thread_safe_printing=True, probabilistic=probabilistic, T1_lde=2,
                        decoherence=decoherence, p_bell_success=bell_pair_creation_success, T1_idle=(5*60), T2_idle=10,
                        measurement_duration=measurement_duration, bell_creation_duration=bell_pair_creation_duration,
                        pulse_duration=pulse_duration, single_qubit_gate_lookup=lkt_1q, two_qubit_gate_lookup=lkt_2q,
                        T1_idle_electron=100, T2_idle_electron=1, T2_lde=2, no_single_qubit_error=True)

    qc.define_node("A", qubits=[18, 11, 10, 9], electron_qubits=11, data_qubits=18)
    qc.define_node("B", qubits=[16, 8, 7, 6], electron_qubits=8, data_qubits=16)
    qc.define_node("C", qubits=[14, 5, 4, 3], electron_qubits=5, data_qubits=14)
    qc.define_node("D", qubits=[12, 2, 1, 0], electron_qubits=2, data_qubits=12)

    qc.define_sub_circuit("AB", [11, 10, 9, 8, 7, 6, 18, 16], waiting_qubits=[10, 7, 18, 16])
    qc.define_sub_circuit("CD", [5, 4, 3, 2, 1, 0, 14, 12], waiting_qubits=[4, 1, 14, 12], concurrent_sub_circuits="AB")
    qc.define_sub_circuit("AC", [11, 5, 10, 9, 4, 3, 18, 14], waiting_qubits=[10, 4, 18, 14])
    qc.define_sub_circuit("BD", [8, 2, 7, 6, 1, 0, 16, 12], waiting_qubits=[7, 1, 16, 12], concurrent_sub_circuits="AC")
    qc.define_sub_circuit("A", [18, 11, 10, 9])
    qc.define_sub_circuit("B", [16, 8, 7, 6])
    qc.define_sub_circuit("C", [14, 5, 4, 3])
    qc.define_sub_circuit("D", [12, 2, 1, 0], concurrent_sub_circuits=["A", "B", "C"])

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
        # Return success (even parity of measurement outcome). If False (uneven), X-gate must be drawn at second single
        # dot
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
            qc.correct_for_failed_ghz_check({"AC": ghz_success_1, "BD": ghz_success_2})
            ghz_success = False
            continue

        pbar.update(20) if pbar is not None else None

    # Step 9 from Table D.1 (Thesis Naomi Nickerson)
    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    qc.start_sub_circuit("B")
    qc.apply_gate(operation, cqubit=8, tqubit=16)
    qc.measure(8)

    qc.start_sub_circuit("A")
    qc.apply_gate(operation, cqubit=11, tqubit=18)
    qc.measure(11)

    qc.start_sub_circuit("D")
    qc.apply_gate(operation, cqubit=2, tqubit=12)
    qc.measure(2)

    qc.start_sub_circuit("C")
    qc.apply_gate(operation, cqubit=5, tqubit=14)
    qc.measure(5)

    qc.end_current_sub_circuit(total=True)

    end_circuit = time.time()

    if pbar is not None:
        pbar.update(10)

    if draw_circuit:
        qc.draw_circuit(no_color=not color, color_nodes=True)

    start_superoperator = time.time()
    if save_latex_pdf:
        qc.draw_circuit_latex()
    stab_rep = "Z" if operation == CZ_gate else "X"
    _, dataframe = qc.get_superoperator([18, 16, 14, 12], stab_rep, no_color=(not color), stabilizer_protocol=True,
                                        print_to_console=to_console, use_exact_path=True)
    end_superoperator = time.time()

    if pbar is not None:
        pbar.update(10)

    qc.append_print_lines("\nTotal duration of the circuit is {} seconds".format(qc.total_duration))
    qc.append_print_lines("\nCircuit simulation took {} seconds".format(end_circuit - start))
    qc.append_print_lines("\nCalculating the superoperator took {} seconds".format(end_superoperator -
                                                                                   start_superoperator))
    qc.append_print_lines("\nTotal time is {}\n".format(time.time() - start))

    return (dataframe, qc.cut_off_time_reached), qc.print_lines


def stringent(*, operation, pg, pm, pm_1, pn, color, decoherence, bell_pair_creation_success, probabilistic, lkt_1q,
              bell_pair_creation_duration, measurement_duration, pulse_duration, lkt_2q, save_latex_pdf, pbar,
              draw_circuit, to_console):
    start = time.time()
    qc = QuantumCircuit(20, 2, noise=True, basis_transformation_noise=False, pg=pg, pm=pm, pn=pn, pm_1=pm_1,
                        network_noise_type=1, thread_safe_printing=True, probabilistic=probabilistic,
                        decoherence=decoherence, p_bell_success=bell_pair_creation_success, T1_lde=2, T2_lde=2,
                        measurement_duration=measurement_duration, bell_creation_duration=bell_pair_creation_duration,
                        pulse_duration=pulse_duration, single_qubit_gate_lookup=lkt_1q, two_qubit_gate_lookup=lkt_2q,
                        T1_idle=(5*60), T2_idle=10, T1_idle_electron=100, T2_idle_electron=1)

    qc.define_node("A", qubits=[18, 11, 10, 9], electron_qubits=11, data_qubits=18)
    qc.define_node("B", qubits=[16, 8, 7, 6], electron_qubits=8, data_qubits=16)
    qc.define_node("C", qubits=[14, 5, 4, 3], electron_qubits=5, data_qubits=14)
    qc.define_node("D", qubits=[12, 2, 1, 0], electron_qubits=2, data_qubits=12)

    qc.define_sub_circuit("AB", [11, 10, 9, 8, 7, 6, 18, 16], waiting_qubits=[10, 7, 18, 16])
    qc.define_sub_circuit("CD", [5, 4, 3, 2, 1, 0, 14, 12], waiting_qubits=[4, 1, 14, 12], concurrent_sub_circuits="AB")
    qc.define_sub_circuit("AC", [11, 5, 10, 9, 4, 3, 18, 14], waiting_qubits=[10, 4, 18, 14])
    qc.define_sub_circuit("BD", [8, 2, 7, 6, 1, 0, 16, 12], waiting_qubits=[7, 1, 16, 12], concurrent_sub_circuits="AC")
    qc.define_sub_circuit("A", [18, 11, 10, 9])
    qc.define_sub_circuit("B", [16, 8, 7, 6])
    qc.define_sub_circuit("C", [14, 5, 4, 3])
    qc.define_sub_circuit("D", [12, 2, 1, 0], concurrent_sub_circuits=["A", "B", "C"])

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
        # Return success (even parity of measurement outcome). If False (uneven), X-gate must be drawn at second single
        # dot
        success_1 = qc.double_dot(CZ_gate, 10, 4, parity_check=False)
        qc.start_sub_circuit("BD")
        ghz_success = qc.double_dot(CZ_gate, 7, 1, draw_X_gate=not success_1, retry=False)
        if not ghz_success:
            continue

        pbar.update(20) if pbar is not None else None

        # Step 12-14 from Table D.2 (Thesis Naomi Nickerson)
        qc.start_sub_circuit("AC", forced_level=True)
        ghz_success_1 = qc.double_dot(CZ_gate, 10, 4, retry=False)
        qc.start_sub_circuit("BD")
        ghz_success_2 = qc.double_dot(CZ_gate, 7, 1, retry=False)
        if any([not ghz_success_1, not ghz_success_2]):
            qc.correct_for_failed_ghz_check({"AC": ghz_success_1, "BD": ghz_success_2})
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

    end_circuit = time.time()

    pbar.update(10) if pbar is not None else None

    if draw_circuit:
        qc.draw_circuit(no_color=not color, color_nodes=True)

    if save_latex_pdf:
        qc.draw_circuit_latex()

    stab_rep = "Z" if operation == CZ_gate else "X"
    start_superoperator = time.time()
    _, dataframe = qc.get_superoperator([18, 16, 14, 12], stab_rep, no_color=(not color), stabilizer_protocol=True,
                                        print_to_console=to_console)
    end_superoperator = time.time()

    pbar.update(10) if pbar is not None else None

    qc.append_print_lines("\nTotal duration of the circuit is {} seconds".format(qc.total_duration))
    qc.append_print_lines("\nCircuit simulation took {} seconds".format(end_circuit - start))
    qc.append_print_lines("\nCalculating the superoperator took {} seconds".format(end_superoperator -
                                                                                   start_superoperator))
    qc.append_print_lines("\nTotal time is {}\n".format(time.time() - start))

    return (dataframe, qc.cut_off_time_reached), qc.print_lines


def expedient_swap(*, operation, pg, pm, pm_1, pn, color, decoherence, bell_pair_creation_success,
                   bell_pair_creation_duration, measurement_duration, pulse_duration, probabilistic, lkt_1q,
                   lkt_2q, save_latex_pdf, pbar, draw_circuit, to_console):
    start = time.time()
    qc = QuantumCircuit(20, 2, noise=True, basis_transformation_noise=False, pg=pg, pm=pm, pm_1=pm_1, pn=pn,
                        network_noise_type=1, thread_safe_printing=True, probabilistic=probabilistic,
                        decoherence=decoherence, p_bell_success=bell_pair_creation_success, T1_lde=2, T2_lde=2,
                        measurement_duration=measurement_duration, bell_creation_duration=bell_pair_creation_duration,
                        pulse_duration=pulse_duration, single_qubit_gate_lookup=lkt_1q, two_qubit_gate_lookup=lkt_2q,
                        T1_idle=(5*60), T2_idle=10, T1_idle_electron=100, T2_idle_electron=1)

    qc.define_node("A", qubits=[18, 11, 10, 9], electron_qubits=9)
    qc.define_node("B", qubits=[16, 8, 7, 6], electron_qubits=6)
    qc.define_node("C", qubits=[14, 5, 4, 3], electron_qubits=3)
    qc.define_node("D", qubits=[12, 2, 1, 0], electron_qubits=0)

    qc.define_sub_circuit("AB", [11, 10, 9, 8, 7, 6, 18, 16], waiting_qubits=[10, 7, 18, 16])
    qc.define_sub_circuit("CD", [5, 4, 3, 2, 1, 0, 14, 12], waiting_qubits=[4, 1, 14, 12], concurrent_sub_circuits="AB")
    qc.define_sub_circuit("AC", [11, 5, 10, 9, 4, 3, 18, 14], waiting_qubits=[10, 4, 18, 14])
    qc.define_sub_circuit("BD", [8, 2, 7, 6, 1, 0, 16, 12], waiting_qubits=[7, 1, 16, 12], concurrent_sub_circuits="AC")
    qc.define_sub_circuit("A", [18, 11, 10, 9])
    qc.define_sub_circuit("B", [16, 8, 7, 6])
    qc.define_sub_circuit("C", [14, 5, 4, 3])
    qc.define_sub_circuit("D", [12, 2, 1, 0], concurrent_sub_circuits=["A", "B", "C"])

    qc.start_sub_circuit("AB")
    qc.create_bell_pair(9, 6)
    qc.SWAP(9, 10, efficient=True)
    qc.SWAP(6, 7, efficient=True)
    qc.double_selection_swap(CZ_gate, 9, 6)
    qc.double_selection_swap(CNOT_gate, 9, 6)

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit("CD")
    qc.create_bell_pair(3, 0)
    qc.SWAP(3, 4, efficient=True)
    qc.SWAP(0, 1, efficient=True)
    qc.double_selection_swap(CZ_gate, 3, 0)
    qc.double_selection_swap(CNOT_gate, 3, 0)

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit('AC')
    success_1 = qc.single_dot_swap(CZ_gate, 9, 3, parity_check=False)
    success_2 = qc.single_dot_swap(CZ_gate, 9, 3, parity_check=False)

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit('BD')
    qc.single_dot_swap(CZ_gate, 6, 0, draw_X_gate=not success_1)
    qc.single_dot_swap(CZ_gate, 6, 0, draw_X_gate=not success_2)

    if pbar is not None:
        pbar.update(20)

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

    end_circuit = time.time()

    if pbar is not None:
        pbar.update(10)

    if draw_circuit:
        qc.draw_circuit(no_color=not color, color_nodes=True)

    start_superoperator = time.time()
    if save_latex_pdf:
        qc.draw_circuit_latex()
    stab_rep = "Z" if operation == CZ_gate else "X"
    _, dataframe = qc.get_superoperator([18, 16, 14, 12], stab_rep, no_color=(not color), stabilizer_protocol=True,
                                        print_to_console=to_console)
    end_superoperator = time.time()

    if pbar is not None:
        pbar.update(10)

    qc.append_print_lines("\nTotal duration of the circuit is {} seconds".format(qc.total_duration))
    qc.append_print_lines("\nCircuit simulation took {} seconds".format(end_circuit - start))
    qc.append_print_lines("\nCalculating the superoperator took {} seconds".format(end_superoperator -
                                                                                   start_superoperator))
    qc.append_print_lines("\nTotal time is {}\n".format(time.time() - start))

    return (dataframe, qc.cut_off_time_reached), qc.print_lines


def stringent_swap(*, operation, pg, pm, pm_1, pn, color, decoherence, bell_pair_creation_success,
                   bell_pair_creation_duration, measurement_duration, pulse_duration, probabilistic, lkt_1q,
                   lkt_2q, save_latex_pdf, pbar, draw_circuit, to_console):
    start = time.time()
    qc = QuantumCircuit(20, 2, noise=True, basis_transformation_noise=False, pg=pg, pm=pm, pn=pn, pm_1=pm_1,
                        network_noise_type=1, thread_safe_printing=True, probabilistic=probabilistic,
                        decoherence=decoherence, p_bell_success=bell_pair_creation_success,
                        measurement_duration=measurement_duration, bell_creation_duration=bell_pair_creation_duration,
                        pulse_duration=pulse_duration, single_qubit_gate_lookup=lkt_1q, two_qubit_gate_lookup=lkt_2q,
                        T1_idle=(5*60), T2_idle=10, T1_idle_electron=100, T2_idle_electron=1, T1_lde=2, T2_lde=2)

    qc.define_node("A", qubits=[18, 11, 10, 9], electron_qubits=9)
    qc.define_node("B", qubits=[16, 8, 7, 6], electron_qubits=6)
    qc.define_node("C", qubits=[14, 5, 4, 3], electron_qubits=3)
    qc.define_node("D", qubits=[12, 2, 1, 0], electron_qubits=0)

    qc.define_sub_circuit("AB", [11, 10, 9, 8, 7, 6, 18, 16], waiting_qubits=[10, 7, 18, 16])
    qc.define_sub_circuit("CD", [5, 4, 3, 2, 1, 0, 14, 12], waiting_qubits=[4, 1, 14, 12], concurrent_sub_circuits="AB")
    qc.define_sub_circuit("AC", [11, 5, 10, 9, 4, 3, 18, 14], waiting_qubits=[10, 4, 18, 14])
    qc.define_sub_circuit("BD", [8, 2, 7, 6, 1, 0, 16, 12], waiting_qubits=[7, 1, 16, 12], concurrent_sub_circuits="AC")
    qc.define_sub_circuit("A", [18, 11, 10, 9])
    qc.define_sub_circuit("B", [16, 8, 7, 6])
    qc.define_sub_circuit("C", [14, 5, 4, 3])
    qc.define_sub_circuit("D", [12, 2, 1, 0], concurrent_sub_circuits=["A", "B", "C"])

    qc.start_sub_circuit("AB")
    qc.create_bell_pair(9, 6)
    qc.SWAP(9, 10, efficient=True)
    qc.SWAP(6, 7, efficient=True)
    qc.double_selection_swap(CZ_gate, 9, 6)
    qc.double_selection_swap(CNOT_gate, 9, 6)
    qc.double_dot_swap(CZ_gate, 9, 6)
    qc.double_dot_swap(CNOT_gate, 9, 6)

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit("CD")
    qc.create_bell_pair(3, 0)
    qc.SWAP(3, 4, efficient=True)
    qc.SWAP(0, 1, efficient=True)
    qc.double_selection_swap(CZ_gate, 3, 0)
    qc.double_selection_swap(CNOT_gate, 3, 0)
    qc.double_dot_swap(CZ_gate, 3, 0)
    qc.double_dot_swap(CNOT_gate, 3, 0)

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit("AC")
    success_1 = qc.double_dot_swap(CZ_gate, 9, 3, parity_check=False)
    success_2 = qc.double_dot_swap(CZ_gate, 9, 3, parity_check=False)

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit("BD")
    qc.double_dot_swap(CZ_gate, 6, 0, draw_X_gate=not success_1)
    qc.double_dot_swap(CZ_gate, 6, 0, draw_X_gate=not success_2)

    if pbar is not None:
        pbar.update(20)

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

    end_circuit = time.time()

    if pbar is not None:
        pbar.update(10)

    if draw_circuit:
        qc.draw_circuit(no_color=not color, color_nodes=True)

    if save_latex_pdf:
        qc.draw_circuit_latex()

    stab_rep = "Z" if operation == CZ_gate else "X"
    start_superoperator = time.time()
    _, dataframe = qc.get_superoperator([18, 16, 14, 12], stab_rep, no_color=(not color), stabilizer_protocol=True,
                                        print_to_console=to_console)
    end_superoperator = time.time()

    if pbar is not None:
        pbar.update(10)

    qc.append_print_lines("\nTotal duration of the circuit is {} seconds".format(qc.total_duration))
    qc.append_print_lines("\nCircuit simulation took {} seconds".format(end_circuit - start))
    qc.append_print_lines("\nCalculating the superoperator took {} seconds".format(end_superoperator -
                                                                                   start_superoperator))
    qc.append_print_lines("\nTotal time is {}\n".format(time.time() - start))

    return (dataframe, qc.cut_off_time_reached), qc.print_lines
