from circuit_simulation.circuit_simulator import *
import time


def monolithic(operation, pg, pm, pm_1, color, bell_dur, meas_dur, time_step, lkt_1q, lkt_2q,
               save_latex_pdf, save_csv, csv_file_name, pbar, draw, to_console):
    qc = QuantumCircuit(9, 2, noise=True, pg=pg, pm=pm, pm_1=pm_1, basis_transformation_noise=True,
                        thread_safe_printing=True, bell_creation_duration=bell_dur, measurement_duration=meas_dur,
                        time_step=time_step, single_qubit_gate_lookup=lkt_1q, two_qubit_gate_lookup=lkt_2q)
    qc.set_qubit_states({0: ket_p})
    qc.apply_2_qubit_gate(operation, 0, 1)
    qc.apply_2_qubit_gate(operation, 0, 3)
    qc.apply_2_qubit_gate(operation, 0, 5)
    qc.apply_2_qubit_gate(operation, 0, 7)
    qc.measure([0])

    if pbar is not None:
        pbar.update(50)

    if draw:
        qc.draw_circuit(not color)
    if save_latex_pdf:
        qc.draw_circuit_latex()
    stab_rep = "Z" if operation == CZ_gate else "X"
    qc.get_superoperator([1, 3, 5, 7], stab_rep, no_color=(not color), to_csv=save_csv,
                         csv_file_name=csv_file_name, stabilizer_protocol=True, print_to_console=to_console)
    if pbar is not None:
        pbar.update(50)

    return qc._print_lines


def expedient(operation, pg, pm, pm_1, pn, color, dec, p_bell, bell_dur, meas_dur, time_step, prb, lkt_1q, lkt_2q,
              save_latex_pdf, save_csv, csv_file_name, pbar, draw, to_console):
    start = time.time()
    qc = QuantumCircuit(20, 2, noise=True, basis_transformation_noise=False, pg=pg, pm=pm, pm_1=pm_1, pn=pn,
                        network_noise_type=1, thread_safe_printing=True, probabilistic=prb, decoherence=dec,
                        p_bell_success=p_bell, measurement_duration=meas_dur, bell_creation_duration=bell_dur,
                        time_step=time_step, single_qubit_gate_lookup=lkt_1q, two_qubit_gate_lookup=lkt_2q,
                        T1_idle=(5*60), T2_idle=10, T1_idle_electron=100, T2_idle_electron=1, T1_lde=2, T2_lde=2)

    qc.define_node("A", qubits=[18, 11, 10, 9], electron_qubits=11)
    qc.define_node("B", qubits=[16, 8, 7, 6], electron_qubits=8)
    qc.define_node("C", qubits=[14, 5, 4, 3], electron_qubits=5)
    qc.define_node("D", qubits=[12, 2, 1, 0], electron_qubits=2)

    qc.define_sub_circuit("AB", [11, 10, 9, 8, 7, 6, 18, 16], waiting_qubits=[10, 7, 18, 16])
    qc.define_sub_circuit("CD", [5, 4, 3, 2, 1, 0, 14, 12], waiting_qubits=[4, 1, 14, 12], concurrent_sub_circuits="AB")
    qc.define_sub_circuit("AC", [11, 5, 10, 9, 4, 3, 18, 14], waiting_qubits=[10, 4, 18, 14])
    qc.define_sub_circuit("BD", [8, 2, 7, 6, 1, 0, 16, 12], waiting_qubits=[7, 1, 16, 12], concurrent_sub_circuits="AC")
    qc.define_sub_circuit("A", [18, 11, 10, 9])
    qc.define_sub_circuit("B", [16, 8, 7, 6])
    qc.define_sub_circuit("C", [14, 5, 4, 3])
    qc.define_sub_circuit("D", [12, 2, 1, 0], concurrent_sub_circuits=["A", "B", "C"])

    qc.start_sub_circuit("AB")
    qc.create_bell_pair(11, 8)
    qc.double_selection(CZ_gate, 10, 7)
    qc.double_selection(CNOT_gate, 10, 7)

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit("CD")
    qc.create_bell_pair(5, 2)
    qc.double_selection(CZ_gate, 4, 1)
    qc.double_selection(CNOT_gate, 4, 1)

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit("AC")
    qc.single_dot(CZ_gate, 10, 4)
    qc.single_dot(CZ_gate, 10, 4)

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit("BD")
    qc.single_dot(CZ_gate, 7, 1)
    qc.single_dot(CZ_gate, 7, 1)

    if pbar is not None:
        pbar.update(20)

    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    qc.start_sub_circuit("B")
    qc.apply_2_qubit_gate(operation, 8, 16)
    qc.measure(8)

    qc.start_sub_circuit("A")
    qc.apply_2_qubit_gate(operation, 11, 18)
    qc.measure(11)

    qc.start_sub_circuit("D")
    qc.apply_2_qubit_gate(operation, 2, 12)
    qc.measure(2)

    qc.start_sub_circuit("C")
    qc.apply_2_qubit_gate(operation, 5, 14)
    qc.measure(5)

    qc.end_current_sub_circuit(total=True)

    end_circuit = time.time()

    if pbar is not None:
        pbar.update(10)

    if draw:
        qc.draw_circuit(no_color=not color, color_nodes=True)

    start_superoperator = time.time()
    if save_latex_pdf:
        qc.draw_circuit_latex()
    stab_rep = "Z" if operation == CZ_gate else "X"
    qc.get_superoperator([18, 16, 14, 12], stab_rep, no_color=(not color), to_csv=save_csv,
                         csv_file_name=csv_file_name, stabilizer_protocol=True, print_to_console=to_console)
    end_superoperator = time.time()

    if pbar is not None:
        pbar.update(10)

    qc._print_lines.append("\nTotal duration of the circuit is {} seconds".format(qc.total_duration))
    qc._print_lines.append("\nCircuit simulation took {} seconds".format(end_circuit - start))
    qc._print_lines.append("\nCalculating the superoperator took {} seconds".format(end_superoperator -
                                                                                    start_superoperator))
    qc._print_lines.append("\nTotal time is {}\n".format(time.time() - start))

    return qc._print_lines


def stringent(operation, pg, pm, pm_1, pn, color, dec, p_bell, bell_dur, meas_dur, time_step, prb, lkt_1q, lkt_2q,
              save_latex_pdf, save_csv, csv_file_name, pbar, draw, to_console):
    start = time.time()
    qc = QuantumCircuit(20, 2, noise=True, basis_transformation_noise=False, pg=pg, pm=pm, pn=pn, pm_1=pm_1,
                        network_noise_type=1, thread_safe_printing=True, probabilistic=prb, decoherence=dec,
                        p_bell_success=p_bell, measurement_duration=meas_dur, bell_creation_duration=bell_dur,
                        time_step=time_step, single_qubit_gate_lookup=lkt_1q, two_qubit_gate_lookup=lkt_2q,
                        T1_idle=(5*60), T2_idle=10, T1_idle_electron=100, T2_idle_electron=1, T1_lde=2, T2_lde=2)

    qc.define_node("A", qubits=[18, 11, 10, 9], electron_qubits=11)
    qc.define_node("B", qubits=[16, 8, 7, 6], electron_qubits=8)
    qc.define_node("C", qubits=[14, 5, 4, 3], electron_qubits=5)
    qc.define_node("D", qubits=[12, 2, 1, 0], electron_qubits=2)

    qc.define_sub_circuit("AB", [11, 10, 9, 8, 7, 6, 18, 16], waiting_qubits=[10, 7, 18, 16])
    qc.define_sub_circuit("CD", [5, 4, 3, 2, 1, 0, 14, 12], waiting_qubits=[4, 1, 14, 12], concurrent_sub_circuits="AB")
    qc.define_sub_circuit("AC", [11, 5, 10, 9, 4, 3, 18, 14], waiting_qubits=[10, 4, 18, 14])
    qc.define_sub_circuit("BD", [8, 2, 7, 6, 1, 0, 16, 12], waiting_qubits=[7, 1, 16, 12], concurrent_sub_circuits="AC")
    qc.define_sub_circuit("A", [18, 11, 10, 9])
    qc.define_sub_circuit("B", [16, 8, 7, 6])
    qc.define_sub_circuit("C", [14, 5, 4, 3])
    qc.define_sub_circuit("D", [12, 2, 1, 0], concurrent_sub_circuits=["A", "B", "C"])

    qc.start_sub_circuit("AB")

    qc.create_bell_pair(11, 8)
    qc.double_selection(CZ_gate, 10, 7)
    qc.double_selection(CNOT_gate, 10, 7)
    qc.double_dot(CZ_gate, 10, 7)
    qc.double_dot(CNOT_gate, 10, 7)

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit("CD")

    qc.create_bell_pair(5, 2)
    qc.double_selection(CZ_gate, 4, 1)
    qc.double_selection(CNOT_gate, 4, 1)
    qc.double_dot(CZ_gate, 4, 1)
    qc.double_dot(CNOT_gate, 4, 1)

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit("AC")
    qc.double_dot(CZ_gate, 10, 4)
    qc.double_dot(CZ_gate, 10, 4)

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit("BD")
    qc.double_dot(CZ_gate, 7, 1)
    qc.double_dot(CZ_gate, 7, 1)

    if pbar is not None:
        pbar.update(20)

    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    qc.start_sub_circuit("B")
    qc.apply_2_qubit_gate(operation, 8, 16)
    qc.measure(8, probabilistic=False)

    qc.start_sub_circuit("A")
    qc.apply_2_qubit_gate(operation, 11, 18)
    qc.measure(11, probabilistic=False)

    qc.start_sub_circuit("D")
    qc.apply_2_qubit_gate(operation, 2, 12)
    qc.measure(2, probabilistic=False)

    qc.start_sub_circuit("C")
    qc.apply_2_qubit_gate(operation, 5, 14)
    qc.measure(5, probabilistic=False)

    qc.end_current_sub_circuit(total=True)

    end_circuit = time.time()

    if pbar is not None:
        pbar.update(10)

    if draw:
        qc.draw_circuit(no_color=not color)

    if save_latex_pdf:
        qc.draw_circuit_latex()

    stab_rep = "Z" if operation == CZ_gate else "X"
    start_superoperator = time.time()
    qc.get_superoperator([18, 16, 14, 12], stab_rep, no_color=(not color), to_csv=save_csv,
                         csv_file_name=csv_file_name, stabilizer_protocol=True, print_to_console=to_console)
    end_superoperator = time.time()

    if pbar is not None:
        pbar.update(10)

    qc._print_lines.append("\nTotal duration of the circuit is {} seconds".format(qc.total_duration))
    qc._print_lines.append("\nCircuit simulation took {} seconds".format(end_circuit - start))
    qc._print_lines.append("\nCalculating the superoperator took {} seconds".format(end_superoperator -
                                                                                  start_superoperator))
    qc._print_lines.append("\nTotal time is {}\n".format(time.time() - start))

    return qc._print_lines


def expedient_swap(operation, pg, pm, pm_1, pn, color, dec, p_bell, bell_dur, meas_dur, time_step, prb, lkt_1q,
                   lkt_2q, save_latex_pdf, save_csv, csv_file_name, pbar, draw, to_console):
    start = time.time()
    qc = QuantumCircuit(20, 2, noise=True, basis_transformation_noise=False, pg=pg, pm=pm, pm_1=pm_1, pn=pn,
                        network_noise_type=1, thread_safe_printing=True, probabilistic=prb, decoherence=dec,
                        p_bell_success=p_bell, measurement_duration=meas_dur, bell_creation_duration=bell_dur,
                        time_step=time_step, single_qubit_gate_lookup=lkt_1q, two_qubit_gate_lookup=lkt_2q,
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
    qc.single_dot_swap(CZ_gate, 9, 3)
    qc.single_dot_swap(CZ_gate, 9, 3)

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit('BD')
    qc.single_dot_swap(CZ_gate, 6, 0)
    qc.single_dot_swap(CZ_gate, 6, 0)

    if pbar is not None:
        pbar.update(20)

    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    qc.start_sub_circuit("B")
    qc.SWAP(6, 7, efficient=True)
    qc.apply_2_qubit_gate(operation, 6, 16)
    qc.measure(6, probabilistic=False)

    qc.start_sub_circuit("A")
    qc.SWAP(9, 10, efficient=True)
    qc.apply_2_qubit_gate(operation, 9, 18)
    qc.measure(9, probabilistic=False)

    qc.start_sub_circuit("D")
    qc.SWAP(0, 1, efficient=True)
    qc.apply_2_qubit_gate(operation, 0, 12)
    qc.measure(0, probabilistic=False)

    qc.start_sub_circuit("C")
    qc.SWAP(3, 4, efficient=True)
    qc.apply_2_qubit_gate(operation, 3, 14)
    qc.measure(3, probabilistic=False)

    qc.end_current_sub_circuit(total=True)

    end_circuit = time.time()

    if pbar is not None:
        pbar.update(10)

    if draw:
        qc.draw_circuit(no_color=not color, color_nodes=True)

    start_superoperator = time.time()
    if save_latex_pdf:
        qc.draw_circuit_latex()
    stab_rep = "Z" if operation == CZ_gate else "X"
    qc.get_superoperator([18, 16, 14, 12], stab_rep, no_color=(not color), to_csv=save_csv,
                         csv_file_name=csv_file_name, stabilizer_protocol=True, print_to_console=to_console,
                         use_exact_path=True)
    end_superoperator = time.time()

    if pbar is not None:
        pbar.update(10)

    qc._print_lines.append("\nTotal duration of the circuit is {} seconds".format(qc.total_duration))
    qc._print_lines.append("\nCircuit simulation took {} seconds".format(end_circuit - start))
    qc._print_lines.append("\nCalculating the superoperator took {} seconds".format(end_superoperator -
                                                                                    start_superoperator))
    qc._print_lines.append("\nTotal time is {}\n".format(time.time() - start))

    return qc._print_lines


def stringent_swap(operation, pg, pm, pm_1, pn, color, dec, p_bell, bell_dur, meas_dur, time_step, prb, lkt_1q,
                   lkt_2q, save_latex_pdf, save_csv, csv_file_name, pbar, draw, to_console):
    start = time.time()
    qc = QuantumCircuit(20, 2, noise=True, basis_transformation_noise=False, pg=pg, pm=pm, pn=pn, pm_1=pm_1,
                        network_noise_type=1, thread_safe_printing=True, probabilistic=prb, decoherence=dec,
                        p_bell_success=p_bell, measurement_duration=meas_dur, bell_creation_duration=bell_dur,
                        time_step=time_step, single_qubit_gate_lookup=lkt_1q, two_qubit_gate_lookup=lkt_2q,
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
    qc.double_dot_swap(CZ_gate, 9, 3)
    qc.double_dot_swap(CZ_gate, 9, 3)

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit("BD")
    qc.double_dot_swap(CZ_gate, 6, 0)
    qc.double_dot_swap(CZ_gate, 6, 0)

    if pbar is not None:
        pbar.update(20)

    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    qc.start_sub_circuit("B")
    qc.SWAP(6, 7, efficient=True)
    qc.apply_2_qubit_gate(operation, 6, 16)
    qc.measure(6, probabilistic=False)

    qc.start_sub_circuit("A")
    qc.SWAP(9, 10, efficient=True)
    qc.apply_2_qubit_gate(operation, 9, 18)
    qc.measure(9, probabilistic=False)

    qc.start_sub_circuit("D")
    qc.SWAP(0, 1, efficient=True)
    qc.apply_2_qubit_gate(operation, 0, 12)
    qc.measure(0, probabilistic=False)

    qc.start_sub_circuit("C")
    qc.SWAP(3, 4, efficient=True)
    qc.apply_2_qubit_gate(operation, 3, 14)
    qc.measure(3, probabilistic=False)

    qc.end_current_sub_circuit(total=True)

    end_circuit = time.time()

    if pbar is not None:
        pbar.update(10)

    if draw:
        qc.draw_circuit(no_color=not color)

    if save_latex_pdf:
        qc.draw_circuit_latex()

    stab_rep = "Z" if operation == CZ_gate else "X"
    start_superoperator = time.time()
    qc.get_superoperator([18, 16, 14, 12], stab_rep, no_color=(not color), to_csv=save_csv,
                         csv_file_name=csv_file_name, stabilizer_protocol=True, print_to_console=to_console,
                         use_exact_path=True)
    end_superoperator = time.time()

    if pbar is not None:
        pbar.update(10)

    qc._print_lines.append("\nTotal duration of the circuit is {} seconds".format(qc.total_duration))
    qc._print_lines.append("\nCircuit simulation took {} seconds".format(end_circuit - start))
    qc._print_lines.append("\nCalculating the superoperator took {} seconds".format(end_superoperator -
                                                                                  start_superoperator))
    qc._print_lines.append("\nTotal time is {}\n".format(time.time() - start))

    return qc._print_lines


