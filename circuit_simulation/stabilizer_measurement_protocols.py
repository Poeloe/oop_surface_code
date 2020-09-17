import os
import sys
import argparse
sys.path.insert(1, os.path.abspath(os.getcwd()))
from circuit_simulation.circuit_simulator import *
from multiprocessing import Pool
import time
from tqdm import tqdm
import pickle
from pprint import pprint


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


def expedient(operation, pg, pm, pm_1, pn, color, p_dec, p_bell, bell_dur, meas_dur, time_step, prb, lkt_1q, lkt_2q,
              save_latex_pdf, save_csv, csv_file_name, pbar, draw, to_console):
    start = time.time()
    qc = QuantumCircuit(20, 2, noise=True, basis_transformation_noise=False, pg=pg, pm=pm, pm_1=pm_1, pn=pn,
                        network_noise_type=1, thread_safe_printing=True, probabilistic=prb, decoherence=False,
                        p_bell_success=p_bell, measurement_duration=meas_dur, bell_creation_duration=bell_dur,
                        time_step=time_step, single_qubit_gate_lookup=lkt_1q, two_qubit_gate_lookup=lkt_2q,
                        T1_idle=(5*60), T2_idle=10, T1_idle_electron=100, T2_idle_electron=1, T1_lde=2, T2_lde=2)

    qc.define_node("A", qubits=[18, 11, 10, 9], electron_qubits=11)
    qc.define_node("B", qubits=[16, 8, 7, 6], electron_qubits=8)
    qc.define_node("C", qubits=[14, 5, 4, 3], electron_qubits=5)
    qc.define_node("D", qubits=[12, 2, 1, 0], electron_qubits=2)

    qc.start_sub_circuit("AB", [11, 10, 9, 8, 7, 6, 18, 16], waiting_qubits=[11, 8, 18, 16])
    qc.create_bell_pair(11, 8)
    qc.double_selection(CZ_gate, 10, 7)
    qc.double_selection(CNOT_gate, 10, 7)

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit("CD", [5, 4, 3, 2, 1, 0, 14, 12], waiting_qubits=[5, 2, 14, 12])
    qc.create_bell_pair(5, 2)
    qc.double_selection(CZ_gate, 4, 1)
    qc.double_selection(CNOT_gate, 4, 1)

    qc.apply_decoherence_to_fastest_sub_circuit("AB", "CD")

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit("AC", [11, 5, 10, 9, 4, 3, 18, 14], waiting_qubits=[11, 5, 18, 14])
    qc.single_dot(CZ_gate, 10, 4)
    qc.single_dot(CZ_gate, 10, 4)

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit("BD", [8, 2, 7, 6, 1, 0, 16, 12], waiting_qubits=[8, 2, 16, 12])
    qc.single_dot(CZ_gate, 7, 1)
    qc.single_dot(CZ_gate, 7, 1)

    qc.apply_decoherence_to_fastest_sub_circuit("AC", "BD")

    if pbar is not None:
        pbar.update(20)

    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    qc.start_sub_circuit("B", [8, 16])
    qc.apply_2_qubit_gate(operation, 8, 16)
    qc.measure(8)

    qc.start_sub_circuit("A", [11, 18])
    qc.apply_2_qubit_gate(operation, 11, 18)
    qc.measure(11)

    qc.start_sub_circuit("D", [2, 12])
    qc.apply_2_qubit_gate(operation, 2, 12)
    qc.measure(2)

    qc.start_sub_circuit("C", [5, 14])
    qc.apply_2_qubit_gate(operation, 5, 14)
    qc.measure(5)

    qc.end_current_sub_circuit()

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


def stringent(operation, pg, pm, pm_1, pn, color, p_dec, p_bell, bell_dur, meas_dur, time_step, prb, lkt_1q, lkt_2q,
              save_latex_pdf, save_csv, csv_file_name, pbar, draw, to_console):
    start = time.time()
    qc = QuantumCircuit(20, 2, noise=True, basis_transformation_noise=False, pg=pg, pm=pm, pn=pn, pm_1=pm_1,
                        network_noise_type=1, thread_safe_printing=True, probabilistic=prb, decoherence=False,
                        p_bell_success=p_bell, measurement_duration=meas_dur, bell_creation_duration=bell_dur,
                        time_step=time_step, single_qubit_gate_lookup=lkt_1q, two_qubit_gate_lookup=lkt_2q,
                        T1_idle=(5*60), T2_idle=10, T1_idle_electron=100, T2_idle_electron=1, T1_lde=2, T2_lde=2)

    qc.define_node("A", qubits=[18, 11, 10, 9], electron_qubits=11)
    qc.define_node("B", qubits=[16, 8, 7, 6], electron_qubits=8)
    qc.define_node("C", qubits=[14, 5, 4, 3], electron_qubits=5)
    qc.define_node("D", qubits=[12, 2, 1, 0], electron_qubits=2)

    qc.start_sub_circuit("AB", [11, 10, 9, 8, 7, 6, 18, 16], waiting_qubits=[11, 8, 18, 16])

    qc.create_bell_pair(11, 8)
    qc.double_selection(CZ_gate, 10, 7)
    qc.double_selection(CNOT_gate, 10, 7)
    qc.double_dot(CZ_gate, 10, 7)
    qc.double_dot(CNOT_gate, 10, 7)

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit("CD", [5, 4, 3, 2, 1, 0, 14, 12], waiting_qubits=[5, 2, 14, 12])

    qc.create_bell_pair(5, 2)
    qc.double_selection(CZ_gate, 4, 1)
    qc.double_selection(CNOT_gate, 4, 1)
    qc.double_dot(CZ_gate, 4, 1)
    qc.double_dot(CNOT_gate, 4, 1)

    qc.apply_decoherence_to_fastest_sub_circuit("AB", "CD")

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit("AC", [11, 5, 10, 9, 4, 3, 18, 14], waiting_qubits=[11, 5, 18, 14])
    qc.double_dot(CZ_gate, 10, 4)
    qc.double_dot(CZ_gate, 10, 4)

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit("BD", [8, 2, 7, 6, 1, 0, 16, 12], waiting_qubits=[8, 2, 16, 12])
    qc.double_dot(CZ_gate, 7, 1)
    qc.double_dot(CZ_gate, 7, 1)

    qc.apply_decoherence_to_fastest_sub_circuit("AC", "BD")

    if pbar is not None:
        pbar.update(20)

    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    qc.start_sub_circuit("B", [8, 16])
    qc.apply_2_qubit_gate(operation, 8, 16)
    qc.measure(8, probabilistic=False)

    qc.start_sub_circuit("A", [11, 18])
    qc.apply_2_qubit_gate(operation, 11, 18)
    qc.measure(11, probabilistic=False)

    qc.start_sub_circuit("D", [2, 12])
    qc.apply_2_qubit_gate(operation, 2, 12)
    qc.measure(2, probabilistic=False)

    qc.start_sub_circuit("C", [5, 14])
    qc.apply_2_qubit_gate(operation, 5, 14)
    qc.measure(5, probabilistic=False)

    qc.end_current_sub_circuit()

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


def expedient_swap(operation, pg, pm, pm_1, pn, color, p_dec, p_bell, bell_dur, meas_dur, time_step, prb, lkt_1q,
                   lkt_2q, save_latex_pdf, save_csv, csv_file_name, pbar, draw, to_console):
    start = time.time()
    qc = QuantumCircuit(20, 2, noise=True, basis_transformation_noise=False, pg=pg, pm=pm, pm_1=pm_1, pn=pn,
                        network_noise_type=1, thread_safe_printing=True, probabilistic=prb, decoherence=False,
                        p_bell_success=p_bell, measurement_duration=meas_dur, bell_creation_duration=bell_dur,
                        time_step=time_step, single_qubit_gate_lookup=lkt_1q, two_qubit_gate_lookup=lkt_2q,
                        T1_idle=(5*60), T2_idle=10, T1_idle_electron=100, T2_idle_electron=1, T1_lde=2, T2_lde=2)

    qc.define_node("A", qubits=[18, 11, 10, 9], electron_qubits=9)
    qc.define_node("B", qubits=[16, 8, 7, 6], electron_qubits=6)
    qc.define_node("C", qubits=[14, 5, 4, 3], electron_qubits=3)
    qc.define_node("D", qubits=[12, 2, 1, 0], electron_qubits=0)

    qc.start_sub_circuit("AB", [11, 10, 9, 8, 7, 6, 18, 16], waiting_qubits=[10, 7, 18, 16])
    qc.create_bell_pair(9, 6)
    qc.SWAP(9, 10, efficient=True)
    qc.SWAP(6, 7, efficient=True)
    qc.double_selection_swap(CZ_gate, 9, 6)
    qc.double_selection_swap(CNOT_gate, 9, 6)

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit("CD", [5, 4, 3, 2, 1, 0, 14, 12], waiting_qubits=[4, 1, 14, 12])
    qc.create_bell_pair(3, 0)
    qc.SWAP(3, 4, efficient=True)
    qc.SWAP(0, 1, efficient=True)
    qc.double_selection_swap(CZ_gate, 3, 0)
    qc.double_selection_swap(CNOT_gate, 3, 0)

    qc.apply_decoherence_to_fastest_sub_circuit("AB", "CD")

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit("AC", [11, 5, 10, 9, 4, 3, 18, 14], waiting_qubits=[10, 4, 18, 14])
    qc.single_dot_swap(CZ_gate, 9, 3)
    qc.single_dot_swap(CZ_gate, 9, 3)

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit("BD", [8, 2, 7, 6, 1, 0, 16, 12], waiting_qubits=[7, 1, 16, 12])
    qc.single_dot_swap(CZ_gate, 6, 0)
    qc.single_dot_swap(CZ_gate, 6, 0)

    qc.apply_decoherence_to_fastest_sub_circuit("AC", "BD")

    if pbar is not None:
        pbar.update(20)

    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    qc.start_sub_circuit("B", [6, 16])
    qc.SWAP(6, 7, efficient=True)
    qc.apply_2_qubit_gate(operation, 6, 16)
    qc.measure(6, probabilistic=False)

    qc.start_sub_circuit("A", [9, 18])
    qc.SWAP(9, 10, efficient=True)
    qc.apply_2_qubit_gate(operation, 9, 18)
    qc.measure(9, probabilistic=False)

    qc.start_sub_circuit("D", [0, 12])
    qc.SWAP(0, 1, efficient=True)
    qc.apply_2_qubit_gate(operation, 0, 12)
    qc.measure(0, probabilistic=False)

    qc.start_sub_circuit("C", [3, 14])
    qc.SWAP(3, 4, efficient=True)
    qc.apply_2_qubit_gate(operation, 3, 14)
    qc.measure(3, probabilistic=False)

    qc.end_current_sub_circuit()

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


def stringent_swap(operation, pg, pm, pm_1, pn, color, p_dec, p_bell, bell_dur, meas_dur, time_step, prb, lkt_1q,
                   lkt_2q, save_latex_pdf, save_csv, csv_file_name, pbar, draw, to_console):
    start = time.time()
    qc = QuantumCircuit(20, 2, noise=True, basis_transformation_noise=False, pg=pg, pm=pm, pn=pn, pm_1=pm_1,
                        network_noise_type=1, thread_safe_printing=True, probabilistic=prb, decoherence=False,
                        p_bell_success=p_bell, measurement_duration=meas_dur, bell_creation_duration=bell_dur,
                        time_step=time_step, single_qubit_gate_lookup=lkt_1q, two_qubit_gate_lookup=lkt_2q,
                        T1_idle=(5*60), T2_idle=10, T1_idle_electron=100, T2_idle_electron=1, T1_lde=2, T2_lde=2)

    qc.define_node("A", qubits=[18, 11, 10, 9], electron_qubits=9)
    qc.define_node("B", qubits=[16, 8, 7, 6], electron_qubits=6)
    qc.define_node("C", qubits=[14, 5, 4, 3], electron_qubits=3)
    qc.define_node("D", qubits=[12, 2, 1, 0], electron_qubits=0)

    qc.start_sub_circuit("AB", [11, 10, 9, 8, 7, 6, 18, 16], waiting_qubits=[10, 7, 18, 16])

    qc.create_bell_pair(9, 6)
    qc.SWAP(9, 10, efficient=True)
    qc.SWAP(6, 7, efficient=True)
    qc.double_selection_swap(CZ_gate, 9, 6)
    qc.double_selection_swap(CNOT_gate, 9, 6)
    qc.double_dot_swap(CZ_gate, 9, 6)
    qc.double_dot_swap(CNOT_gate, 9, 6)

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit("CD", [5, 4, 3, 2, 1, 0, 14, 12], waiting_qubits=[4, 1, 14, 12])

    qc.create_bell_pair(3, 0)
    qc.SWAP(3, 4, efficient=True)
    qc.SWAP(0, 1, efficient=True)
    qc.double_selection_swap(CZ_gate, 3, 0)
    qc.double_selection_swap(CNOT_gate, 3, 0)
    qc.double_dot_swap(CZ_gate, 3, 0)
    qc.double_dot_swap(CNOT_gate, 3, 0)

    qc.apply_decoherence_to_fastest_sub_circuit("AB", "CD")

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit("AC", [11, 5, 10, 9, 4, 3, 18, 14], waiting_qubits=[10, 4, 18, 14])
    qc.double_dot_swap(CZ_gate, 9, 3)
    qc.double_dot_swap(CZ_gate, 9, 3)

    if pbar is not None:
        pbar.update(20)

    qc.start_sub_circuit("BD", [8, 2, 7, 6, 1, 0, 16, 12], waiting_qubits=[7, 1, 16, 12])
    qc.double_dot_swap(CZ_gate, 6, 0)
    qc.double_dot_swap(CZ_gate, 6, 0)

    qc.apply_decoherence_to_fastest_sub_circuit("AC", "BD")

    if pbar is not None:
        pbar.update(20)

    # ORDER IS ON PURPOSE: EVERYTIME THE TOP QUBIT IS MEASURED, WHICH DECREASES RUNTIME SIGNIFICANTLY
    qc.start_sub_circuit("B", [6, 16])
    qc.SWAP(6, 7, efficient=True)
    qc.apply_2_qubit_gate(operation, 6, 16)
    qc.measure(6, probabilistic=False)

    qc.start_sub_circuit("A", [9, 18])
    qc.SWAP(9, 10, efficient=True)
    qc.apply_2_qubit_gate(operation, 9, 18)
    qc.measure(9, probabilistic=False)

    qc.start_sub_circuit("D", [0, 12])
    qc.SWAP(0, 1, efficient=True)
    qc.apply_2_qubit_gate(operation, 0, 12)
    qc.measure(0, probabilistic=False)

    qc.start_sub_circuit("C", [3, 14])
    qc.SWAP(3, 4, efficient=True)
    qc.apply_2_qubit_gate(operation, 3, 14)
    qc.measure(3, probabilistic=False)

    qc.end_current_sub_circuit()

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


def compose_parser():
    parser = argparse.ArgumentParser(prog='Stabilizer measurement protocol simulations')
    group = parser.add_mutually_exclusive_group(required=True)

    parser.add_argument('-it',
                        '--iterations',
                        help='Specifies the number of iterations that should be done (use only in combination with '
                             '--prb)',
                        type=int,
                        default=1)
    parser.add_argument('-p',
                        '--protocol',
                        help='Specifies which protocol should be used. - options: {monolithic/expedient/stringent}',
                        nargs="*",
                        choices=['monolithic', 'expedient', 'stringent'],
                        type=str.lower,
                        default='monolithic')
    parser.add_argument('-s',
                        '--stabilizer_type',
                        help='Specifies what the kind of stabilizer should be.',
                        choices=['Z', 'X'],
                        type=str.upper,
                        default='Z')
    parser.add_argument('-p_dec',
                        '--decoherence_probability',
                        help='Specifies the decoherence probability for the protocol.',
                        type=float,
                        default=0.)
    parser.add_argument('-pg',
                        '--gate_error_probability',
                        help='Specifies the amount of gate error present in the system',
                        type=float,
                        nargs="*",
                        default=[0.006])
    group.add_argument('--pm_equals_pg',
                       help='Specify if measurement error equals the gate error. "-pm" will then be disregarded',
                       required=False,
                       action='store_true')
    group.add_argument('-pm',
                       '--measurement_error_probability',
                       help='Specifies the amount of measurement error present in the system',
                       type=float,
                       nargs="*",
                       default=[0.006])
    parser.add_argument('-pm_1',
                        '--measurement_error_probability_one_state',
                        help='The measurement error rate in case an 1-state is supposed to be measured',
                        required=False,
                        type=float,
                        nargs="*",
                        default=None)
    parser.add_argument('-pn',
                        '--network_error_probability',
                        help='Specifies the amount of network error present in the system',
                        type=float,
                        nargs="*",
                        default=[0.0])
    parser.add_argument('-p_bell',
                        '--bell_pair_creation_success',
                        help='Specifies the success probability of the creation of a Bell pair (if probabilistic).',
                        type=float,
                        default=1.0)
    parser.add_argument('-prb',
                        '--probabilistic',
                        help='Specifies if the processes in the protocol are probabilistic.',
                        required=False,
                        action='store_true')
    parser.add_argument('-m_dur',
                        '--measurement_duration',
                        help='Specifies the duration of a measurement operation.',
                        type=float,
                        default=0.)
    parser.add_argument('-b_dur',
                        '--bell_pair_creation_duration',
                        help='Specifies the duration of a measurement operation.',
                        type=float,
                        default=0.)
    parser.add_argument('-ts',
                        '--time_step',
                        help='Specifies the duration of a measurement operation.',
                        type=float,
                        default=1)
    parser.add_argument('-c',
                        '--color',
                        help='Specifies if the console output should display color. Optional',
                        required=False,
                        action='store_true')
    parser.add_argument('-ltsv',
                        '--save_latex_pdf',
                        help='If given, a pdf containing a drawing of the noisy circuit in latex will be saved to the '
                             '`circuit_pdfs` folder. Optional',
                        required=False,
                        action='store_true')
    parser.add_argument('-sv',
                        '--save_csv',
                        help='Specifies if a csv file of the superoperator should be saved. Optional',
                        required=False,
                        action='store_true')
    parser.add_argument('-fn',
                        '--csv_filename',
                        required=False,
                        nargs="*",
                        help='Give the file name of the csv file that will be saved.')
    parser.add_argument("-tr",
                        "--threaded",
                        help="Use when the program should run in multi-threaded mode. Optional",
                        required=False,
                        action="store_true")
    parser.add_argument("--to_console",
                        help="Print the superoperator results to the console.",
                        required=False,
                        action="store_true")
    parser.add_argument("-draw",
                        "--draw_circuit",
                        help="Print a drawing of the circuit to the console",
                        required=False,
                        action="store_true")
    parser.add_argument("--print_run_order",
                        help="When added, the program will only print out the run order for the typed command. This can"
                             "be useful for debugging or file naming purposes",
                        required=False,
                        action="store_true")
    parser.add_argument("-lkt_1q",
                        "--lookup_table_single_qubit_gates",
                        help="Name of a .pkl single-qubit gate lookup file.",
                        required=False,
                        type=str,
                        default=None)
    parser.add_argument("-lkt_2q",
                        "--lookup_table_two_qubit_gates",
                        help="Name of a .pkl two-qubit gate lookup file.",
                        required=False,
                        type=str,
                        default=None)
    parser.add_argument("-swap",
                        "--use_swap_gates",
                        help="A version of the protocol will be run that uses SWAP gates to ensure NV-center realism.",
                        required=False,
                        action="store_true")

    return parser


def main(i, it, protocol, stab_type, color, ltsv, sv, pg, pm, pm_1, pn, p_dec, p_bell, bell_dur, meas_dur, time_step,
         lkt_1q, lkt_2q, prb, fn, print_mode, draw, to_console, swap, pbar=None):

    if i == 0:
        _print_circuit_parameters(**locals())

    if print_mode:
        return []

    gate = CZ_gate if stab_type == "Z" else CNOT_gate

    if protocol == "monolithic":
        return monolithic(gate, pg, pm, pm_1, color, bell_dur, meas_dur, time_step, lkt_1q, lkt_2q, ltsv, sv, fn, pbar,
                          draw, to_console)
    elif protocol == "expedient":
        if swap:
            return expedient_swap(gate, pg, pm, pm_1, pn, color, p_dec, p_bell, bell_dur, meas_dur, time_step, prb,
                                  lkt_1q, lkt_2q, ltsv, sv, fn, pbar, draw, to_console)
        return expedient(gate, pg, pm, pm_1, pn, color, p_dec, p_bell, bell_dur, meas_dur, time_step, prb, lkt_1q,
                         lkt_2q, ltsv, sv, fn, pbar, draw, to_console)
    elif protocol == "stringent":
        if swap:
            return stringent_swap(gate, pg, pm, pm_1, pn, color, p_dec, p_bell, bell_dur, meas_dur, time_step, prb,
                                  lkt_1q, lkt_2q, ltsv, sv, fn, pbar, draw, to_console)
        return stringent(gate, pg, pm, pm_1, pn, color, p_dec, p_bell, bell_dur, meas_dur, time_step, prb, lkt_1q,
                         lkt_2q, ltsv, sv, fn, pbar, draw, to_console)


def _print_circuit_parameters(**kwargs):
    it = kwargs.get('it')
    protocol = kwargs.get('protocol')
    sv = kwargs.get('sv')
    fn = kwargs.get('superoperator_filename')
    pg = kwargs.get('pg')
    pm = kwargs.get('pm')
    pn = kwargs.get('pn')
    stab_type= kwargs.get('stab_type')
    lkt_1q = bool(kwargs.get('lkt_1q'))
    lkt_2q = bool(kwargs.get('lkt_2q'))
    kwargs.pop('pbar')
    kwargs.pop('i')
    kwargs.update(lkt_1q=lkt_1q, lkt_2q=lkt_2q)

    protocol = protocol.lower()
    fn_text = ""
    if sv and fn is not None:
        fn_text = "A CSV file will be saved with the name: {}".format(fn)
    print("\nRunning the {} protocol, with pg={}, pm={}{}, for a {} stabilizer {} time{}. {}\n"
          .format(protocol, pg, pm, (' and pn=' + str(pn) if protocol != 'monolithic' else ""),
                  "plaquette" if stab_type == "Z" else "star", it, "s" if it > 1 else "", fn_text))

    print("All circuit parameters:\n-----------------------\n")
    pprint(kwargs)
    print('\n-----------------------\n')


if __name__ == "__main__":
    parser = compose_parser()

    args = vars(parser.parse_args())
    it = args.pop('iterations')
    protocols = args.pop('protocol')
    stab_type = args.pop('stabilizer_type')
    color = args.pop('color')
    p_dec = args.pop('decoherence_probability')
    time_step = args.pop('time_step')
    meas_errors = args.pop('measurement_error_probability')
    meas_1_errors = args.pop('measurement_error_probability_one_state')
    meas_eq_gate = args.pop('pm_equals_pg')
    meas_dur = args.pop('measurement_duration')
    network_errors = args.pop('network_error_probability')
    p_bell = args.pop('bell_pair_creation_success')
    bell_dur = args.pop('bell_pair_creation_duration')
    gate_errors = args.pop('gate_error_probability')
    ltsv = args.pop('save_latex_pdf')
    sv = args.pop('save_csv')
    filenames = args.pop('csv_filename')
    threaded = args.pop('threaded')
    print_mode = args.pop('print_run_order')
    prb = args.pop('probabilistic')
    lkt_1q = args.pop('lookup_table_single_qubit_gates')
    lkt_2q = args.pop('lookup_table_two_qubit_gates')
    draw = args.pop('draw_circuit')
    to_console = args.pop('to_console')
    swap = args.pop('use_swap_gates')

    file_dir = os.path.dirname(__file__)
    # THIS IS NOT GENERIC, will error when directories are moved or renamed
    look_up_table_dir = os.path.join(file_dir, 'gates', 'gate_lookup_tables')

    if meas_1_errors is not None and len(meas_1_errors) != len(meas_errors):
        raise ValueError("Amount of values for --pm_1 should equal the amount of values for -pm.")
    elif meas_1_errors is None:
        meas_1_errors = len(meas_errors) * [None]

    if lkt_1q is not None:
        with open(os.path.join(look_up_table_dir, lkt_1q), 'rb') as obj:
            lkt_1q = pickle.load(obj)

    if lkt_2q is not None:
        with open(os.path.join(look_up_table_dir, lkt_2q), "rb") as obj2:
            lkt_2q = pickle.load(obj2)

    pbar = tqdm(total=100)

    if threaded:
        workers = it if 1 < it < 11 else 10
        thread_pool = Pool(workers)
        results = []

    for i in range(it):
        filename_count = 0
        for protocol in protocols:
            for pg in gate_errors:
                if meas_eq_gate:
                    meas_errors = [pg]
                for k, pm in enumerate(meas_errors):
                    pm_1 = meas_1_errors[k]
                    for pn in network_errors:
                        fn = None if (filenames is None or len(filenames) <= filename_count) else \
                            filenames[filename_count]
                        if threaded:
                            results.append(thread_pool.
                                           apply_async(main,
                                                       (i, it, protocol, stab_type, color, ltsv, sv, pg, pm, pm_1, pn,
                                                        p_dec, p_bell, bell_dur, meas_dur, time_step, lkt_1q, lkt_2q,
                                                        prb, fn, print_mode, draw, to_console, swap)))
                        else:
                            print(*main(i, it, protocol, stab_type, color, ltsv, sv, pg, pm, pm_1, pn, p_dec, p_bell,
                                        bell_dur, meas_dur, time_step, lkt_1q, lkt_2q, prb, fn, print_mode, draw,
                                        to_console, swap, pbar))
                            pbar.reset()
                        filename_count += 1

    if threaded:
        print_results = []
        for res in results:
            print_results.extend(res.get())
            pbar.update(100*(1/it))
        print(*print_results)
        thread_pool.close()




