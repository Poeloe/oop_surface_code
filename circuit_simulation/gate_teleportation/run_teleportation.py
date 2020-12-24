import os
import sys
sys.path.insert(1, os.path.abspath(os.getcwd()))
import circuit_simulation.gate_teleportation.teleportation_circuits as tel_circuits
from circuit_simulation.circuit_simulator import QuantumCircuit
from circuit_simulation.basic_operations.basic_operations import *
from circuit_simulation.states.states import *
from circuit_simulation.gate_teleportation.argument_parsing import compose_parser
from circuit_simulation.stabilizer_measurement_protocols.run_protocols import _additional_parsing_of_arguments, \
    _additional_qc_arguments
from itertools import product
import pandas as pd
from copy import copy
from tqdm import tqdm
import math
import multiprocessing
from pprint import pprint
import os


def get_perfect_matrix():
    bell_pair = (1 / math.sqrt(2)) * (ket_0 * ket_0 + (ket_1 * ket_1))
    bell_pair_cnot = (1 / math.sqrt(2)) * (ket_0 * ket_1 + (ket_1 * ket_0))
    maximally_entangled = (1 / math.sqrt(2)) * ((ket_0 * ket_0) * bell_pair + ((ket_1 * ket_1) * bell_pair_cnot))
    perfect_cnot = CT(maximally_entangled)

    return perfect_cnot


def get_average_fidelity(matrices):
    avg_matrix = sum(matrices) / len(matrices)
    perfect_matrix = get_perfect_matrix()
    entanglement_fidelity = fidelity(perfect_matrix, avg_matrix)
    d = perfect_matrix.shape[0]

    return (d * entanglement_fidelity + 1) / (d + 1)


def create_data_frame(data_frame, **kwargs):

    pop_list = ['iterations', 'save_latex_pdf', 'color', 'draw_circuit', 'pb', 'two_qubit_gate_lookup',
                'single_qubit_gate_lookup', 'thread_safe_printing']
    index_columns = copy(kwargs)
    [index_columns.pop(item) for item in pop_list]

    if data_frame is not None:
        return data_frame, index_columns

    index = pd.MultiIndex.from_product([[item] for item in index_columns.values()], names=list(index_columns.keys()))
    data_frame = pd.DataFrame(index=index)
    data_frame['avg_fidelity'] = 0
    data_frame['iterations'] = 0

    return data_frame, index_columns


def run_series(iterations, gate, use_swap_gates, draw_circuit, color, pb, save_latex_pdf, **kwargs):
    pbar = tqdm(total=iterations, position=1) if pb else None
    qc = QuantumCircuit(6, 4, **kwargs)
    gate = gate if not use_swap_gates else gate + '_swap'
    total_print_lines = []
    matrices = []
    for i in range(iterations):
        pbar.update(1) if pb else None
        noisy_matrix, print_lines = run_gate_teleportation(qc, gate, draw_circuit, color, **kwargs)
        total_print_lines.extend(print_lines)
        matrices.append(noisy_matrix)

    return matrices, total_print_lines


def run_threaded(iterations, **kwargs):
    threads = multiprocessing.cpu_count() if iterations > multiprocessing.cpu_count() else iterations
    pool = multiprocessing.Pool(threads)
    iterations_thread = iterations // threads

    results = []
    for _ in range(threads):
        results.append(pool.apply_async(run_series, args=[iterations_thread], kwds=kwargs))

    noisy_matrices = []
    print_lines = []
    for result in results:
        noisy_matrices_run, print_lines_run = result.get()
        noisy_matrices.extend(noisy_matrices_run)
        print_lines.extend(print_lines_run)
    pool.close()

    return noisy_matrices, print_lines


def run_gate_teleportation(qc: QuantumCircuit, gate, draw_circuit, color, **kwargs):
    teleportation_circuit = getattr(tel_circuits, gate)
    noisy_matrix = teleportation_circuit(qc)

    if draw_circuit:
        qc.draw_circuit(no_color=not color, color_nodes=True)

    print_lines = qc.print_lines
    qc.reset()

    return noisy_matrix, print_lines


def main(data_frame, kwargs, print_lines_total, threaded):
    data_frame, index_columns = create_data_frame(data_frame, **kwargs)
    if threaded:
        noisy_matrices, print_lines = run_threaded(**kwargs)
    else:
        noisy_matrices, print_lines = run_series(**kwargs)

    print_lines_total.extend(print_lines)
    data_frame.loc[tuple(index_columns.values()), :] = 0
    data_frame.loc[tuple(index_columns.values()), 'iterations'] += len(noisy_matrices)
    data_frame.loc[tuple(index_columns.values()), 'avg_fidelity'] = get_average_fidelity(noisy_matrices)

    return data_frame, index_columns


def run_for_arguments(gates, gate_error_probabilities, network_error_probabilities, meas_error_probabilities,
                      meas_error_probabilities_one_state, csv_filename, pm_equals_pg, cp_path,
                      fixed_lde_attempts, threaded, **kwargs):

    meas_1_errors = [None] if meas_error_probabilities_one_state is None else meas_error_probabilities_one_state
    meas_errors = [None] if meas_error_probabilities is None else meas_error_probabilities
    pb = kwargs.pop('no_progress_bar')
    iter_list = [gates, gate_error_probabilities, network_error_probabilities, meas_errors, meas_1_errors,
                 fixed_lde_attempts]
    pbar1 = tqdm(total=len(list(product(*iter_list))), position=0)

    data_frame, index_columns = (None, None)
    print_lines_total = []

    # Loop over command line arguments
    for gate, pg, pn, pm, pm_1, lde in product(*iter_list):
        pbar1.update(1)
        pm = pg if pm is None or pm_equals_pg else pm
        loop_arguments = {
            'gate': gate,
            'pg': pg,
            'pm': pm,
            'pn': pn,
            'pm_1': pm_1,
            'fixed_lde_attempts': lde,
            'pb': pb
        }
        kwargs.update(loop_arguments)
        kwargs = _additional_qc_arguments(**kwargs)
        data_frame, index_columns = main(data_frame, kwargs, print_lines_total, threaded)

    print(*print_lines_total)
    if csv_filename:
        file_path = csv_filename.replace('.csv', '') + ".csv"
        data_frame.to_csv(file_path, sep=';')
    pprint(data_frame)


if __name__ == '__main__':
    parser = compose_parser()

    args = vars(parser.parse_args())
    args = _additional_parsing_of_arguments(args)
    args.pop('gate_duration_file')

    run_for_arguments(**args)
