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
    fidelities = []
    for i in range(iterations):
        pbar.update(1) if pb else None
        fid, print_lines = run_gate_teleportation(qc, gate, draw_circuit, color, **kwargs)
        total_print_lines.extend(print_lines)
        fidelities.append(fid)

    return fidelities, total_print_lines


def run_threaded(iterations, **kwargs):
    threads = multiprocessing.cpu_count() if iterations > multiprocessing.cpu_count() else iterations
    pool = multiprocessing.Pool(threads)
    iterations_thread = iterations // threads

    results = []
    for _ in range(threads):
        results.append(pool.apply_async(run_series, args=[iterations_thread], kwds=kwargs))

    fidelities = []
    print_lines = []
    for result in results:
        fidelities_run, print_lines_run = result.get()
        fidelities.extend(fidelities_run)
        print_lines.extend(print_lines_run)
    pool.close()

    return fidelities, print_lines


def run_gate_teleportation(qc: QuantumCircuit, gate, draw_circuit, color, **kwargs):
    teleportation_circuit = getattr(tel_circuits, gate)
    qubits = teleportation_circuit(qc)
    bell_pair = (1/math.sqrt(2))*(ket_0 * ket_0 + (ket_1 * ket_1))
    bell_pair_cnot = (1 / math.sqrt(2)) * (ket_0 * ket_1 + (ket_1 * ket_0))
    maximally_entangled = (1 / math.sqrt(2)) * ((ket_0 * ket_0) * bell_pair + ((ket_1 * ket_1) * bell_pair_cnot))
    compare_matrix = CT(maximally_entangled)
    fid = qc.get_state_fidelity(qubits=qubits, compare_matrix=compare_matrix)

    if draw_circuit:
        qc.draw_circuit(no_color=not color, color_nodes=True)
        qc.append_print_lines("\nFidelity: {}".format(fid))

    print_lines = qc.print_lines
    qc.reset()

    return fid, print_lines


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
        data_frame, index_columns = create_data_frame(data_frame, **kwargs)
        if threaded:
            fidelities, print_lines = run_threaded(**kwargs)
        else:
            fidelities, print_lines = run_series(**kwargs)

        print_lines_total.extend(print_lines)
        data_frame.loc[tuple(index_columns.values()), :] = 0
        data_frame.loc[tuple(index_columns.values()), 'iterations'] += len(fidelities)
        data_frame.loc[tuple(index_columns.values()), 'avg_fidelity'] = sum(fidelities)/len(fidelities)

    print(*print_lines_total)
    if csv_filename:
        file_path = csv_filename.replace('.csv', '') + ".csv"
        if os.path.exists(file_path):
            ex_dataframe = pd.read_csv(file_path, index_col=list(index_columns.keys()), float_precision='round_trip')
            prev_fids = ex_dataframe['avg_fidelity'] * ex_dataframe['iterations']
            new_fids = data_frame['avg_fidelity'] * data_frame['iterations']
            data_frame['iterations'] = ex_dataframe['iterations'].add(data_frame['iterations'], fill_value=0)
            data_frame['avg_fidelity'] = (new_fids.add(prev_fids, fill_value=0)) / data_frame['iterations']
        data_frame.to_csv(file_path)
    pprint(data_frame)


if __name__ == '__main__':
    parser = compose_parser()

    args = vars(parser.parse_args())
    args = _additional_parsing_of_arguments(args)
    args.pop('gate_duration_file')

    run_for_arguments(**args)
