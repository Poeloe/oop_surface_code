import os
import sys
sys.path.insert(1, os.path.abspath(os.getcwd()))
import circuit_simulation.gate_teleportation.teleportation_circuits as tel_circuits
from circuit_simulation.circuit_simulator import QuantumCircuit
from circuit_simulation.basic_operations.basic_operations import *
from circuit_simulation.states.states import *
from circuit_simulation.gate_teleportation.argument_parsing import compose_parser
from circuit_simulation.stabilizer_measurement_protocols.run_protocols import additional_parsing_of_arguments, \
    _additional_qc_arguments
from itertools import product
import pandas as pd
from copy import copy
from tqdm import tqdm
import math
import multiprocessing
from pprint import pprint
import pickle
from plot_superoperator.analyse_simulation_data import confidence_interval


def get_perfect_matrix():
    bell_pair = (1 / math.sqrt(2)) * (ket_0 * ket_0 + (ket_1 * ket_1))
    bell_pair_cnot = (1 / math.sqrt(2)) * (ket_0 * ket_1 + (ket_1 * ket_0))
    maximally_entangled = (1 / math.sqrt(2)) * ((ket_0 * ket_0) * bell_pair + ((ket_1 * ket_1) * bell_pair_cnot))
    perfect_cnot = CT(maximally_entangled)

    return perfect_cnot


def get_average_fidelity(matrices):
    perfect_matrix = get_perfect_matrix()
    d = math.sqrt(perfect_matrix.shape[0])

    # Error bar data
    entanglement_fidelities = [fidelity_elementwise(perfect_matrix, mat) for mat in matrices]

    avg_matrix = sum(matrices) / len(matrices)
    entanglement_fidelity = fidelity_elementwise(perfect_matrix, avg_matrix)

    return (d * entanglement_fidelity + 1) / (d + 1), entanglement_fidelities


def create_data_frame(data_frame: pd.DataFrame, **kwargs):

    pop_list = ['iterations', 'save_latex_pdf', 'color', 'draw_circuit', 'pb', 'two_qubit_gate_lookup',
                'single_qubit_gate_lookup', 'thread_safe_printing']
    index_columns = copy(kwargs)
    index_columns['pm_1'] = index_columns['pm_1'] if index_columns['pm_1'] is not None else index_columns['pm']
    [index_columns.pop(item) for item in pop_list]

    index = pd.MultiIndex.from_product([[item] for item in index_columns.values()], names=list(index_columns.keys()))
    if data_frame is not None:
        if index.nlevels != data_frame.index.nlevels:
            data_frame = data_frame.set_index(list(index_columns.keys()))
        return data_frame, index_columns

    data_frame = pd.DataFrame(index=index)
    data_frame['avg_fidelity'] = 0
    data_frame['iterations'] = 0
    data_frame['dur_std'] = 0
    data_frame['avg_duration'] = 0

    return data_frame, index_columns


def run_series(iterations, gate, use_swap_gates, draw_circuit, color, pb, save_latex_pdf, **kwargs):
    qc = QuantumCircuit(6, 4, **kwargs)
    gate = gate if not use_swap_gates else gate + '_swap'

    durations = []
    total_print_lines = []
    matrices = []
    for i in range(iterations):
        pb.update(1) if pb else None
        noisy_matrix, print_lines, total_duration = run_gate_teleportation(qc, gate, draw_circuit, color, **kwargs)
        total_print_lines.extend(print_lines)
        matrices.append(noisy_matrix)
        durations.append(total_duration)

    return matrices, total_print_lines, durations


def run_threaded(iterations, **kwargs):
    threads = multiprocessing.cpu_count() if iterations > multiprocessing.cpu_count() else iterations
    pool = multiprocessing.Pool(threads)
    iterations_thread = iterations // threads

    results = []
    for _ in range(threads):
        results.append(pool.apply_async(run_series, args=[iterations_thread], kwds=kwargs))

    noisy_matrices = []
    print_lines = []
    durations = []
    for result in results:
        noisy_matrices_run, print_lines_run, durations_run = result.get()
        noisy_matrices.extend(noisy_matrices_run)
        print_lines.extend(print_lines_run)
        durations.extend(durations_run)
    pool.close()

    return noisy_matrices, print_lines, durations


def run_gate_teleportation(qc: QuantumCircuit, gate, draw_circuit, color, **kwargs):
    teleportation_circuit = getattr(tel_circuits, gate)
    noisy_matrix = teleportation_circuit(qc)

    if draw_circuit:
        qc.draw_circuit(no_color=not color, color_nodes=True)

    print_lines = qc.print_lines
    total_duration = qc.total_duration
    qc.reset()

    return noisy_matrix, print_lines, total_duration


def main(data_frame, kwargs, print_lines_total, threaded, csv_filename, it):
    data_frame, index_columns = create_data_frame(data_frame, **kwargs)
    if threaded:
        noisy_matrices, print_lines, durations = run_threaded(**kwargs)
    else:
        noisy_matrices, print_lines, durations = run_series(**kwargs)

    print_lines_total.extend(print_lines)
    avg_fid, fidelities = get_average_fidelity(noisy_matrices)

    if csv_filename:
        ind = tuple(index_columns.items())
        pkl_data = {ind: {"fidelities": fidelities, "durations": durations}}
        pkl_fn = csv_filename + ".pkl"
        if os.path.exists(pkl_fn):
            pkl_data_old = pickle.load(open(pkl_fn, 'rb'))
            if ind in pkl_data_old:
                [pkl_data_old[ind][key].extend(value) for key, value in pkl_data_old[ind].items()]
            else:
                pkl_data_old.update(pkl_data)
            pkl_data = pkl_data_old
        pickle.dump(pkl_data, open(pkl_fn, 'wb'))

    index = tuple(index_columns.values())
    if index not in data_frame.index:
        data_frame.loc[index, :] = 0

    data_frame.loc[index, 'iterations'] += len(noisy_matrices)
    data_frame.loc[index, 'avg_fidelity'] = avg_fid
    data_frame.loc[index, 'avg_duration'] = np.mean(durations)
    data_frame.loc[index, 'fid_int'] = str(confidence_interval(fidelities))
    data_frame.loc[index, 'dur_std'] = str(confidence_interval(durations))

    return data_frame, index_columns


def run_for_arguments(gates, gate_error_probabilities, network_error_probabilities, meas_error_probabilities,
                      meas_error_probabilities_one_state, csv_filename, pm_equals_pg, cp_path,
                      fixed_lde_attempts, threaded, pulse_duration, **kwargs):

    meas_1_errors = [None] if meas_error_probabilities_one_state is None else meas_error_probabilities_one_state
    meas_errors = [None] if meas_error_probabilities is None else meas_error_probabilities
    pb = kwargs.pop('no_progress_bar')
    iter_list = [gates, gate_error_probabilities, network_error_probabilities, meas_errors, meas_1_errors,
                 fixed_lde_attempts, pulse_duration]
    pbar1 = tqdm(total=len(list(product(*iter_list))), position=0)

    if csv_filename and os.path.exists(csv_filename + ".csv"):
        data_frame = pd.read_csv(csv_filename + ".csv", sep=";", float_precision='round_trip')
    else:
        data_frame, index_columns = (None, None)
    print_lines_total = []

    # Loop over command line arguments
    pbar = tqdm(total=kwargs['iterations'], position=1) if pb else None
    for it, (gate, pg, pn, pm, pm_1, lde, pulse) in enumerate(product(*iter_list)):
        pbar.reset() if pbar is not None else None
        pm = pg if pm is None or pm_equals_pg else pm
        lde = lde if pulse else 0
        loop_arguments = {
            'gate': gate,
            'pg': pg,
            'pm': pm,
            'pn': pn,
            'pm_1': pm_1,
            'fixed_lde_attempts': lde,
            'pb': pbar,
            'pulse_duration': pulse
        }
        kwargs.update(loop_arguments)
        kwargs = _additional_qc_arguments(**kwargs)
        print("\n\nRunning {} iterations with arguments:".format(kwargs['iterations']))
        pprint(loop_arguments)
        data_frame, index_columns = main(data_frame, kwargs, print_lines_total, threaded, csv_filename, it)
        pbar1.update(1)

        if csv_filename:
            data_frame.to_csv(csv_filename + ".csv", sep=';')

    print(*print_lines_total)
    pprint(data_frame)


if __name__ == '__main__':
    parser = compose_parser()

    args = vars(parser.parse_args())
    args = additional_parsing_of_arguments(**args)
    args.pop('gate_duration_file')

    run_for_arguments(**args)
