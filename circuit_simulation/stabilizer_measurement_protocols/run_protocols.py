import os
import sys
sys.path.insert(1, os.path.abspath(os.getcwd()))
from pprint import pprint
from multiprocessing import Pool, cpu_count
import threading
import pickle
import pandas as pd
import circuit_simulation.stabilizer_measurement_protocols.stabilizer_measurement_protocols as stab_protocols
from circuit_simulation.stabilizer_measurement_protocols.argument_parsing import compose_parser
from circuit_simulation.gates.gates import *
import itertools
import time
from copy import copy
import random


def _init_random_seed(timestamp=None, worker=0, iteration=0):
    if timestamp is None:
        timestamp = time.time()
    seed = int("{:.0f}".format(timestamp * 10 ** 7) + str(worker) + str(iteration))
    random.seed(float(seed))
    return seed


def _combine_superoperator_dataframes(dataframe_1, dataframe_2):
    """
        Combines two given superoperator dataframes into one dataframe

        Parameters
        ----------
        dataframe_1 : pd.DataFrame
            Superoperator dataframe to be combined
        dataframe_2 : pd.DataFrame
            Superoperator dataframe to be combined
    """
    if dataframe_1 is None and dataframe_2 is None:
        return None
    if dataframe_1 is None:
        return dataframe_2
    if dataframe_2 is None:
        return dataframe_1

    # First combine the total amount of iterations, such that it can be used later
    written_to_original = dataframe_1.iloc[0, dataframe_1.columns.get_loc("written_to")]
    written_to_new = dataframe_2.iloc[0, dataframe_2.columns.get_loc("written_to")]
    corrected_written_to = written_to_new + written_to_original
    dataframe_1.iloc[0, dataframe_1.columns.get_loc("written_to")] = corrected_written_to

    # Calculate the average probability of the error configurations per stabilizer
    dataframe_2[['p', 's']] = dataframe_2[['p', 's']].mul(written_to_new)
    dataframe_1[['p', 's']] = dataframe_1[['p', 's']].mul(written_to_original)

    dataframe_1[['s', 'p']] = (dataframe_1[['s', 'p']] + dataframe_2[['s', 'p']]) / corrected_written_to

    # Update the average of the other system characteristics
    dataframe_1['total_duration'] = (dataframe_1['total_duration'] + dataframe_2['total_duration'])
    dataframe_1['lde_attempts'] = (dataframe_1['lde_attempts'] + dataframe_2['lde_attempts'])
    dataframe_1['succeeded_lde'] = (dataframe_1['succeeded_lde'] + dataframe_2['succeeded_lde'])

    dataframe_1['avg_lde'] = dataframe_1['lde_attempts'] / corrected_written_to
    dataframe_1['avg_duration'] = dataframe_1['total_duration'] / corrected_written_to
    dataframe_1['avg_lde_to_succeed'] = dataframe_1['lde_attempts'] / dataframe_1['succeeded_lde']

    # Update fidelity
    dataframe_2['ghz_fidelity'] = dataframe_2['ghz_fidelity'].mul(written_to_new)
    dataframe_1['ghz_fidelity'] = dataframe_1['ghz_fidelity'].mul(written_to_original)

    dataframe_1['ghz_fidelity'] = (dataframe_1['ghz_fidelity'] + dataframe_2['ghz_fidelity']) / corrected_written_to

    return dataframe_1


def _additional_qc_arguments(**kwargs):
    additional_arguments = {
        'noise': True,
        'basis_transformation_noise': False,
        'thread_safe_printing': True,
        'T1_lde': 2,
        'T1_idle': (5 * 60),
        'T2_idle': 10,
        'T2_idle_electron': 1,
        'T2_lde': 2,
        'T1_idle_electron': 1000,
        'no_single_qubit_error': True
    }
    kwargs.update(additional_arguments)
    return kwargs


def _print_circuit_parameters(**kwargs):
    protocol = kwargs.get('protocol')
    pg = kwargs.get('gate_error_probability')
    pm = kwargs.get('measurement_error_probability')
    pn = kwargs.get('network_error_probability')
    it = kwargs.get('iterations')
    pm_1 = kwargs.get('measurement_error_probability_one_state')
    stab_type = kwargs.get('stab_type')

    print("All circuit parameters:\n-----------------------\n")
    pprint(kwargs)
    print('\n-----------------------\n')


def _additional_parsing_of_arguments(args):
    # Pop the argument_file since it is no longer needed from this point
    args.pop("argument_file")

    # THIS IS NOT GENERIC, will error when directories are moved or renamed
    file_dir = os.path.dirname(__file__)
    look_up_table_dir = os.path.join(file_dir, '../gates', 'gate_lookup_tables')

    if args['single_qubit_gate_lookup'] is not None:
        with open(os.path.join(look_up_table_dir, args['single_qubit_gate_lookup']), 'rb') as obj:
            args['single_qubit_gate_lookup'] = pickle.load(obj)

    if args['two_qubit_gate_lookup'] is not None:
        with open(os.path.join(look_up_table_dir, args['two_qubit_gate_lookup']), "rb") as obj2:
            args['two_qubit_gate_lookup'] = pickle.load(obj2)

    gate_duration_file = args.pop('gate_duration_file')
    if gate_duration_file is not None and os.path.exists(gate_duration_file):
        set_gate_durations_from_file(gate_duration_file)
    elif gate_duration_file is not None:
        raise ValueError("Cannot find file to set gate durations with. File path: {}".format(gate_duration_file))

    return args


def main_threaded(*, iterations, fn, cp_path, **kwargs):
    # Run main method asynchronously with each worker getting an equal amount of iterations to run
    results = []
    workers = iterations if 0 < iterations < cpu_count() else cpu_count()
    thread_pool = Pool(workers)
    kwargs['iterations'] = iterations // workers
    for _ in range(workers):
        kwargs["threaded"] = True
        kwargs["progress_bar"] = None

        results.append(thread_pool.apply_async(main, kwds=kwargs))

    # Collect all the results from the workers and close the threadpool
    superoperator_results = []
    print_lines_results = []
    for res in results:
        superoperator_tuple, print_lines = res.get()
        superoperator_results.append(superoperator_tuple)
        print_lines_results.append(print_lines)
    thread_pool.close()

    # Check if csv already exists to append new data to it, if user requested saving of csv file
    normal = (pd.read_csv(fn + ".csv", sep=';', index_col=[0, 1])
                                   if fn and os.path.exists(fn + ".csv") else None)
    cut_off = (pd.read_csv(fn + "_failed.csv", sep=';', index_col=[0, 1]) if
                                  fn and os.path.exists(fn + "_failed.csv") else None)
    idle = (pd.read_csv(fn + "_idle.csv", sep=';', index_col=[0, 1]) if
                                fn and os.path.exists(fn + "_idle.csv") else None)

    # Combine the superoperator results obtained for each worker
    for (superoperator_succeed, superoperator_failed, superoperator_idle), print_line in zip(superoperator_results,
                                                                                             print_lines_results):
        normal = _combine_superoperator_dataframes(normal, superoperator_succeed)
        cut_off = _combine_superoperator_dataframes(cut_off, superoperator_failed)
        idle = _combine_superoperator_dataframes(idle, superoperator_idle)
        print(*print_line)

    # Save superoperator dataframe to csv if exists and requested by user
    if fn:
        for superoperator, fn_add in zip([normal, cut_off, idle], ['.csv', '_failed.csv', '_idle.csv']):
            filename = fn + fn_add
            superoperator.to_csv(filename, sep=';') if superoperator is not None else None
            os.system("rsync -rvu {} {}".format(os.path.dirname(filename), cp_path)) if cp_path and superoperator is \
                      not None else None


def main_series(fn, cp_path, **kwargs):
    (normal, cut_off, idle), print_lines = main(**kwargs)
    print(*print_lines)

    # Save the superoperator to the according csv files (options: normal, cut-off, idle)
    if fn and not args['print_run_order']:
        for result, fn_add in zip([normal, cut_off, idle], ['.csv', '_failed.csv', '_idle.csv']):
            fn_new = fn + fn_add
            existing_file = pd.read_csv(fn_new, sep=';', index_col=[0, 1]) if os.path.exists(fn_new) else None
            result = _combine_superoperator_dataframes(result, existing_file)
            if result is not None:
                result.to_csv(fn_new, sep=';')


def main(*, iterations, protocol, stabilizer_type, print_run_order, threaded=False, gate_duration_file=None,
         color=False, draw_circuit=True, save_latex_pdf=False, to_console=False, **kwargs):
    supop_dataframe_failed = None
    supop_dataframe_succeed = None
    supop_dataframe_idle = None
    total_print_lines = []

    # Progress bar initialisation
    progress_bar = kwargs.pop('progress_bar')
    if progress_bar:
        from tqdm import tqdm
    pbar = tqdm(total=100, position=0) if progress_bar else None
    pbar_2 = tqdm(total=iterations, position=1) if progress_bar and iterations > 1 else None

    # Get the QuantumCircuit object corresponding to the protocol and the protocol method by its name
    kwargs = _additional_qc_arguments(**kwargs)
    qc = stab_protocols.create_quantum_circuit(protocol, pbar, **kwargs)
    protocol_method = getattr(stab_protocols, protocol)

    # Run iterations of the protocol
    for iter in range(iterations):
        pbar.reset() if pbar else None
        if pbar_2 and iterations > 1:
            pbar_2.update(1)
        if pbar_2 is None:
            print(">>> At iteration {} of the {}.".format(iter + 1, iterations), end='\r', flush=True)

        if threaded:
            _init_random_seed(worker=threading.get_ident(), iteration=iter)
            set_gate_durations_from_file(gate_duration_file)

        if print_run_order:
            return (None, None), []

        # Run the user requested protocol
        operation = CZ_gate if stabilizer_type == "Z" else CNOT_gate
        superoperator_qubits_list = protocol_method(qc, operation=operation)

        if draw_circuit:
            qc.draw_circuit(no_color=not color, color_nodes=True)

        if save_latex_pdf:
            qc.draw_circuit_latex()

        # If no superoperator qubits are returned, take the data qubits as such
        if superoperator_qubits_list is None:
            superoperator_qubits_list = [qc.data_qubits]
        if type(superoperator_qubits_list[0]) == int:
            superoperator_qubits_list = [superoperator_qubits_list]

        supop_dataframe = []
        for i, superoperator_qubits in enumerate(superoperator_qubits_list):
            idle_data_qubit = 4 if i != 0 else False
            _, dataframe = qc.get_superoperator(superoperator_qubits, stabilizer_type, no_color=(not color),
                                                stabilizer_protocol=True, print_to_console=to_console,
                                                idle_data_qubit=idle_data_qubit, protocol_name=protocol)
            supop_dataframe.append(dataframe)

        pbar.update(10) if pbar is not None else None

        # Check if possible additional idle data qubit superoperator is present
        if len(supop_dataframe) > 1:
            supop_dataframe_idle = _combine_superoperator_dataframes(supop_dataframe_idle, supop_dataframe[1])

        # Fuse the superoperator dataframes obtained in each iteration
        if qc.cut_off_time_reached:
            supop_dataframe_failed = _combine_superoperator_dataframes(supop_dataframe_failed, supop_dataframe[0])
        else:
            supop_dataframe_succeed = _combine_superoperator_dataframes(supop_dataframe_succeed, supop_dataframe[0])

        total_print_lines.extend(qc.print_lines)
        total_print_lines.append("\nGHZ fidelity: {} ".format(qc.ghz_fidelity)) \
            if draw_circuit else None
        total_print_lines.append("\nTotal circuit duration: {} seconds".format(qc.total_duration)) \
            if draw_circuit else None
        qc.reset()

    return (supop_dataframe_succeed, supop_dataframe_failed, supop_dataframe_idle), total_print_lines


def run_for_arguments(protocols, gate_error_probabilities, network_error_probabilities, meas_error_probabilities,
                      meas_error_probabilities_one_state, csv_filename, no_progress_bar, pm_equals_pg,
                      use_swap_gates, fixed_lde_attempts, **args):

    meas_1_errors = [None] if meas_error_probabilities_one_state is None else meas_error_probabilities_one_state
    meas_errors = [None] if meas_error_probabilities is None else meas_error_probabilities

    # Loop over command line arguments
    for protocol, pg, pn, pm, pm_1, lde in itertools.product(protocols, gate_error_probabilities,
                                                             network_error_probabilities, meas_errors, meas_1_errors,
                                                             fixed_lde_attempts):
        pm = pg if pm is None or pm_equals_pg else pm
        protocol = protocol + "_swap" if use_swap_gates else protocol

        fn = "{}_{}_pg{}_pn{}_pm{}_pm_1{}_lde{}"\
            .format(csv_filename, protocol, pg, pn, pm, pm_1 if pm_1 is not None else "", lde) \
            if csv_filename else None

        print("\nRunning {} iteration(s): protocol={}, pg={}, pn={}, pm={}, pm_1={}, fixed_lde_attempts={}"
              .format(args['iterations'], protocol, pg, pn, pm, pm_1, lde))

        if args['threaded']:
            main_threaded(protocol=protocol, pg=pg, pm=pm, pm_1=pm_1, pn=pn, progress_bar=no_progress_bar, fn=fn,
                          fixed_lde_attempts=lde, **args)
        else:
            main_series(protocol=protocol, pg=pg, pm=pm, pm_1=pm_1, pn=pn, progress_bar=no_progress_bar, fn=fn,
                        fixed_lde_attempts=lde, **args)


if __name__ == "__main__":
    parser = compose_parser()

    args = vars(parser.parse_args())
    args = _additional_parsing_of_arguments(args)
    _print_circuit_parameters(**copy(args))

    # Loop over all possible combinations of the user determined parameters
    run_for_arguments(**args)
