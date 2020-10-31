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
    dataframe_1['avg_lde'] = dataframe_1['lde_attempts'] / corrected_written_to
    dataframe_1['avg_duration'] = dataframe_1['total_duration'] / corrected_written_to

    return dataframe_1


def _print_circuit_parameters(**kwargs):
    protocol = kwargs.get('protocol')
    pg = kwargs.get('gate_error_probability')
    pm = kwargs.get('measurement_error_probability')
    pn = kwargs.get('network_error_probability')
    it = kwargs.get('iterations')
    pm_1 = kwargs.get('measurement_error_probability_one_state')
    stab_type = kwargs.get('stab_type')

    print("\nRunning the {} protocols, with pg={}, pm={}, pm_1={}{}, for a {} stabilizer {} time{}.\n"
          .format(protocol, pg, pm, pm_1, (' and pn=' + str(pn) if protocol != 'monolithic' else ""),
                  "plaquette" if stab_type == "Z" else "star", it, "s" if it > 1 else ""))

    print("All circuit parameters:\n-----------------------\n")
    pprint(kwargs)
    print('\n-----------------------\n')


def main_threaded(*, it, workers, fn, **kwargs):
    # Run main method asynchronously with each worker getting an equal amount of iterations to run
    results = []
    for _ in range(workers):
        kwargs["it"] = it // workers
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
    total_superoperator_succeed = (pd.read_csv(fn + ".csv", sep=';', index_col=[0, 1])
                                   if fn and os.path.exists(fn + ".csv") else None)
    total_superoperator_failed = (pd.read_csv(fn + "_failed.csv", sep=';', index_col=[0, 1]) if
                                  fn and os.path.exists(fn + "_failed.csv") else None)

    # Combine the superoperator results obtained for each worker
    for (superoperator_succeed, superoperator_failed), print_line in zip(superoperator_results, print_lines_results):
        total_superoperator_succeed = _combine_superoperator_dataframes(total_superoperator_succeed,
                                                                        superoperator_succeed)
        total_superoperator_failed = _combine_superoperator_dataframes(total_superoperator_failed, superoperator_failed)
        print(*print_line)

    # Save superoperator dataframe to csv if exists and requested by user
    if total_superoperator_succeed is not None and fn:
        total_superoperator_succeed.to_csv(fn + ".csv", sep=';')
    if total_superoperator_failed is not None and fn:
        total_superoperator_failed.to_csv(fn + "_failed.csv", sep=';')


def main(*, it, protocol, stabilizer_type, print_run_order, threaded=False, gate_duration_file=None,
         **kwargs):
    supop_dataframe_failed = None
    supop_dataframe_succeed = None
    total_print_lines = []

    # Progress bar initialisation
    progress_bar = kwargs.pop('progress_bar')
    pbar = tqdm(total=100, position=0) if progress_bar else None
    pbar_2 = tqdm(total=it, position=1) if progress_bar and it > 1 else None

    # Gather method arguments
    operation = CZ_gate if stabilizer_type == "Z" else CNOT_gate
    arguments = ['color', 'save_latex_pdf', 'draw_circuit', 'to_console']
    protocol_args = {'operation': operation}
    for argument in arguments:
        argument_value = kwargs.pop(argument)
        protocol_args[argument] = argument_value

    # Get the QuantumCircuit object corresponding to the protocol and the protocol method by its name
    qc = stab_protocols.create_quantum_circuit(protocol, **kwargs)
    protocol_method = getattr(stab_protocols, protocol)

    # Run iterations of the protocol
    for i in range(it):
        pbar.reset() if pbar else None
        if pbar_2 and it > 1:
            pbar_2.update(1)
        if pbar_2 is None:
            print(">>> At iteration {} of the {}.".format(i + 1, it), end='\r', flush=True)

        if threaded:
            _init_random_seed(worker=threading.get_ident(), iteration=i)
            set_gate_durations_from_file(gate_duration_file)

        if print_run_order:
            return (None, None), []

        # Run the user requested protocol
        (supop_dataframe, cut_off), print_lines = protocol_method(qc, pbar=pbar, **protocol_args)

        # Fuse the superoperator dataframes obtained in each iteration
        if cut_off:
            supop_dataframe_failed = _combine_superoperator_dataframes(supop_dataframe_failed, supop_dataframe)
        else:
            supop_dataframe_succeed = _combine_superoperator_dataframes(supop_dataframe_succeed, supop_dataframe)

        total_print_lines.extend(print_lines)

    return (supop_dataframe_succeed, supop_dataframe_failed), total_print_lines


if __name__ == "__main__":
    parser = compose_parser()

    args = vars(parser.parse_args())
    _print_circuit_parameters(**copy(args))

    # Pop the command line arguments from the list that should be evaluated first
    it = args.pop('iterations')
    protocols = args.pop('protocol')
    meas_errors = args.pop('measurement_error_probability')
    meas_1_errors = args.pop('measurement_error_probability_one_state')
    meas_eq_gate = args.pop('pm_equals_pg')
    network_errors = args.pop('network_error_probability')
    gate_errors = args.pop('gate_error_probability')
    filename = args.pop('csv_filename')
    threaded = args.pop('threaded')
    lkt_1q = args.pop('lookup_table_single_qubit_gates')
    lkt_2q = args.pop('lookup_table_two_qubit_gates')
    gate_duration_file = args.pop('gate_duration_file')
    progress_bar = args.pop('no_progress_bar')
    args.pop("argument_file")
    use_swap_gates = args.pop('use_swap_gates')

    # THIS IS NOT GENERIC, will error when directories are moved or renamed
    file_dir = os.path.dirname(__file__)
    look_up_table_dir = os.path.join(file_dir, '../gates', 'gate_lookup_tables')

    if lkt_1q is not None:
        with open(os.path.join(look_up_table_dir, lkt_1q), 'rb') as obj:
            lkt_1q = pickle.load(obj)

    if lkt_2q is not None:
        with open(os.path.join(look_up_table_dir, lkt_2q), "rb") as obj2:
            lkt_2q = pickle.load(obj2)

    if gate_duration_file is not None and os.path.exists(gate_duration_file):
        set_gate_durations_from_file(gate_duration_file)
    elif gate_duration_file is not None:
        raise ValueError("Cannot find file to set gate durations with. File path: {}".format(gate_duration_file))

    if progress_bar:
        from tqdm import tqdm

    meas_1_errors = [None] if meas_1_errors is None else meas_1_errors
    meas_errors = [None] if meas_errors is None else meas_errors

    # Loop over all possible combinations of the user determined parameters
    for protocol, pg, pn, pm, pm_1 in itertools.product(protocols, gate_errors, network_errors, meas_errors,
                                                        meas_1_errors):
        pm = pg if pm is None else pm
        protocol = protocol + "_swap" if use_swap_gates else protocol

        fn = "{}_{}_pg{}_pn{}_pm{}_pm_1{}".format(filename, protocol, pg, pn, pm, pm_1 if pm_1 is not None else "") \
            if filename else None
        print("\nRunning now {} iterations: protocol={}, pg={}, pn={}, pm={}, pm_1={}".format(it, protocol, pg, pn, pm,
                                                                                             pm_1))
        if threaded:
            workers = it if 0 < it < cpu_count() else cpu_count()
            thread_pool = Pool(workers)
            main_threaded(it=it, protocol=protocol, pg=pg, pm=pm, pm_1=pm_1, pn=pn, lkt_1q=lkt_1q, lkt_2q=lkt_2q, fn=fn,
                          progress_bar=progress_bar, gate_duration_file=gate_duration_file, workers=workers, **args)
        else:
            (dataframe, cut_off), print_lines = main(it=it, protocol=protocol, pg=pg, pm=pm, pm_1=pm_1, pn=pn,
                                                     progress_bar=progress_bar, lkt_1q=lkt_1q, lkt_2q=lkt_2q,
                                                     gate_duration_file=gate_duration_file, **args)
            print(*print_lines)
            if filename and not args['print_run_order'] and os.path.exists(fn + ".csv"):
                dataframe = _combine_superoperator_dataframes(pd.read_csv(fn + ".csv", sep=';', index_col=[0, 1]),
                                                              dataframe)
            if fn:
                dataframe.to_csv(fn + ".csv", sep=';')
