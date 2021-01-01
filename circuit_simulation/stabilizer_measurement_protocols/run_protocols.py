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
from circuit_simulation.circuit_simulator import QuantumCircuit
import itertools
import time
from copy import copy
import random
from plot_non_local_cnot import confidence_interval
from circuit_simulation.termcolor.termcolor import cprint
from collections import defaultdict
import numpy as np


def print_signature():
    cprint("\nQuantum Circuit Simulator®", color='cyan')
    print("--------------------------\n")


def _get_cut_off_dataframe(file):
    if file is None:
        return
    if not os.path.exists(file):
        raise ValueError('File containing the cut-off times could not be found!')

    return pd.read_csv(file, sep=";", float_precision='round_trip')


def _get_cut_off_time(dataframe, **kwargs):
    cut_off_time = kwargs.pop('cut_off_time')

    if cut_off_time != np.inf or dataframe is None:
        return cut_off_time

    kwarg_cols = ['pm_1', 'pg', 'fixed_lde_attempts', 'pulse_duration', 'network_noise_type', 'pm', 'pn',
                  'probabilistic', 'decoherence']
    index = [kwargs[key] for key in kwarg_cols]
    index_dict = dict(zip(kwarg_cols, index))
    protocol_name = kwargs['protocol'] if kwargs['T1_lde'] == 2 else kwargs['protocol'] + "_na"
    index_dict['protocol_name'] = protocol_name
    index_dict['fixed_lde_attempts'] = 0 if index_dict['pulse_duration'] == 0 else index_dict['fixed_lde_attempts']

    dataframe = dataframe.set_index(list(index_dict.keys()))

    if tuple(index_dict.values()) not in dataframe.index:
        raise ValueError("Cut-off value not found: Index does not exist in dataframe:\n{}".format(index_dict))

    kwargs.pop('protocol')

    return dataframe.loc[tuple(index_dict.values()), '99_duration']


def _open_existing_superoperator_file(filename, addition=""):
    if filename is None:
        return
    if not os.path.exists(filename + addition):
        return

    existing_file = pd.read_csv(filename + addition, sep=';')
    index = ['error_config', 'lie'] if 'error_idle' not in existing_file else ['error_stab', 'error_idle', 'lie']

    existing_file = existing_file.set_index(index)

    return existing_file


def _combine_idle_and_stabilizer_superoperator(dataframes):
    def combine_dataframes(stab, stab_idle, type):
        for stab_item in stab.iteritems():
            for stab_idle_item in stab_idle.iteritems():
                (p_error, p_lie), p_value = stab_item
                (p_error_idle, p_lie_idle), p_value_idle = stab_idle_item

                if p_lie == p_lie_idle:
                    combined_dataframe.loc[(p_error, p_error_idle, p_lie), type] = p_value * (p_value_idle*2)

    if len(dataframes) < 2:
        return dataframes[0]

    superoperator_stab = dataframes[0]
    superoperator_idle = dataframes[1]

    index = pd.MultiIndex.from_product([[item[0] for item in superoperator_stab.index if item[1]],
                                        [item[0] for item in superoperator_idle.index if item[1]],
                                        [False, True]],
                                       names=['error_stab', 'error_idle', 'lie'])
    combined_dataframe = pd.DataFrame(columns=superoperator_stab.columns, index=index)

    combined_dataframe = combined_dataframe.sort_index()

    combine_dataframes(superoperator_stab['s'], superoperator_idle['s'], 's')
    combine_dataframes(superoperator_stab['p'], superoperator_idle['p'], 'p')

    combined_dataframe.iloc[0, 2:] = superoperator_stab.iloc[0, 2:]
    combined_dataframe.iloc[0, combined_dataframe.columns.get_loc('qubit_order')] = \
        (superoperator_stab.iloc[0, superoperator_stab.columns.get_loc('qubit_order')] +
         superoperator_idle.iloc[0, superoperator_idle.columns.get_loc('qubit_order')])
    combined_dataframe = combined_dataframe[(combined_dataframe.T
                                             .applymap(lambda x: x != 0 and x is not None and not pd.isna(x))).any()]

    return combined_dataframe


def _init_random_seed(timestamp=None, worker=0, iteration=0):
    if timestamp is None:
        timestamp = time.time()
    seed = int("{:.0f}".format(timestamp * 10 ** 7) + str(worker) + str(iteration))
    random.seed(float(seed))
    return seed


def add_column_values(dataframe, columns, values):
    for column, value in zip(columns, values):
        dataframe[column] = None
        dataframe.iloc[0, dataframe.columns.get_loc(column)] = value


def _combine_superoperator_dataframes(dataframe_1, dataframe_2):
    """
        Combines two given superoperator dataframes into one dataframe

        Parameters
        ----------
        dataframe_1 : pd.DataFrame or None
            Superoperator dataframe to be combined
        dataframe_2 : pd.DataFrame or None
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
    dataframe_1['total_lde_attempts'] = (dataframe_1['total_lde_attempts'] + dataframe_2['total_lde_attempts'])

    dataframe_1['avg_lde_attempts'] = dataframe_1['total_lde_attempts'] / corrected_written_to
    dataframe_1['avg_duration'] = dataframe_1['total_duration'] / corrected_written_to

    # Update fidelity
    dataframe_2['ghz_fidelity'] = dataframe_2['ghz_fidelity'].mul(written_to_new)
    dataframe_1['ghz_fidelity'] = dataframe_1['ghz_fidelity'].mul(written_to_original)

    dataframe_1['ghz_fidelity'] = (dataframe_1['ghz_fidelity'] + dataframe_2['ghz_fidelity']) / corrected_written_to

    return dataframe_1


def add_decoherence_if_cut_off(qc: QuantumCircuit):
    if qc.cut_off_time < np.inf and not qc.cut_off_time_reached:
        waiting_time = qc.cut_off_time - qc.total_duration
        if waiting_time > 0:
            qc._increase_duration(waiting_time, [], involved_nodes=list(qc.nodes.keys()), check=False)
            qc.end_current_sub_circuit(total=True, duration=waiting_time, sub_circuit="Waiting")


def _additional_qc_arguments(**kwargs):
    additional_arguments = {
        'noise': True,
        'basis_transformation_noise': False,
        'thread_safe_printing': True,
        'no_single_qubit_error': True
    }
    kwargs.update(additional_arguments)
    return kwargs


def print_circuit_parameters(**kwargs):
    print("All circuit parameters:\n-----------------------\n")
    pprint(kwargs)
    print('\n-----------------------\n')


def additional_parsing_of_arguments(**args):
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

    gate_duration_file = args.get('gate_duration_file')
    if gate_duration_file is not None and os.path.exists(gate_duration_file):
        set_gate_durations_from_file(gate_duration_file)
    elif gate_duration_file is not None:
        raise ValueError("Cannot find file to set gate durations with. File path: {}".format(gate_duration_file))

    return args


def _save_superoperator_dataframe(fn, characteristics, succeeded, cut_off, cp_path):
    # Adding confidence intervals to the superoperator
    succeeded = _add_interval_to_dataframe(succeeded, characteristics)

    if fn:
        pickle.dump(characteristics, file=open(fn + '.pkl', 'wb')) if characteristics else None
        for result, fn_add in zip([succeeded, cut_off], ['.csv', '_failed.csv']):
            fn_new = fn + fn_add
            existing_file = _open_existing_superoperator_file(fn_new)
            result = _combine_superoperator_dataframes(result, existing_file)
            if result is not None:
                result.to_csv(fn_new, sep=';')
                os.system("rsync -rvu {} {}".format(os.path.dirname(fn), cp_path)) if cp_path else None


def _add_interval_to_dataframe(dataframe, characteristics):
    if dataframe is not None:
        add_column_values(dataframe, ['int_stab_682', 'int_ghz_682', 'int_dur_682', 'dur_99'],
                          [str(confidence_interval(characteristics['stab_fid'])),
                           str(confidence_interval(characteristics['ghz_fid'])),
                           str(confidence_interval(characteristics['dur'])),
                           str(confidence_interval(characteristics['dur'], 0.98)[1])])
    return dataframe


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

    # Collect all the results from the workers and close the thread pool
    succeeded = None
    cut_off = None
    print_lines_results = []
    tot_characteristics = defaultdict(list)
    for res in results:
        (succeeded_res, cut_off_res), print_lines, characteristics = res.get()
        succeeded = _combine_superoperator_dataframes(succeeded, succeeded_res)
        cut_off = _combine_superoperator_dataframes(cut_off, cut_off_res)
        print_lines_results.extend(print_lines)
        [tot_characteristics[key].extend(value) for key, value in characteristics.items()]
    thread_pool.close()

    print(*print_lines_results)

    # Save superoperator dataframe to csv if exists and requested by user
    _save_superoperator_dataframe(fn, tot_characteristics, succeeded, cut_off, cp_path)


def main_series(fn, cp_path, **kwargs):
    (succeeded, cut_off), print_lines, characteristics = main(**kwargs)
    print(*print_lines)

    # Save the superoperator to the according csv files (options: normal, cut-off)
    _save_superoperator_dataframe(fn, characteristics, succeeded, cut_off, cp_path)


def main(*, iterations, protocol, stabilizer_type, threaded=False, gate_duration_file=None,
         color=False, draw_circuit=True, save_latex_pdf=False, to_console=False, **kwargs):
    supop_dataframe_failed = None
    supop_dataframe_succeed = None
    total_print_lines = []
    characteristics = {'dur': [], 'stab_fid': [], 'ghz_fid': []}

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

    if threaded:
        set_gate_durations_from_file(gate_duration_file)

    # Run iterations of the protocol
    for iter in range(iterations):
        pbar.reset() if pbar else None
        pbar_2.update(1) if pbar_2 and iterations > 1 else None

        print(">>> At iteration {}/{}.".format(iter + 1, iterations), end='\r', flush=True) if pbar_2 is None else None

        _init_random_seed(worker=threading.get_ident(), iteration=iter)

        # Run the user requested protocol
        operation = CZ_gate if stabilizer_type == "Z" else CNOT_gate
        superoperator_qubits_list = protocol_method(qc, operation=operation)
        add_decoherence_if_cut_off(qc)

        qc.draw_circuit(no_color=not color, color_nodes=True) if draw_circuit else None
        qc.draw_circuit_latex() if save_latex_pdf else None

        # If no superoperator qubits are returned, take the data qubits as such
        superoperator_qubits_list = [qc.data_qubits] if superoperator_qubits_list is None else superoperator_qubits_list

        # Obtain the superoperator in a dataframe format
        supop_dataframe = []
        for i, superoperator_qubits in enumerate(superoperator_qubits_list):
            idle_data_qubit = 4 if i != 0 else False
            _, dataframe = qc.get_superoperator(superoperator_qubits, stabilizer_type, no_color=(not color),
                                                stabilizer_protocol=True, print_to_console=to_console,
                                                idle_data_qubit=idle_data_qubit, protocol_name=protocol)
            supop_dataframe.append(dataframe)

        supop_dataframe = _combine_idle_and_stabilizer_superoperator(supop_dataframe)
        pbar.update(10) if pbar is not None else None

        if not qc.cut_off_time_reached:
            characteristics['dur'] += [qc.total_duration]
            characteristics['ghz_fid'] += [qc.ghz_fidelity]
            characteristics['stab_fid'] += [supop_dataframe.iloc[0, 0]]

        # Fuse the superoperator dataframes obtained in each iteration
        if qc.cut_off_time_reached:
            supop_dataframe_failed = _combine_superoperator_dataframes(supop_dataframe_failed, supop_dataframe)
        else:
            supop_dataframe_succeed = _combine_superoperator_dataframes(supop_dataframe_succeed, supop_dataframe)

        total_print_lines.extend(qc.print_lines)
        total_print_lines.append("\nGHZ fidelity: {} ".format(qc.ghz_fidelity)) if draw_circuit else None
        total_print_lines.append("\nTotal circuit duration: {} s".format(qc.total_duration)) if draw_circuit else None
        qc.reset()

    return (supop_dataframe_succeed, supop_dataframe_failed), total_print_lines, characteristics


def run_for_arguments(protocols, gate_error_probabilities, network_error_probabilities, meas_error_probabilities,
                      meas_error_probabilities_one_state, csv_filename, no_progress_bar, pm_equals_pg,
                      use_swap_gates, fixed_lde_attempts, pulse_duration, cut_off_file, force_run, **args):

    meas_1_errors = [None] if meas_error_probabilities_one_state is None else meas_error_probabilities_one_state
    meas_errors = [None] if meas_error_probabilities is None else meas_error_probabilities
    cut_off_dataframe = _get_cut_off_dataframe(cut_off_file)
    filenames = []

    # Loop over command line arguments
    for protocol, pg, pn, pm, pm_1, lde, pulse in itertools.product(protocols, gate_error_probabilities,
                                                                    network_error_probabilities, meas_errors,
                                                                    meas_1_errors, fixed_lde_attempts, pulse_duration):
        pm = pg if pm is None or pm_equals_pg else pm
        protocol = protocol + "_swap" if use_swap_gates else protocol
        cut_off_time = _get_cut_off_time(cut_off_dataframe, protocol=protocol, pg=pg, pm=pm, pm_1=pm_1, pn=pn,
                                         pulse_duration=pulse, progress_bar=no_progress_bar,
                                         fixed_lde_attempts=lde, **args)
        args.pop('cut_off_time')
        node = "pur" if args.get('T1_lde') == 2 else 'nat_ab'

        fn = "{}_{}_pg{}_pn{}_pm{}_pm_1{}_lde{}_pulse{}_node_{}_cutoff_{}"\
            .format(csv_filename, protocol, pg, pn, pm, pm_1 if pm_1 is not None else "", lde, pulse, node,
                    cut_off_time) if csv_filename else None
        filenames.append(fn)

        if not force_run and fn is not None and os.path.exists(fn + ".csv"):
            print("Skipping circuit for file '{}', since it already exists.".format(fn))
            continue

        print("\nRunning {} iteration(s): protocol={}, pg={}, pn={}, pm={}, pm_1={}, fixed_lde_attempts={}, pulse={}, "
              "cut_off_time={}".format(args['iterations'], protocol, pg, pn, pm, pm_1, lde, pulse, cut_off_time))

        if args['threaded']:
            main_threaded(protocol=protocol, pg=pg, pm=pm, pm_1=pm_1, pn=pn, progress_bar=no_progress_bar, fn=fn,
                          fixed_lde_attempts=lde, pulse_duration=pulse, cut_off_time=cut_off_time, **args)
        else:
            main_series(protocol=protocol, pg=pg, pm=pm, pm_1=pm_1, pn=pn, progress_bar=no_progress_bar, fn=fn,
                        fixed_lde_attempts=lde, pulse_duration=pulse, cut_off_time=cut_off_time, **args)

    return filenames


if __name__ == "__main__":
    parser = compose_parser()

    args = vars(parser.parse_args())
    args = additional_parsing_of_arguments(**args)
    print_signature()
    print_circuit_parameters(**args)

    # Loop over all possible combinations of the user determined parameters
    run_for_arguments(**args)
