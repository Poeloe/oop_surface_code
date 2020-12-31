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
from plot_non_local_cnot import mean_confidence_interval
from circuit_simulation.termcolor.termcolor import cprint


def print_signature():
    cprint("\n Quantum Circuit Simulator® wishes you\n", color='cyan')
    cprint(''.join(['\t\t', ' ' * 10 + '*' + ' ' * 10]), color='yellow')
    cprint("".join(
        '\t\t{0}{1}{0}\n'.format(' ' * ((21 - c) // 2), ''.join(map(lambda i: '#' if i % 2 else 'o', range(c)))) for
        c in
        range(3, 22, 2)), color='green')
    cprint("".join(['\t\t', ' ' * 9 + '/|\\' + ' ' * 9]), color='red')
    cprint("\n a Merry Christmas and a Happy New Year!\n\n", color="red")
    print("\n --------------------------------------------------- \n")


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


def _print_circuit_parameters(**kwargs):
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

    gate_duration_file = args.get('gate_duration_file')
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
    characteristics_dicts = []
    for res in results:
        superoperator_tuple, print_lines, characteristics = res.get()
        superoperator_results.append(superoperator_tuple)
        print_lines_results.append(print_lines)
        characteristics_dicts.append(characteristics)
    thread_pool.close()

    # Check if csv already exists to append new data to it, if user requested saving of csv file
    normal = _open_existing_superoperator_file(fn, ".csv")
    cut_off = _open_existing_superoperator_file(fn, "_failed.csv")

    # Combine the superoperator results obtained for each worker
    for (superoperator_succeed, superoperator_failed), print_line in zip(superoperator_results, print_lines_results):
        normal = _combine_superoperator_dataframes(normal, superoperator_succeed)
        cut_off = _combine_superoperator_dataframes(cut_off, superoperator_failed)
        print(*print_line)

    # Adding confidence intervals to the superoperator
    characteristics_dict = None
    if normal is not None:
        stab_fids = []
        ghz_fids = []
        dur = []
        [(stab_fids.extend(d['stab_fid']), ghz_fids.extend(d['ghz_fid']), dur.extend(d['dur']), )
         for d in characteristics_dicts]
        characteristics_dict = {'stab_fid': stab_fids, 'ghz_fid': ghz_fids, 'dur': dur}
        add_column_values(normal, ['int_stab', 'int_ghz', 'int_dur', 'dur_99', 'stab_std', 'ghz_std', 'dur_std'],
                          [mean_confidence_interval(stab_fids),
                           mean_confidence_interval(ghz_fids),
                           mean_confidence_interval(dur),
                           mean_confidence_interval(dur, 0.99, True),
                           np.std(stab_fids),
                           np.std(ghz_fids),
                           np.std(dur)])

    # Save superoperator dataframe to csv if exists and requested by user
    if fn:
        pickle.dump(characteristics_dict, file=open(fn + '.pkl', 'wb')) if characteristics_dict else None
        for superoperator, fn_add in zip([normal, cut_off], ['.csv', '_failed.csv']):
            filename = fn + fn_add
            superoperator.to_csv(filename, sep=';') if superoperator is not None else None
            os.system("rsync -rvu {} {}".format(os.path.dirname(filename), cp_path)) if cp_path and superoperator is \
                                                                                        not None else None


def main_series(fn, cp_path, **kwargs):
    (normal, cut_off), print_lines, characteristics = main(**kwargs)
    print(*print_lines)

    # Adding the confidence intervals to the superoperator
    if normal is not None:
        add_column_values(normal, ['int_stab', 'int_ghz', 'int_dur', 'dur_99', 'stab_std', 'ghz_std', 'dur_std'],
                          [mean_confidence_interval(characteristics['stab_fid']),
                           mean_confidence_interval(characteristics['ghz_fid']),
                           mean_confidence_interval(characteristics['dur']),
                           mean_confidence_interval(characteristics['dur'], 0.99, True),
                           np.std(characteristics['stab_fid']),
                           np.std(characteristics['ghz_fid']),
                           np.std(characteristics['dur'])])

    # Save the superoperator to the according csv files (options: normal, cut-off, idle)
    if fn and not args['print_run_order']:
        pickle.dump(characteristics, file=open(fn + '.pkl', 'wb')) if characteristics else None
        for result, fn_add in zip([normal, cut_off], ['.csv', '_failed.csv']):
            fn_new = fn + fn_add
            existing_file = _open_existing_superoperator_file(fn_new)
            result = _combine_superoperator_dataframes(result, existing_file)
            if result is not None:
                result.to_csv(fn_new, sep=';')


def main(*, iterations, protocol, stabilizer_type, print_run_order, threaded=False, gate_duration_file=None,
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
        if pbar_2 and iterations > 1:
            pbar_2.update(1)
        if pbar_2 is None:
            print(">>> At iteration {} of the {}.".format(iter + 1, iterations), end='\r', flush=True)

        _init_random_seed(worker=threading.get_ident(), iteration=iter)

        # Run the user requested protocol
        operation = CZ_gate if stabilizer_type == "Z" else CNOT_gate
        superoperator_qubits_list = protocol_method(qc, operation=operation)
        qc.end_current_sub_circuit(total=True)
        add_decoherence_if_cut_off(qc)

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
        total_print_lines.append("\nGHZ fidelity: {} ".format(qc.ghz_fidelity)) \
            if draw_circuit else None
        total_print_lines.append("\nTotal circuit duration: {} seconds".format(qc.total_duration)) \
            if draw_circuit else None
        qc.reset()

    return (supop_dataframe_succeed, supop_dataframe_failed), total_print_lines, characteristics


def run_for_arguments(protocols, gate_error_probabilities, network_error_probabilities, meas_error_probabilities,
                      meas_error_probabilities_one_state, csv_filename, no_progress_bar, pm_equals_pg,
                      use_swap_gates, fixed_lde_attempts, pulse_duration, **args):

    meas_1_errors = [None] if meas_error_probabilities_one_state is None else meas_error_probabilities_one_state
    meas_errors = [None] if meas_error_probabilities is None else meas_error_probabilities

    # Loop over command line arguments
    for protocol, pg, pn, pm, pm_1, lde, pulse in itertools.product(protocols, gate_error_probabilities,
                                                                    network_error_probabilities, meas_errors,
                                                                    meas_1_errors, fixed_lde_attempts, pulse_duration):
        pm = pg if pm is None or pm_equals_pg else pm
        protocol = protocol + "_swap" if use_swap_gates else protocol

        fn = "{}_{}_pg{}_pn{}_pm{}_pm_1{}_lde{}_pulse{}"\
            .format(csv_filename, protocol, pg, pn, pm, pm_1 if pm_1 is not None else "", lde, pulse) \
            if csv_filename else None

        print("\nRunning {} iteration(s): protocol={}, pg={}, pn={}, pm={}, pm_1={}, fixed_lde_attempts={}, pulse={}"
              .format(args['iterations'], protocol, pg, pn, pm, pm_1, lde, pulse))

        if args['threaded']:
            main_threaded(protocol=protocol, pg=pg, pm=pm, pm_1=pm_1, pn=pn, progress_bar=no_progress_bar, fn=fn,
                          fixed_lde_attempts=lde, pulse_duration=pulse, **args)
        else:
            main_series(protocol=protocol, pg=pg, pm=pm, pm_1=pm_1, pn=pn, progress_bar=no_progress_bar, fn=fn,
                        fixed_lde_attempts=lde, pulse_duration=pulse, **args)


if __name__ == "__main__":
    parser = compose_parser()

    args = vars(parser.parse_args())
    args = _additional_parsing_of_arguments(args)
    print_signature()
    time.sleep(1)
    _print_circuit_parameters(**copy(args))

    # Loop over all possible combinations of the user determined parameters
    run_for_arguments(**args)
