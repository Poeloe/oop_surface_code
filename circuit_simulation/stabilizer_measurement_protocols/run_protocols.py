import os
import sys
sys.path.insert(1, os.path.abspath(os.getcwd()))
from pprint import pprint
from multiprocessing import Pool, cpu_count
import threading
import pickle
import pandas as pd
import circuit_simulation.stabilizer_measurement_protocols.stabilizer_measurement_protocols as stab_protocols
from circuit_simulation.stabilizer_measurement_protocols.argument_parsing import compose_parser, group_arguments
from circuit_simulation.gates.gates import *
from circuit_simulation.circuit_simulator import QuantumCircuit
import itertools as it
import time
import random
from plot_non_local_cnot import confidence_interval
from circuit_simulation.termcolor.termcolor import cprint
from collections import defaultdict
import numpy as np
from tqdm import tqdm


def print_signature():
    cprint("\nQuantum Circuit Simulator®", color='cyan')
    print("--------------------------")


def create_file_name(filename, **kwargs):
    protocol = kwargs.pop('protocol')
    filename = "{}{}{}".format(filename, "_" if filename[-1] not in "/_" else "", protocol)

    for key, value in kwargs.items():
        # Do not include if value is None, 0 or np.inf (default cut_off_time) or if key is pulse_duration
        if not value or value == np.inf or key == 'pulse_duration':
            continue
        if value is True:
            value = ""
        value = value.capitalize() if type(value) == str else str(value)
        filename += "_" + str(key) + value

    return filename.strip('_')


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
    index_dict['protocol_name'] = kwargs['protocol']
    index_dict['node'] = 'Purified' if kwargs['T1_lde'] == 2 else "Natural Abundance"

    dataframe = dataframe.set_index(list(index_dict.keys()))

    if tuple(index_dict.values()) not in dataframe.index:
        raise ValueError("Cut-off value not found: Index does not exist in dataframe:\n{}".format(index_dict))

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

    dataframe_1[['p', 's']] = dataframe_1[['p', 's']].add(dataframe_2[['p', 's']], fill_value=0) / corrected_written_to

    # Update the average of the other system characteristics
    dataframe_1['total_duration'] = (dataframe_1['total_duration'] + dataframe_2['total_duration'])
    dataframe_1['total_lde_attempts'] = (dataframe_1['total_lde_attempts'] + dataframe_2['total_lde_attempts'])

    dataframe_1['avg_lde_attempts'] = dataframe_1['total_lde_attempts'] / corrected_written_to
    dataframe_1['avg_duration'] = dataframe_1['total_duration'] / corrected_written_to

    # Update fidelity
    dataframe_2['ghz_fidelity'] = dataframe_2['ghz_fidelity'].mul(written_to_new)
    dataframe_1['ghz_fidelity'] = dataframe_1['ghz_fidelity'].mul(written_to_original)

    dataframe_1['ghz_fidelity'] = (dataframe_1['ghz_fidelity'] + dataframe_2['ghz_fidelity']) / corrected_written_to
    dataframe_1 = dataframe_1[(dataframe_1.T.applymap(lambda x: x != 0 and x is not None and not pd.isna(x))).any()]

    return dataframe_1


def add_decoherence_if_cut_off(qc: QuantumCircuit):
    if qc.cut_off_time < np.inf and not qc.cut_off_time_reached:
        waiting_time = qc.cut_off_time - qc.total_duration
        if waiting_time > 0:
            qc._increase_duration(waiting_time, [], involved_nodes=list(qc.nodes.keys()), check=False)
            qc.end_current_sub_circuit(total=True, duration=waiting_time, sub_circuit="Waiting", apply_decoherence=True)


def _additional_qc_arguments(**kwargs):
    additional_arguments = {
        'noise': True,
        'basis_transformation_noise': False,
        'thread_safe_printing': True,
        'no_single_qubit_error': True
    }
    kwargs.update(additional_arguments)
    return kwargs


def print_circuit_parameters(operational_args, circuit_args, varational_circuit_args):
    print('\n' + 80*'#')
    for args_name, args_values in locals().items():
        print("\n{}:\n-----------------------".format(args_name.capitalize()))
        pprint(args_values)
    print('\n' + 80*'#' + '\n')


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


def _save_superoperator_dataframe(fn, characteristics, succeeded, cut_off):
    # Adding confidence intervals to the superoperator
    print("Probability sum: {}".format(sum(succeeded['p'])))
    succeeded = _add_interval_to_dataframe(succeeded, characteristics)

    if fn:
        # Save pickle the characteristics file
        if os.path.exists(fn + '.pkl') and characteristics:
            characteristics_old = pickle.load(open(fn + '.pkl', 'rb'))
            [characteristics[key].extend(value) for key, value in characteristics_old.items() if key != 'index']
        pickle.dump(characteristics, file=open(fn + '.pkl', 'wb+')) if characteristics else None

        # Save the superoperators to a csv file
        for result, fn_add in zip([succeeded, cut_off], ['.csv', '_failed.csv']):
            fn_new = fn + fn_add
            existing_file = _open_existing_superoperator_file(fn_new)
            result = _combine_superoperator_dataframes(result, existing_file)
            if result is not None:
                result.to_csv(fn_new, sep=';')


def _add_interval_to_dataframe(dataframe, characteristics):
    if dataframe is not None:
        add_column_values(dataframe, ['int_stab_682', 'int_ghz_682', 'int_dur_682', 'dur_99'],
                          [str(confidence_interval(characteristics['stab_fid'])),
                           str(confidence_interval(characteristics['ghz_fid'])),
                           str(confidence_interval(characteristics['dur'])),
                           str(confidence_interval(characteristics['dur'], 0.98)[1])])
    return dataframe


def main_threaded(*, iterations, fn, **kwargs):
    # Run main method asynchronously with each worker getting an equal amount of iterations to run
    results = []
    workers = iterations if 0 < iterations < cpu_count() else cpu_count()
    thread_pool = Pool(workers)
    kwargs['iterations'] = iterations // workers
    pbar_2 = (tqdm(total=kwargs['iterations'] * workers) if kwargs.get('progress_bar') else None)

    for i in range(workers):
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
        pbar_2.update(kwargs['iterations']) if pbar_2 else None
    thread_pool.close()
    pbar_2.close() if pbar_2 else None

    print(*print_lines_results)

    # Save superoperator dataframe to csv if exists and requested by user
    _save_superoperator_dataframe(fn, tot_characteristics, succeeded, cut_off)


def main_series(fn, **kwargs):
    pbar_2 = tqdm(total=kwargs['iterations']) if kwargs.get('progress_bar') else None
    (succeeded, cut_off), print_lines, characteristics = main(pbar_2=pbar_2, **kwargs)
    print(*print_lines)

    # Save the superoperator to the according csv files (options: normal, cut-off)
    _save_superoperator_dataframe(fn, characteristics, succeeded, cut_off)


def main(*, iterations, protocol, stabilizer_type, threaded=False, gate_duration_file=None,
         color=False, draw_circuit=True, save_latex_pdf=False, to_console=False, pbar_2=None, **kwargs):
    supop_dataframe_failed = None
    supop_dataframe_succeed = None
    total_print_lines = []
    characteristics = {'dur': [], 'stab_fid': [], 'ghz_fid': []}

    # Progress bar initialisation
    pbar = None
    if pbar_2:
        # Second bar not working properly within PyCharm. Uncomment when using in normal terminal
        pass
        #pbar = tqdm(total=100, position=1, desc='Current circuit simulation')

    # Get the QuantumCircuit object corresponding to the protocol and the protocol method by its name
    kwargs = _additional_qc_arguments(**kwargs)
    qc = stab_protocols.create_quantum_circuit(protocol, pbar, **kwargs)
    protocol_method = getattr(stab_protocols, protocol)

    if threaded:
        set_gate_durations_from_file(gate_duration_file)

    # Run iterations of the protocol
    for iter in range(iterations):
        pbar.reset() if pbar else None
        if pbar_2 is not None:
            pbar_2.update(1) if pbar_2 else None
        elif not kwargs['progress_bar']:
            print(">>> At iteration {}/{}.".format(iter + 1, iterations), end='\r', flush=True)

        _init_random_seed(worker=threading.get_ident(), iteration=iter)

        # Run the user requested protocol
        operation = CZ_gate if stabilizer_type == "Z" else CNOT_gate
        superoperator_qubits_list = protocol_method(qc, operation=operation)
        qc.end_current_sub_circuit(total=True)
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

        if ((not qc.cut_off_time_reached and qc.ghz_fidelity is None) or (qc.cut_off_time_reached and qc.ghz_fidelity)
            or round(sum(dataframe['p']), 10) != 1.0):
            print("Warning: Superoperator calculation was corrupted. Please check the protocol.", file=sys.stderr)
            qc.draw_circuit(color_nodes=True)
            qc.print()

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
        total_print_lines.append("\nStab fidelity: {}".format(supop_dataframe.iloc[0, 0])) if draw_circuit else None
        total_print_lines.append("\nGHZ fidelity: {} ".format(qc.ghz_fidelity)) if draw_circuit else None
        total_print_lines.append("\nTotal circuit duration: {} s".format(qc.total_duration)) if draw_circuit else None
        qc.reset()

    pbar_2.close() if pbar_2 else None
    pbar.close() if pbar is not None else None
    return (supop_dataframe_succeed, supop_dataframe_failed), total_print_lines, characteristics


def run_for_arguments(operational_args, circuit_args, var_circuit_args, **kwargs):
    filenames = []
    fn = None
    cut_off_dataframe = _get_cut_off_dataframe(operational_args['cut_off_file'])

    # Loop over command line arguments
    for run in it.product(*(it.product([key], var_circuit_args[key]) for key in var_circuit_args.keys())):
        run_dict = dict(run)

        # Set run_dict values based on circuit arguments
        run_dict['lde_success'] = run_dict['lde_success'] if circuit_args['probabilistic'] else 0
        run_dict['fixed_lde_attempts'] = run_dict['fixed_lde_attempts'] if run_dict['pulse_duration'] > 0 else 0
        run_dict['pm'] = (run_dict['pg'] if circuit_args['pm_equals_pg'] else run_dict['pm'])
        run_dict['protocol'] = (run_dict['protocol'] + "_swap" if circuit_args['use_swap_gates']
                                else run_dict['protocol'])
        run_dict['cut_off_time'] = _get_cut_off_time(cut_off_dataframe, **run_dict, **circuit_args)

        if operational_args['csv_filename']:
            # Create parameter specific filename
            node = {2: 'Pur', 0.021: 'NatAb', 0: 'Ideal'}
            fn = create_file_name(operational_args['csv_filename'], dec=circuit_args['decoherence'],
                                  prob=circuit_args['probabilistic'], node=node[circuit_args['T1_lde']],
                                  decoupling=run_dict['pulse_duration'], **run_dict)
            filenames.append(fn)

            # Check if parameter settings has not yet been evaluated, else skip
            if not operational_args['force_run'] and fn is not None and os.path.exists(fn + ".csv"):
                data = pd.read_csv(fn + '.csv', sep=";", float_precision='round_trip')
                res_iterations = int(circuit_args['iterations'] - data.loc[0, 'written_to'])
                # iterations within 1% margin
                if not circuit_args['probabilistic'] or circuit_args['iterations'] * 0.01 > res_iterations:
                    print("Skipping circuit for file '{}', since it already exists.".format(fn))
                    continue
                else:
                    print("File found with too less iterations. Running for {} iterations\n".format(res_iterations))
                    circuit_args['iterations'] = res_iterations

        print("\nRunning {} iteration(s) with values for the variational arguments:".format(circuit_args['iterations']))
        pprint({**run_dict})

        if operational_args['threaded']:
            main_threaded(fn=fn, **operational_args, **run_dict, **circuit_args)
        else:
            main_series(fn=fn, **operational_args, **run_dict, **circuit_args)

    return filenames


if __name__ == "__main__":
    parser = compose_parser()
    args = vars(parser.parse_args())
    args = additional_parsing_of_arguments(**args)

    grouped_arguments = group_arguments(parser, **args)
    print_signature()
    print_circuit_parameters(*grouped_arguments)

    # Loop over all possible combinations of the user determined parameters
    run_for_arguments(*grouped_arguments, **args)
