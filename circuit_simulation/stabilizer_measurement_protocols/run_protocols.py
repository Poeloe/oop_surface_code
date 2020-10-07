import os
import sys
sys.path.insert(1, os.path.abspath(os.getcwd()))
from pprint import pprint
from multiprocessing import Pool, cpu_count
import threading
import pickle
import re
import pandas as pd
from circuit_simulation.stabilizer_measurement_protocols.stabilizer_measurement_protocols import *
from circuit_simulation.stabilizer_measurement_protocols.argument_parsing import compose_parser


def _init_random_seed(timestamp=None, worker=0, iteration=0):
    if timestamp is None:
        timestamp = time.time()
    seed = int("{:.0f}".format(timestamp * 10 ** 7) + str(worker) + str(iteration))
    random.seed(float(seed))
    return seed


def _combine_multiple_csv_files(filenames, cut_off=False, delete=False):
    for filename in filenames:
        csv_dir = os.path.dirname(os.path.abspath(filename))
        original_data_frame = None
        plain_file_name = os.path.split(filename)[1]
        handle_failed = "(?<!failed)" if not cut_off else "(?<=failed)"
        regex_pattern = re.compile('^' + plain_file_name + '_.*' + handle_failed + "\.csv$")
        plain_file_name = plain_file_name if not cut_off else plain_file_name + "_failed"
        final_file_name = os.path.join(csv_dir, "combined_" + plain_file_name + ".csv")
        if os.path.exists(final_file_name):
            original_data_frame = pd.read_csv(final_file_name, sep=';', index_col=[0, 1])

        for i, file in enumerate(os.listdir(csv_dir)):
            if regex_pattern.fullmatch(file):
                data_frame = pd.read_csv(os.path.join(csv_dir, file), sep=';', index_col=[0, 1])
                if original_data_frame is None:
                    original_data_frame = data_frame
                else:
                    original_data_frame = original_data_frame.add(data_frame, axis=0, fill_value=0)
                    original_data_frame = original_data_frame.div(2, axis=0)
                    if 'written_to' in original_data_frame:
                        original_data_frame.iloc[0, original_data_frame.columns.get_loc("written_to")] *= 2
                if delete:
                    os.remove(os.path.join(csv_dir, file))

        if original_data_frame is not None:
            original_data_frame.to_csv(final_file_name, sep=';')


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
    kwargs.update(lkt_1q=lkt_1q, lkt_2q=lkt_2q)
    kwargs.pop('i')
    kwargs.pop('pbar')

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


def main(i, it, protocol, stab_type, color, ltsv, sv, pg, pm, pm_1, pn, dec, p_bell, bell_dur, meas_dur, time_step,
         lkt_1q, lkt_2q, prb, fn, print_mode, draw, to_console, swap, threaded=False, gate_duration_file=None,
         pbar=None):

    if i == 0:
        _print_circuit_parameters(**locals())

    if threaded and fn:
        fn += ("_" + str(threading.get_ident()))
        _init_random_seed(worker=threading.get_ident(), iteration=it)

    if print_mode:
        return []

    gate = CZ_gate if stab_type == "Z" else CNOT_gate

    if protocol == "monolithic":
        return monolithic(gate, pg, pm, pm_1, color, bell_dur, meas_dur, time_step, lkt_1q, lkt_2q, ltsv, sv, fn, pbar,
                          draw, to_console)
    elif protocol == "expedient":
        if swap:
            return expedient_swap(gate, pg, pm, pm_1, pn, color, dec, p_bell, bell_dur, meas_dur, time_step, prb,
                                  lkt_1q, lkt_2q, ltsv, sv, fn, pbar, draw, to_console)
        return expedient(gate, pg, pm, pm_1, pn, color, dec, p_bell, bell_dur, meas_dur, time_step, prb, lkt_1q,
                         lkt_2q, ltsv, sv, fn, pbar, draw, to_console)
    elif protocol == "stringent":
        if swap:
            return stringent_swap(gate, pg, pm, pm_1, pn, color, dec, p_bell, bell_dur, meas_dur, time_step, prb,
                                  lkt_1q, lkt_2q, ltsv, sv, fn, pbar, draw, to_console)
        return stringent(gate, pg, pm, pm_1, pn, color, dec, p_bell, bell_dur, meas_dur, time_step, prb, lkt_1q,
                         lkt_2q, ltsv, sv, fn, pbar, draw, to_console)


if __name__ == "__main__":
    parser = compose_parser()

    args = vars(parser.parse_args())
    it = args.pop('iterations')
    protocols = args.pop('protocol')
    stab_type = args.pop('stabilizer_type')
    color = args.pop('color')
    dec = args.pop('decoherence')
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
    filenames = args.pop('csv_filename')
    threaded = args.pop('threaded')
    print_mode = args.pop('print_run_order')
    prb = args.pop('probabilistic')
    lkt_1q = args.pop('lookup_table_single_qubit_gates')
    lkt_2q = args.pop('lookup_table_two_qubit_gates')
    draw = args.pop('draw_circuit')
    to_console = args.pop('to_console')
    swap = args.pop('use_swap_gates')
    gate_duration_file = args.pop('gate_duration_file')
    progress_bar = args.pop('no_progress_bar')

    file_dir = os.path.dirname(__file__)
    # THIS IS NOT GENERIC, will error when directories are moved or renamed
    look_up_table_dir = os.path.join(file_dir, '../gates', 'gate_lookup_tables')

    sv = True if filenames else False

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

    if gate_duration_file is not None:
        if os.path.exists(gate_duration_file):
            set_gate_durations_from_file(gate_duration_file)
        else:
            raise ValueError("Cannot find file to set gate durations with. File path: {}".format(gate_duration_file))

    if progress_bar:
        from tqdm import tqdm
        pbar = tqdm(total=100)
    else:
        pbar = None

    if threaded:
        workers = it if 1 < it < cpu_count() else cpu_count()
        thread_pool = Pool(workers)
        results = []
        if progress_bar:
            pbar = tqdm(total=it)

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
                                                        dec, p_bell, bell_dur, meas_dur, time_step, lkt_1q, lkt_2q,
                                                        prb, fn, print_mode, draw, to_console, swap, threaded,
                                                        gate_duration_file)))
                        else:
                            print(*main(i, it, protocol, stab_type, color, ltsv, sv, pg, pm, pm_1, pn, dec, p_bell,
                                        bell_dur, meas_dur, time_step, lkt_1q, lkt_2q, prb, fn, print_mode, draw,
                                        to_console, swap, gate_duration_file=gate_duration_file, pbar=pbar))
                            print("\nFinished iteration {} of the {}\n".format(i+1, it))
                            if progress_bar:
                                pbar.reset()
                        filename_count += 1

    if threaded:
        print_results = []
        for res in results:
            print_results.extend(res.get())
            if pbar is not None:
                pbar.update(1)
        if filenames:
            _combine_multiple_csv_files(filenames, delete=False)
            _combine_multiple_csv_files(filenames, cut_off=True, delete=False)

        print(*print_results)
        thread_pool.close()
