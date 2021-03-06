from circuit_simulation.stabilizer_measurement_protocols.argument_parsing import compose_parser, group_arguments
from run_threshold import add_arguments
from circuit_simulation.stabilizer_measurement_protocols.run_protocols import run_for_arguments, \
    additional_parsing_of_arguments, print_circuit_parameters
from oopsc.threshold.sim import sim_thresholds
from itertools import product
from collections import defaultdict
import os
import numpy as np
import pandas as pd
from pprint import pprint
from copy import copy


def create_index_slice(df, column, begin=None, end=None):
    idx = tuple()
    column = [df.index.names.index(value) for value in column]
    for i in range(len(df.index.names)):
        if i in column:
            index = column.index(i)
            cur_begin = begin[index] if begin is not None else None
            cur_end = end[index] if end is not None else None
            idx += (slice(cur_begin, cur_end, None),)
        else:
            idx += (slice(None, None, None),)
    return idx


def determine_superoperators(superoperator_filenames, args):
    primary_superoperators = []
    primary_superoperators_failed = []
    secondary_superoperators = []
    secondary_superoperators_failed = []

    for filename in superoperator_filenames:
        if 'secondary' in filename:
            secondary_superoperators.append(filename)
            secondary_superoperators_failed.append(filename + "_failed") if 'time' in filename else None
        else:
            primary_superoperators.append(filename)
            primary_superoperators_failed.append(filename + "_failed") if 'time' in filename else None

    args['superoperator_filenames'] = primary_superoperators
    args['superoperator_filenames_failed'] = primary_superoperators_failed if primary_superoperators_failed else None
    args['superoperator_filenames_additional'] = secondary_superoperators if secondary_superoperators else None
    args['superoperator_filenames_additional_failed'] = (secondary_superoperators_failed
                                                         if secondary_superoperators_failed else None)

    if primary_superoperators_failed:
        args['GHZ_successes'] = [0.99]

    args['folder'] = os.path.join(os.path.dirname(primary_superoperators[0]), "threshold_sim")
    args['save_result'] = True

    return args


def determine_lattice_evaluation_by_result(surface_args, opp_args, circuit_args, var_circuit_args):
    folder = surface_args['folder']
    var_circuit_args['GHZ_success'] = [0.99 if cut != np.inf else 1.1 for cut in var_circuit_args['cut_off_time']]
    var_circuit_args['node'] = ['Purified'] if circuit_args['T1_lde'] == 2 else ["Natural Abundance"]
    var_circuit_args['p_bell_success'] = var_circuit_args['lde_success'] if circuit_args['probabilistic'] else [1]
    var_circuit_args['protocol_name'] = set([p.strip("_secondary") + "_swap" if circuit_args['use_swap_gates'] else
                                            p.strip("_secondary") for p in var_circuit_args['protocol']])
    res_iters = defaultdict(int)
    parameters = {}

    if os.path.exists(folder):
        for file in os.listdir(folder):
            data = pd.read_csv(os.path.join(folder, file), float_precision='round_trip')
            data.replace({'None': None}, inplace=True)
            parameters = {col: var_circuit_args[col] for col in data if col not in ['L', 'N', 'success']}
            data.set_index(['L'] + list(parameters.keys()), inplace=True)
            data.sort_index()
            for index in product(*[surface_args['lattices'], *parameters.values()]):
                res_iters[index[0]] += (1 if index in data.index and
                                        (data.loc[index, "N"] * 1.05) >= surface_args['iters'] else 0)

    for L, count in res_iters.items():
        if count == len(list(product(*parameters.values()))):
            surface_args['lattices'].remove(L)
            print("\n[INFO] Skipping simulations for L={} since it has already run for all parameters".format(L))

    # If there are no lattices left to evaluate, the program can exit
    if not surface_args['lattices']:
        pprint(data)
        print("\nAll surface code simulations have already been performed. Exiting Surface Code simulations")
        exit(1)

    return surface_args


if __name__ == "__main__":
    parser = compose_parser()
    add_arguments(parser)
    args = vars(parser.parse_args())

    # Run circuit simulation if superoperator file does not yet exist
    print('\n #############################################')
    print(' ############ CIRCUIT SIMULATIONS ############')
    print(' #############################################\n')
    circuit_sim_args = {action.dest: args[action.dest] for action in compose_parser()._actions if action.dest != 'help'}
    circuit_sim_args = additional_parsing_of_arguments(**circuit_sim_args)
    grouped_arguments = group_arguments(compose_parser(), **circuit_sim_args)
    print_circuit_parameters(*grouped_arguments)
    superoperator_filenames = run_for_arguments(*grouped_arguments, **circuit_sim_args)
    print('\n -----------------------------------------------------------')

    # Run surface code simulations
    print('\n ##################################################')
    print(' ############ SURFACE CODE SIMULATIONS ############')
    print(' ##################################################\n')
    surface_code_args = {action.dest: args[action.dest] for action in add_arguments()._actions if action.dest != 'help'}
    surface_code_args = determine_superoperators(superoperator_filenames, surface_code_args)
    surface_code_args = determine_lattice_evaluation_by_result(surface_code_args, *grouped_arguments)

    decoder = surface_code_args.pop("decoder")

    decoders = __import__("oopsc.decoder", fromlist=[decoder])
    decode = getattr(decoders, decoder)

    decoder_names = {
        "mwpm": "minimum weight perfect matching (blossom5)",
        "uf": "union-find",
        "uf_uwg": "union-find non weighted growth",
        "ufbb": "union-find balanced bloom"
    }
    decoder_name = decoder_names[decoder] if decoder in decoder_names else decoder
    print(f"{'_' * 75}\n\ndecoder type: " + decoder_name)

    sim_thresholds(decode, **surface_code_args)
