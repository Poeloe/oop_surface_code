from circuit_simulation.stabilizer_measurement_protocols.argument_parsing import compose_parser, group_arguments
from run_threshold import add_arguments
from circuit_simulation.stabilizer_measurement_protocols.run_protocols import run_for_arguments, \
    additional_parsing_of_arguments, print_circuit_parameters
from oopsc.threshold.sim import sim_thresholds
import re
import os
import pandas as pd
from pprint import pprint
from copy import copy


def determine_superoperators(superoperator_filenames, args):
    multiple_superoperators = args.get('protocol') in ['weight_2_4']
    primary_superoperators = []
    primary_superoperators_failed = []
    secondary_superoperators = []
    secondary_superoperators_failed = []

    for filename in superoperator_filenames:
        if 'secondary' in filename:
            secondary_superoperators.append(filename)
            secondary_superoperators.append(filename) if 'timeinf' not in filename else None
        else:
            primary_superoperators.append(filename)
            primary_superoperators_failed.append(filename + "_failed") if 'timeinf' not in filename else None

    args['superoperator_filenames'] = primary_superoperators
    args['superoperator_filenames_failed'] = primary_superoperators_failed if 'timeinf' not in filename else None
    args['superoperator_filenames_additional'] = secondary_superoperators if multiple_superoperators else None
    args['superoperator_filenames_additional_failed'] = (secondary_superoperators_failed if multiple_superoperators
                                                         and 'timeinf' not in filename else None)

    if primary_superoperators_failed:
        args['GHZ_successes'] = [0.99]

    args['folder'] = re.sub('pg[0-9.]*_', '', primary_superoperators[0])
    args['save_result'] = True

    return args


def determine_args_by_file(surface_args, var_circuit_args):
    folder = surface_args['folder']
    lattices = copy(surface_args['lattices'])
    pg = var_circuit_args['pg']
    ghz = 0.99 if not 'timeinf' in folder else 1.1
    res_iters = []

    if os.path.exists(folder):
        for file in os.listdir(folder):
            data = pd.read_csv(os.path.join(folder, file), index_col=['L', 'p', 'GHZ_success'])
            for lat in surface_args['lattices']:
                res_it = [int(surface_args['iters'] - n) for n in data.loc[pd.IndexSlice[lat, pg, ghz], 'N']]
                res_iters.extend(res_it)
                if all([surface_args['iters'] * 0.01 > it for it in res_it]):
                    lattices.remove(lat)

    if not lattices:
        pprint(data)
        print("\nAll surface code simulations have already been performed. Exiting Surface Code simulations")
        exit(1)

    surface_args['lattices'] = lattices
    surface_args['iters'] = max(res_iters)

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
    surface_code_args = determine_args_by_file(surface_code_args, grouped_arguments[2])

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
