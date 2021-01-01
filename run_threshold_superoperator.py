'''
2020 Mark Shui Hu, QuTech

www.github.com/watermarkhu/oop_surface_code
_____________________________________________

'''
from circuit_simulation.stabilizer_measurement_protocols.argument_parsing import compose_parser
from run_threshold import add_arguments
from circuit_simulation.stabilizer_measurement_protocols.run_protocols import run_for_arguments, \
    additional_parsing_of_arguments, print_circuit_parameters
import numpy as np
from oopsc.threshold.sim import sim_thresholds


def determine_superoperators(superoperator_filenames, args):
    multiple_superoperators = args.get('protocol') in ['weight_2_4']
    primary_superoperators = []
    primary_superoperators_failed = []
    secondary_superoperators = []
    secondary_superoperators_failed = []

    for filename in superoperator_filenames:
        if 'secondary' in filename:
            secondary_superoperators.append(filename)
            secondary_superoperators.append(filename) if 'cutoff_inf' not in filename else None
        else:
            primary_superoperators.append(filename)
            primary_superoperators_failed.append(filename + "_failed") if 'cutoff_inf' not in filename else None

    args['superoperator_filenames'] = primary_superoperators
    args['superoperator_filenames_failed'] = primary_superoperators_failed if 'cutoff_inf' not in filename else None
    args['superoperator_filenames_additional'] = secondary_superoperators if multiple_superoperators else None
    args['superoperator_filenames_additional_failed'] = (secondary_superoperators_failed if multiple_superoperators
                                                         and 'cutoff_inf' not in filename else None)

    return args


if __name__ == "__main__":
    parser = compose_parser()
    add_arguments(parser)
    args = vars(parser.parse_args())

    # Run circuit simulation if superoperator file does not yet exist
    print('\n #############################################')
    print(' ############ CIRCUIT SIMULATIONS ############')
    print(' #############################################\n')
    circuit_args = {action.dest: args[action.dest] for action in compose_parser()._actions if action.dest != 'help'}
    additional_parsing_of_arguments(**circuit_args)
    print_circuit_parameters(**circuit_args)
    superoperator_filenames = run_for_arguments(**circuit_args)
    print('\n -----------------------------------------------------------')

    # Run surface code simulations
    print('\n ##################################################')
    print(' ############ SURFACE CODE SIMULATIONS ############')
    print(' ##################################################')
    surface_code_args = {action.dest: args[action.dest] for action in add_arguments()._actions if action.dest != 'help'}
    surface_code_args = determine_superoperators(superoperator_filenames, surface_code_args)

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
