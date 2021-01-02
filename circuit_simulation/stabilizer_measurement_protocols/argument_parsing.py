import argparse
import numpy as np


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as v:
            parser.parse_args([argument for argument in v.read().split() if "#" not in argument], namespace)


def group_arguments(parser, **kwargs):
    opp_args = {ac.dest: kwargs[ac.dest] for ac in parser._actions
                if ac.container.description == 'Operation arguments' and ac.dest != 'argument_file'}
    circuit_args = {ac.dest: kwargs[ac.dest] for ac in parser._actions
                    if ac.container.description == 'Circuit arguments'}
    var_circuit_args = {ac.dest: kwargs[ac.dest] for ac in parser._actions
                        if ac.container.description == 'Variational circuit arguments'}

    return opp_args, circuit_args, var_circuit_args


def compose_parser():
    parser = argparse.ArgumentParser(prog='Stabilizer measurement protocol simulations')
    opp_arg = parser.add_argument_group(description="Operation arguments")
    circuit_arg = parser.add_argument_group(description="Circuit arguments")
    var_circuit_arg = parser.add_argument_group(description="Variational circuit arguments")

    # Operational Arguments
    opp_arg.add_argument('-c',
                         '--color',
                         help='Specifies if the console output should display color. Optional',
                         required=False,
                         action='store_true')
    opp_arg.add_argument('-ltsv',
                         '--save_latex_pdf',
                         help='If given, a pdf containing a drawing of the noisy circuit in latex will be saved to the '
                              '`circuit_pdfs` folder. Optional',
                         required=False,
                         action='store_true')
    opp_arg.add_argument('-fn',
                         '--csv_filename',
                         required=False,
                         type=str,
                         default=None,
                         help='Give the file name of the csv file that will be saved.')
    opp_arg.add_argument('-cp',
                         '--cp_path',
                         required=False,
                         type=str,
                         default=None,
                         help='Give the path the csv file should be copied to (Cluster runs).')
    opp_arg.add_argument("-tr",
                         "--threaded",
                         help="Use when the program should run in multi-threaded mode. Optional",
                         required=False,
                         action="store_true")
    opp_arg.add_argument("--to_console",
                         help="Print the superoperator results to the console.",
                         required=False,
                         action="store_true")
    opp_arg.add_argument("-draw",
                         "--draw_circuit",
                         help="Print a drawing of the circuit to the console",
                         required=False,
                         action="store_true")
    opp_arg.add_argument("-lkt_1q",
                         "--single_qubit_gate_lookup",
                         help="Name of a .pkl single-qubit gate lookup file.",
                         required=False,
                         type=str,
                         default=None)
    opp_arg.add_argument("-lkt_2q",
                         "--two_qubit_gate_lookup",
                         help="Name of a .pkl two-qubit gate lookup file.",
                         required=False,
                         type=str,
                         default=None)
    opp_arg.add_argument("--argument_file",
                         help="loads values from a file instead of the command line",
                         type=open,
                         action=LoadFromFile)
    opp_arg.add_argument("--gate_duration_file",
                         help="Specify the path to the file that contains the gate duration times.",
                         type=str,
                         required=False)
    opp_arg.add_argument("--no_progress_bar",
                         help="Displays no progress bar for simulation.",
                         action='store_false')
    opp_arg.add_argument('-cut_file',
                         '--cut_off_file',
                         help='Specifies the file to load the cut-off time from for performing a stabilizer '
                              'measurement.',
                         type=str,
                         default=None)
    opp_arg.add_argument("-fr",
                         "--force_run",
                         help="Force simulation to run if file already exists",
                         required=False,
                         action="store_true")

    # Variational Circuit Arguments
    var_circuit_arg.add_argument('-p',
                                 '--protocols',
                                 help='Specifies which protocol should be used. - options: {'
                                      'monolithic/expedient/stringent}',
                                 nargs="*",
                                 choices=['monolithic', 'expedient', 'stringent', 'weight_2_4', 'weight_3',
                                          'dyn_prot_4_14_1', 'dyn_prot_4_22_1', 'bipartite_4',  'bipartite_6', 'plain',
                                          'dyn_prot_4_6_sym_1', 'dejmps_2_4_1', 'dejmps_2_6_1', 'dejmps_2_8_1',
                                          'dyn_prot_4_4_1', 'dyn_prot_3_4_1', 'dyn_prot_3_8_1'],
                                 type=str.lower,
                                 default=['monolithic'])
    var_circuit_arg.add_argument('-pg',
                                 '--pg',
                                 help='Specifies the amount of gate error present in the system',
                                 type=float,
                                 nargs="*",
                                 default=[0.006])
    var_circuit_arg.add_argument('-pm',
                                 '--pm',
                                 help='Specifies the amount of measurement error present in the system',
                                 type=float,
                                 nargs="*",
                                 default=[None])
    var_circuit_arg.add_argument('-pm_1',
                                 '--pm_1',
                                 help='The measurement error rate in case an 1-state is supposed to be measured',
                                 required=False,
                                 type=float,
                                 nargs="*",
                                 default=[None])
    var_circuit_arg.add_argument('-pn',
                                 '--pn',
                                 help='Specifies the amount of network error present in the system',
                                 type=float,
                                 nargs="*",
                                 default=[0.0])
    var_circuit_arg.add_argument('-p_bell',
                                 '--lde_success',
                                 help='Specifies the success probability of the creation of a Bell pair (if '
                                      'probabilistic).',
                                 type=float,
                                 nargs='*',
                                 default=[1.0])
    var_circuit_arg.add_argument("-lde",
                                 "--fixed_lde_attempts",
                                 help="Specify the amount of fixed lde attempts before a pulse is sent to the nuclear "
                                      "qubits.",
                                 type=int,
                                 nargs="*",
                                 default=[1000])
    var_circuit_arg.add_argument('-pulse_dur',
                                 '--pulse_duration',
                                 help='Specifies the duration of a pulse used in the pulse sequence. If no pulse '
                                      'sequence is present, this should NOT be specified.',
                                 type=float,
                                 nargs='*',
                                 default=[0])

    # Constant Circuit arguments
    circuit_arg.add_argument('-it',
                             '--iterations',
                             help='Specifies the number of iterations that should be done (use only in combination '
                                  'with '
                             '--prb)',
                             type=int,
                             default=1)
    circuit_arg.add_argument('-s',
                             '--stabilizer_type',
                             help='Specifies what the kind of stabilizer should be.',
                             choices=['Z', 'X'],
                             type=str.upper,
                             default='Z')
    circuit_arg.add_argument('-dec',
                             '--decoherence',
                             help='Specifies if decoherence is present in the system.',
                             required=False,
                             action='store_true')
    circuit_arg.add_argument('--pm_equals_pg',
                             help='Specify if measurement error equals the gate error. "-pm" will then be disregarded',
                             required=False,
                             action='store_true')
    circuit_arg.add_argument('-prb',
                             '--probabilistic',
                             help='Specifies if the processes in the protocol are probabilistic.',
                             required=False,
                             action='store_true')
    circuit_arg.add_argument('-m_dur',
                             '--measurement_duration',
                             help='Specifies the duration of a measurement operation.',
                             type=float,
                             default=0.)
    circuit_arg.add_argument('-b_dur',
                             '--lde_duration',
                             help='Specifies the duration of a measurement operation.',
                             type=float,
                             default=0.)
    circuit_arg.add_argument("-swap",
                             "--use_swap_gates",
                             help="A version of the protocol will be run that uses SWAP gates to ensure NV-center "
                                  "realism.",
                             required=False,
                             action="store_true")
    circuit_arg.add_argument("-no_swap",
                             "--noiseless_swap",
                             help="A version of the protocol will be run that uses SWAP gates to ensure NV-center "
                                  "realism.",
                             required=False,
                             action="store_true")
    circuit_arg.add_argument("-n_type",
                             "--network_noise_type",
                             help="Specify the network noise type. ",
                             type=int,
                             choices=[1, 0],
                             default=0)
    circuit_arg.add_argument('-T1ni',
                             '--T1_idle',
                             help='T1 relaxation time for a nuclear qubit.',
                             type=float,
                             default=300)
    circuit_arg.add_argument('-T2ni',
                             '--T2_idle',
                             help='T2 relaxation time for a nuclear qubit.',
                             type=float,
                             default=10)
    circuit_arg.add_argument('-T1nl',
                             '--T1_lde',
                             help='T1 relaxation time for a nuclear qubit while LDE is performed.',
                             type=float,
                             default=2)
    circuit_arg.add_argument('-T2nl',
                             '--T2_lde',
                             help='T2 relaxation time for a nuclear qubit while LDE is performed.',
                             type=float,
                             default=2)
    circuit_arg.add_argument('-T1ei',
                             '--T1_idle_electron',
                             help='T1 relaxation time for an electron qubit.',
                             type=float,
                             default=10000)
    circuit_arg.add_argument('-T2ei',
                             '--T2_idle_electron',
                             help='T2 relaxation time for an electron qubit.',
                             type=float,
                             default=1)
    circuit_arg.add_argument('-cut',
                             '--cut_off_time',
                             help='Specifies the cut-off time for performing a stabilizer measurement.',
                             type=float,
                             default=np.inf)
    return parser