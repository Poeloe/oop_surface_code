import argparse


class LoadFromFile(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        with values as v:
            parser.parse_args(v.read().split(), namespace)


def compose_parser():
    parser = argparse.ArgumentParser(prog='Stabilizer measurement protocol simulations')
    group = parser.add_mutually_exclusive_group()

    parser.add_argument('-it',
                        '--iterations',
                        help='Specifies the number of iterations that should be done (use only in combination with '
                             '--prb)',
                        type=int,
                        default=1)
    parser.add_argument('-p',
                        '--protocols',
                        help='Specifies which protocol should be used. - options: {monolithic/expedient/stringent}',
                        nargs="*",
                        choices=['monolithic', 'expedient', 'stringent', 'duo_structure', 'duo_structure_2', 'plain'],
                        type=str.lower,
                        default='monolithic')
    parser.add_argument('-s',
                        '--stabilizer_type',
                        help='Specifies what the kind of stabilizer should be.',
                        choices=['Z', 'X'],
                        type=str.upper,
                        default='Z')
    parser.add_argument('-dec',
                        '--decoherence',
                        help='Specifies if decoherence is present in the system.',
                        required=False,
                        action='store_true')
    parser.add_argument('-pg',
                        '--gate_error_probabilities',
                        help='Specifies the amount of gate error present in the system',
                        type=float,
                        nargs="*",
                        default=[0.006])
    group.add_argument('--pm_equals_pg',
                       help='Specify if measurement error equals the gate error. "-pm" will then be disregarded',
                       required=False,
                       action='store_true')
    group.add_argument('-pm',
                       '--meas_error_probabilities',
                       help='Specifies the amount of measurement error present in the system',
                       type=float,
                       nargs="*")
    parser.add_argument('-pm_1',
                        '--meas_error_probabilities_one_state',
                        help='The measurement error rate in case an 1-state is supposed to be measured',
                        required=False,
                        type=float,
                        nargs="*",
                        default=None)
    parser.add_argument('-pn',
                        '--network_error_probabilities',
                        help='Specifies the amount of network error present in the system',
                        type=float,
                        nargs="*",
                        default=[0.0])
    parser.add_argument('-p_bell',
                        '--lde_success',
                        help='Specifies the success probability of the creation of a Bell pair (if probabilistic).',
                        type=float,
                        default=1.0)
    parser.add_argument('-prb',
                        '--probabilistic',
                        help='Specifies if the processes in the protocol are probabilistic.',
                        required=False,
                        action='store_true')
    parser.add_argument('-m_dur',
                        '--measurement_duration',
                        help='Specifies the duration of a measurement operation.',
                        type=float,
                        default=0.)
    parser.add_argument('-b_dur',
                        '--lde_duration',
                        help='Specifies the duration of a measurement operation.',
                        type=float,
                        default=0.)
    parser.add_argument('-pulse_dur',
                        '--pulse_duration',
                        help='Specifies the duration of a pulse used in the pulse sequence. If no pulse sequence is '
                             'present, this should NOT be specified.',
                        type=float,
                        default=10e-3)
    parser.add_argument('-c',
                        '--color',
                        help='Specifies if the console output should display color. Optional',
                        required=False,
                        action='store_true')
    parser.add_argument('-ltsv',
                        '--save_latex_pdf',
                        help='If given, a pdf containing a drawing of the noisy circuit in latex will be saved to the '
                             '`circuit_pdfs` folder. Optional',
                        required=False,
                        action='store_true')
    parser.add_argument('-fn',
                        '--csv_filename',
                        required=False,
                        type=str,
                        default=None,
                        help='Give the file name of the csv file that will be saved.')
    parser.add_argument('-cp',
                        '--cp_path',
                        required=False,
                        type=str,
                        default=None,
                        help='Give the path the csv file should be copied to (Cluster runs).')
    parser.add_argument("-tr",
                        "--threaded",
                        help="Use when the program should run in multi-threaded mode. Optional",
                        required=False,
                        action="store_true")
    parser.add_argument("--to_console",
                        help="Print the superoperator results to the console.",
                        required=False,
                        action="store_true")
    parser.add_argument("-draw",
                        "--draw_circuit",
                        help="Print a drawing of the circuit to the console",
                        required=False,
                        action="store_true")
    parser.add_argument("--print_run_order",
                        help="When added, the program will only print out the run order for the typed command. This can"
                             "be useful for debugging or file naming purposes",
                        required=False,
                        action="store_true")
    parser.add_argument("-lkt_1q",
                        "--single_qubit_gate_lookup",
                        help="Name of a .pkl single-qubit gate lookup file.",
                        required=False,
                        type=str,
                        default=None)
    parser.add_argument("-lkt_2q",
                        "--two_qubit_gate_lookup",
                        help="Name of a .pkl two-qubit gate lookup file.",
                        required=False,
                        type=str,
                        default=None)
    parser.add_argument("-swap",
                        "--use_swap_gates",
                        help="A version of the protocol will be run that uses SWAP gates to ensure NV-center realism.",
                        required=False,
                        action="store_true")
    parser.add_argument("--argument_file",
                        help="loads values from a file instead of the command line",
                        type=open,
                        action=LoadFromFile)
    parser.add_argument("--gate_duration_file",
                        help="Specify the path to the file that contains the gate duration times.",
                        type=str,
                        required=False)
    parser.add_argument("--no_progress_bar",
                        help="Displays no progress bar for simulation.",
                        action='store_false')
    parser.add_argument("-lde",
                        "--fixed_lde_attempts",
                        help="Specify the amount of fixed lde attempts before a pulse is sent to the nuclear qubits.",
                        type=int,
                        nargs="*",
                        default=[1000])
    parser.add_argument("-n_type",
                        "--network_noise_type",
                        help="Specify the network noise type. ",
                        type=int,
                        choices=[1, 0],
                        default=0)
    parser.add_argument('-T1ni',
                        '--T1_idle',
                        help='Specifies the duration of a pulse used in the pulse sequence. If no pulse sequence is '
                             'present, this should NOT be specified.',
                        type=float,
                        default=300)
    parser.add_argument('-T2ni',
                        '--T2_idle',
                        help='Specifies the duration of a pulse used in the pulse sequence. If no pulse sequence is '
                             'present, this should NOT be specified.',
                        type=float,
                        default=10)
    parser.add_argument('-T1nl',
                        '--T1_lde',
                        help='Specifies the duration of a pulse used in the pulse sequence. If no pulse sequence is '
                             'present, this should NOT be specified.',
                        type=float,
                        default=2)
    parser.add_argument('-T2nl',
                        '--T2_lde',
                        help='Specifies the duration of a pulse used in the pulse sequence. If no pulse sequence is '
                             'present, this should NOT be specified.',
                        type=float,
                        default=2)
    parser.add_argument('-T1ei',
                        '--T1_idle_electron',
                        help='Specifies the duration of a pulse used in the pulse sequence. If no pulse sequence is '
                             'present, this should NOT be specified.',
                        type=float,
                        default=10000)
    parser.add_argument('-T2ei',
                        '--T2_idle_electron',
                        help='Specifies the duration of a pulse used in the pulse sequence. If no pulse sequence is '
                             'present, this should NOT be specified.',
                        type=float,
                        default=1)

    return parser
