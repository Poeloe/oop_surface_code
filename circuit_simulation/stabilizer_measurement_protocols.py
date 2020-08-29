import os
import sys
import argparse
sys.path.insert(1, os.path.abspath(os.getcwd()))
from circuit_simulation.circuit_simulator import *
from multiprocessing import Pool
import time


def monolithic(operation, pg, pm, color, save_latex_pdf, save_csv, csv_file_name):
    qc = QuantumCircuit(8, 2, noise=True, pg=pg, pm=pm, basis_transformation_noise=True, network_noise_type=1,
                        thread_safe_printing=True)
    qc.add_top_qubit(ket_p, p_prep=pm)
    qc.apply_2_qubit_gate(operation, 0, 1)
    qc.apply_2_qubit_gate(operation, 0, 3)
    qc.apply_2_qubit_gate(operation, 0, 5)
    qc.apply_2_qubit_gate(operation, 0, 7)
    qc.measure_first_N_qubits(1)

    qc.draw_circuit(not color)
    if save_latex_pdf:
        qc.draw_circuit_latex()
    qc.get_superoperator([0, 2, 4, 6], operation.representation, no_color=(not color), to_csv=save_csv,
                         csv_file_name=csv_file_name, stabilizer_protocol=True)

    return qc._print_lines


def expedient(operation, pg, pm, pn, color, save_latex_pdf, save_csv, csv_file_name):
    qc = QuantumCircuit(8, 2, noise=True, basis_transformation_noise=False, pg=pg, pm=pm, pn=pn, network_noise_type=1,
                        thread_safe_printing=True)

    # Noisy ancilla Bell pair now between 0 and 1
    qc.create_bell_pairs_top(1, new_qubit=True)
    qc.double_selection(Z_gate, new_qubit=True)
    qc.double_selection(X_gate)

    # New noisy ancilla Bell pair is now between 0 and 1, old ancilla Bell pair now between 2 and 3
    qc.create_bell_pairs_top(1, new_qubit=True)
    qc.double_selection(Z_gate, new_qubit=True)
    qc.double_selection(X_gate)

    # Now entanglement between ancilla 0 and 3 is made
    qc.single_dot(Z_gate, 2, 5)
    qc.single_dot(Z_gate, 2, 5)

    # And finally the entanglement between ancilla 1 and 2 is made, now all ancilla's are entangled
    qc.single_dot(Z_gate, 3, 4)
    qc.single_dot(Z_gate, 3, 4)

    qc.apply_2_qubit_gate(operation, 0, 4)
    qc.apply_2_qubit_gate(operation, 1, 6)
    qc.apply_2_qubit_gate(operation, 2, 8)
    qc.apply_2_qubit_gate(operation, 3, 10)

    qc.measure_first_N_qubits(4)

    qc.draw_circuit(no_color=not color)
    if save_latex_pdf:
        qc.draw_circuit_latex()
    qc.get_superoperator([0, 2, 4, 6], operation.representation, no_color=(not color), to_csv=save_csv,
                         csv_file_name=csv_file_name, stabilizer_protocol=True)

    return qc._print_lines


def stringent(operation, pg, pm, pn, color, save_latex_pdf, save_csv, csv_file_name):
    qc = QuantumCircuit(8, 2, noise=True, basis_transformation_noise=False, pg=pg, pm=pm, pn=pn, network_noise_type=1,
                        thread_safe_printing=True)

    # Noisy ancilla Bell pair between 0 and 1
    qc.create_bell_pairs_top(1, new_qubit=True)
    qc.double_selection(Z_gate, new_qubit=True)
    qc.double_selection(X_gate)
    qc.double_dot(Z_gate, 2, 3)
    qc.double_dot(X_gate, 2, 3)

    # New noisy ancilla Bell pair is now between 0 and 1, old ancilla Bell pair now between 2 and 3
    qc.create_bell_pairs_top(1, new_qubit=True)
    qc.double_selection(Z_gate, new_qubit=True)
    qc.double_selection(X_gate)
    qc.double_dot(Z_gate, 2, 3)
    qc.double_dot(X_gate, 2, 3)

    # Now entanglement between ancilla 0 and 3 is made
    qc.double_dot(Z_gate, 2, 5)
    qc.double_dot(Z_gate, 2, 5)

    # And finally the entanglement between ancilla 1 and 2 is made, now all ancilla's are entangled
    qc.double_dot(Z_gate, 3, 4)
    qc.double_dot(Z_gate, 3, 4)

    qc.apply_2_qubit_gate(operation, 0, 4)
    qc.apply_2_qubit_gate(operation, 1, 6)
    qc.apply_2_qubit_gate(operation, 2, 8)
    qc.apply_2_qubit_gate(operation, 3, 10)

    qc.measure_first_N_qubits(4)

    qc.draw_circuit(no_color=not color)
    if save_latex_pdf:
        qc.draw_circuit_latex()
    qc.get_superoperator([0, 2, 4, 6], operation.representation, no_color=(not color), to_csv=save_csv,
                         csv_file_name=csv_file_name, stabilizer_protocol=True)

    return qc._print_lines


def compose_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--protocol',
                        help='Specifies which protocol should be used. - options: {monolithic/expedient/stringent}',
                        nargs="*",
                        default='monolithic')
    parser.add_argument('-s',
                        '--stabilizer_type',
                        help='Specifies what the kind of stabilizer should be. - options: {Z/X}',
                        default='Z')
    parser.add_argument('-pg',
                        '--gate_error_probability',
                        help='Specifies the amount of gate error present in the system',
                        type=float,
                        nargs="*",
                        default=[0.006])
    parser.add_argument('--pm_equals_pg',
                        help='Specify if measurement error equals the gate error. "-pm" will then be disregarded',
                        required=False,
                        action='store_true')
    parser.add_argument('-pm',
                        '--measurement_error_probability',
                        help='Specifies the amount of measurement error present in the system',
                        type=float,
                        nargs="*",
                        default=[0.006])
    parser.add_argument('-pn',
                        '--network_error_probability',
                        help='Specifies the amount of network error present in the system',
                        type=float,
                        nargs="*",
                        default=[0.0])
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
    parser.add_argument('-sv',
                        '--save_csv',
                        help='Specifies if a csv file of the superoperator should be saved. Optional',
                        required=False,
                        action='store_true')
    parser.add_argument('-fn',
                        '--csv_filename',
                        required=False,
                        nargs="*",
                        help='Give the file name of the csv file that will be saved.')
    parser.add_argument("-tr",
                        "--threaded",
                        help="Use when the program should run in multi-threaded mode. Optional",
                        required=False,
                        action="store_true")
    parser.add_argument("-pr",
                        "--print_run_order",
                        help="When added, the program will only print out the run order for the typed command. This can"
                             "be useful for debugging or filenaming purposes",
                        required=False,
                        action="store_true")

    return parser


def main(protocol, stab_type, color, ltsv, sv, pg, pm, pn, fn, print_mode):
    if stab_type not in ["X", "Z"]:
        print("ERROR: the specified stabilizer type was not recognised. Please choose between: X or Z")
        exit()

    protocol = protocol.lower()
    fn_text = ""
    if sv and fn is not None:
        fn_text = "A CSV file will be saved with the name: {}".format(fn)
    print("\nRunning the {} protocol, with pg={}, pm={}{}, for a {} stabilizer. {}\n"
          .format(protocol, pg, pm, (' and pn=' + str(pn) if protocol != 'monolithic' else ""),
                  "plaquette" if stab_type == "Z" else "star", fn_text))

    if print_mode:
        return []

    gate = Z_gate if stab_type == "Z" else X_gate

    if protocol == "monolithic":
        return monolithic(gate, pg, pm, color, ltsv, sv, fn)
    elif protocol == "expedient":
        return expedient(gate, pg, pm, pn, color, ltsv, sv, fn)
    elif protocol == "stringent":
        return stringent(gate, pg, pm, pn, color, ltsv, sv, fn)
    else:
        print("ERROR: the specified protocol was not recognised. Choose between: monolithic, expedient or stringent.")
        exit()


if __name__ == "__main__":
    parser = compose_parser()

    args = vars(parser.parse_args())
    protocols = args.pop('protocol')
    stab_type = args.pop('stabilizer_type').upper()
    color = args.pop('color')
    meas_errors = args.pop('measurement_error_probability')
    meas_eq_gate = args.pop('pm_equals_pg')
    network_errors = args.pop('network_error_probability')
    gate_errors = args.pop('gate_error_probability')
    ltsv = args.pop('save_latex_pdf')
    sv = args.pop('save_csv')
    filenames = args.pop('csv_filename')
    threaded = args.pop('threaded')
    print_mode = args.pop('print_run_order')

    if threaded:
        workers = len(gate_errors) if len(gate_errors) < 11 else 10
        thread_pool = Pool(workers)
        results = []

    filename_count = 0

    start = time.time()
    for protocol in protocols:
        for pg in gate_errors:
            if meas_eq_gate:
                meas_errors = [pg]
            for pm in meas_errors:
                for pn in network_errors:
                    fn = None if (filenames is None or len(filenames) <= filename_count) else filenames[filename_count]
                    if threaded:
                        results.append(thread_pool.
                                       apply_async(main,
                                                   (protocol, stab_type, color, ltsv, sv, pg, pm, pn, fn, print_mode)))
                    else:
                        print(*main(protocol, stab_type, color, ltsv, sv, pg, pm, pn, fn, print_mode))
                    filename_count += 1

    if threaded:
        [print(*res.get()) for res in results]
        thread_pool.close()




