import os
import sys
import argparse
sys.path.insert(1, os.path.abspath(os.getcwd()))
from circuit_simulation.circuit_simulator import *
from multiprocessing import Pool
import time
from tqdm import tqdm


def monolithic(operation, pg, pm, color, save_latex_pdf, save_csv, csv_file_name, pbar):
    qc = QuantumCircuit(9, 2, noise=True, pg=pg, pm=pm, basis_transformation_noise=True,
                        thread_safe_printing=True)
    qc.set_qubit_states({0: ket_p})
    qc.apply_2_qubit_gate(operation, 0, 1)
    qc.apply_2_qubit_gate(operation, 0, 3)
    qc.apply_2_qubit_gate(operation, 0, 5)
    qc.apply_2_qubit_gate(operation, 0, 7)
    qc.measure([0])

    if pbar is not None:
        pbar.update(50)

    qc.draw_circuit(not color)
    if save_latex_pdf:
        qc.draw_circuit_latex()
    stab_rep = "Z" if operation == CZ_gate else "X"
    qc.get_superoperator([1, 3, 5, 7], stab_rep, no_color=(not color), to_csv=save_csv,
                         csv_file_name=csv_file_name, stabilizer_protocol=True)
    if pbar is not None:
        pbar.update(50)

    return qc._print_lines


def expedient(operation, pg, pm, pn, color, save_latex_pdf, save_csv, csv_file_name, pbar):
    start = time.time()
    qc = QuantumCircuit(20, 2, noise=True, basis_transformation_noise=False, pg=pg, pm=pm, pn=pn, network_noise_type=1,
                        thread_safe_printing=True, probabilistic=False, p_dec=0, p_bell_success=0.8)

    qc.create_bell_pair(11, 8)
    qc.double_selection(CZ_gate, 10, 7)
    qc.double_selection(CNOT_gate, 10, 7)

    if pbar is not None:
        pbar.update(20)

    qc.create_bell_pair(5, 2)
    qc.double_selection(CZ_gate, 4, 1)
    qc.double_selection(CNOT_gate, 4, 1)

    if pbar is not None:
        pbar.update(20)

    qc.single_dot(CZ_gate, 10, 4)
    qc.single_dot(CZ_gate, 10, 4)

    if pbar is not None:
        pbar.update(20)

    # And finally the entanglement between ancilla 1 and 2 is made, now all ancilla's are entangled
    qc.single_dot(CZ_gate, 7, 1)
    qc.single_dot(CZ_gate, 7, 1)

    if pbar is not None:
        pbar.update(20)

    qc.apply_2_qubit_gate(operation, 11, 18)
    qc.apply_2_qubit_gate(operation, 8, 16)
    qc.apply_2_qubit_gate(operation, 5, 14)
    qc.apply_2_qubit_gate(operation, 2, 12)

    qc.measure([8, 11, 2, 5], probabilistic=False)

    end_circuit = time.time()

    if pbar is not None:
        pbar.update(10)

    start_draw = time.time()
    qc.draw_circuit(no_color=not color)
    end_draw = time.time()

    start_superoperator = time.time()
    if save_latex_pdf:
        qc.draw_circuit_latex()
    stab_rep = "Z" if operation == CZ_gate else "X"
    qc.get_superoperator([18, 16, 14, 12], stab_rep, no_color=(not color), to_csv=save_csv,
                         csv_file_name=csv_file_name, stabilizer_protocol=True)
    end_superoperator = time.time()

    if pbar is not None:
        pbar.update(10)

    qc._print_lines.append("\nCircuit simulation took {} seconds".format(end_circuit - start))
    qc._print_lines.append("\nDrawing the circuit took {} seconds".format(end_draw - start_draw))
    qc._print_lines.append("\nCalculating the superoperator took {} seconds".format(end_superoperator -
                                                                                    start_superoperator))
    qc._print_lines.append("\nTotal time is {}\n".format(time.time() - start))

    return qc._print_lines


def stringent(operation, pg, pm, pn, color, save_latex_pdf, save_csv, csv_file_name, pbar):
    start = time.time()
    qc = QuantumCircuit(20, 2, noise=True, basis_transformation_noise=False, pg=pg, pm=pm, pn=pn, network_noise_type=1,
                        thread_safe_printing=True)

    qc.create_bell_pair(11, 8)
    qc.double_selection(CZ_gate, 10, 7)
    qc.double_selection(CNOT_gate, 10, 7)
    qc.double_dot(CZ_gate, 10, 7)
    qc.double_dot(CNOT_gate, 10, 7)

    if pbar is not None:
        pbar.update(20)

    # New noisy ancilla Bell pair is now between 0 and 1, old ancilla Bell pair now between 2 and 3
    qc.create_bell_pair(5, 2)
    qc.double_selection(CZ_gate, 4, 1)
    qc.double_selection(CNOT_gate, 4, 1)
    qc.double_dot(CZ_gate, 4, 1)
    qc.double_dot(CNOT_gate, 4, 1)

    if pbar is not None:
        pbar.update(20)

    # Now entanglement between ancilla 0 and 3 is made
    qc.double_dot(CZ_gate, 10, 4)
    qc.double_dot(CZ_gate, 10, 4)

    if pbar is not None:
        pbar.update(20)

    # And finally the entanglement between ancilla 1 and 2 is made, now all ancilla's are entangled
    qc.double_dot(CZ_gate, 7, 1)
    qc.double_dot(CZ_gate, 7, 1)

    if pbar is not None:
        pbar.update(20)

    qc.apply_2_qubit_gate(operation, 11, 18)
    qc.apply_2_qubit_gate(operation, 8, 16)
    qc.apply_2_qubit_gate(operation, 5, 14)
    qc.apply_2_qubit_gate(operation, 2, 12)

    qc.measure([8, 11, 2, 5], probabilistic=False)

    end_circuit = time.time()

    if pbar is not None:
        pbar.update(10)

    start_draw = time.time()
    qc.draw_circuit(no_color=not color)
    end_draw = time.time()

    if save_latex_pdf:
        qc.draw_circuit_latex()

    stab_rep = "Z" if operation == CZ_gate else "X"
    start_superoperator = time.time()
    qc.get_superoperator([18, 16, 14, 12], stab_rep, no_color=(not color), to_csv=save_csv,
                         csv_file_name=csv_file_name, stabilizer_protocol=True)
    end_superoperator = time.time()

    if pbar is not None:
        pbar.update(10)

    qc._print_lines.append("\nCircuit simulation took {} seconds".format(end_circuit - start))
    qc._print_lines.append("\nDrawing the circuit took {} seconds".format(end_draw - start_draw))
    qc._print_lines.append("\nCalculating the superoperator took {} seconds".format(end_superoperator -
                                                                                  start_superoperator))
    qc._print_lines.append("\nTotal time is {}\n".format(time.time() - start))

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


def main(protocol, stab_type, color, ltsv, sv, pg, pm, pn, fn, print_mode, pbar=None):
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

    gate = CZ_gate if stab_type == "Z" else CNOT_gate

    if protocol == "monolithic":
        return monolithic(gate, pg, pm, color, ltsv, sv, fn, pbar)
    elif protocol == "expedient":
        return expedient(gate, pg, pm, pn, color, ltsv, sv, fn, pbar)
    elif protocol == "stringent":
        return stringent(gate, pg, pm, pn, color, ltsv, sv, fn, pbar)
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

    if not threaded:
        pbar = tqdm(total=100)

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
                        print(*main(protocol, stab_type, color, ltsv, sv, pg, pm, pn, fn, print_mode, pbar))
                        pbar.reset()
                    filename_count += 1

    if threaded:
        [print(*res.get()) for res in results]
        thread_pool.close()




