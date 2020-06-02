import os
import sys
import argparse
sys.path.insert(1, os.path.abspath(os.getcwd()))
from circuit_simulation.circuit_simulator import *
from circuit_simulation.basic_operations import gate_name_to_array


def monolithic(operation, pg, pm, color, save_latex_pdf, save_csv, csv_file_name):
    qc = QuantumCircuit(8, 2, noise=True, pg=pg, pm=pm)
    qc.add_top_qubit(ket_p)
    qc.apply_2_qubit_gate(operation, 0, 1)
    qc.apply_2_qubit_gate(operation, 0, 3)
    qc.apply_2_qubit_gate(operation, 0, 5)
    qc.apply_2_qubit_gate(operation, 0, 7)
    qc.measure_first_N_qubits(1)

    qc.draw_circuit(color)
    if save_latex_pdf:
        qc.draw_circuit_latex()
    qc.get_superoperator([0, 2, 4, 6], gate_name(operation), no_color=color, to_csv=save_csv,
                         csv_file_name=csv_file_name)


def expedient(operation, pg, pm, pn, color, save_latex_pdf, save_csv, csv_file_name):
    qc = QuantumCircuit(8, 2, noise=True, pg=pg, pm=pm, pn=pn)

    # Noisy ancilla Bell pair is now between are now 0 and 1
    qc.create_bell_pairs_top(1, new_qubit=True)
    qc.double_selection(Z, new_qubit=True)
    qc.double_selection(X)

    # New noisy ancilla Bell pair is now between 0 and 1, old ancilla Bell pair now between 2 and 3
    qc.create_bell_pairs_top(1, new_qubit=True)
    qc.double_selection(Z, new_qubit=True)
    qc.double_selection(X)

    # Now entanglement between ancilla 0 and 3 is made
    qc.single_dot(Z, 2, 5)
    qc.single_dot(Z, 2, 5)

    # And finally the entanglement between ancilla 1 and 2 is made, now all ancilla's are entangled
    qc.single_dot(Z, 3, 4)
    qc.single_dot(Z, 3, 4)

    qc.apply_2_qubit_gate(operation, 0, 4)
    qc.apply_2_qubit_gate(operation, 1, 6)
    qc.apply_2_qubit_gate(operation, 2, 8)
    qc.apply_2_qubit_gate(operation, 3, 10)

    qc.measure_first_N_qubits(4)

    qc.draw_circuit(no_color=color)
    if save_latex_pdf:
        qc.draw_circuit_latex()
    qc.get_superoperator([0, 2, 4, 6], gate_name(operation), no_color=color, to_csv=save_csv,
                         csv_file_name=csv_file_name, save_noiseless_density_matrix=False)


def stringent(operation, pg, pm, pn, color, save_latex_pdf, save_csv, csv_file_name):
    qc = QuantumCircuit(8, 2, noise=True, pg=pg, pm=pm, pn=pn)

    # Noisy ancilla Bell pair between 0 and 1
    qc.create_bell_pairs_top(1, new_qubit=True)
    qc.double_selection(Z, new_qubit=True)
    qc.double_selection(X)
    qc.double_dot(Z, 2, 3)
    qc.double_dot(X, 2, 3)

    # New noisy ancilla Bell pair is now between 0 and 1, old ancilla Bell pair now between 2 and 3
    qc.create_bell_pairs_top(1, new_qubit=True)
    qc.double_selection(Z, new_qubit=True)
    qc.double_selection(X)
    qc.double_dot(X, 2, 3)
    qc.double_dot(X, 2, 3)

    # Now entanglement between ancilla 0 and 3 is made
    qc.double_dot(Z, 2, 5)
    qc.double_dot(Z, 2, 5)

    # And finally the entanglement between ancilla 1 and 2 is made, now all ancilla's are entangled
    qc.double_dot(Z, 3, 4)
    qc.double_dot(Z, 3, 4)

    qc.apply_2_qubit_gate(operation, 3, 10)
    qc.apply_2_qubit_gate(operation, 2, 8)
    qc.apply_2_qubit_gate(operation, 1, 6)
    qc.apply_2_qubit_gate(operation, 0, 4)

    qc.measure_first_N_qubits(4)

    qc.draw_circuit(no_color=color)
    if save_latex_pdf:
        qc.draw_circuit_latex()
    qc.get_superoperator([0, 2, 4, 6], gate_name(operation), no_color=color, to_csv=save_csv,
                         csv_file_name=csv_file_name)


def compose_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--protocol',
                        help='Specifies which protocol should be used. - options: {monolithic/expedient/stringent}',
                        default='monolithic')
    parser.add_argument('-s',
                        '--stabilizer_type',
                        help='Specifies what the kind of stabilizer should be. - options: {Z/X}',
                        default='Z')
    parser.add_argument('-pg',
                        '--gate_error_probability',
                        help='Specifies the amount of gate error present in the system',
                        type=float,
                        default=0.006)
    parser.add_argument('-pm',
                        '--measurement_error_probability',
                        help='Specifies the amount of measurement error present in the system',
                        type=float,
                        default=0.006)
    parser.add_argument('-pn',
                        '--network_error_probability',
                        help='Specifies the amount of network error present in the system',
                        type=float,
                        default=0.006)
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
                        help='Give the file name of the csv file that will be saved.')

    return parser


if __name__ == "__main__":
    parser = compose_parser()

    args = vars(parser.parse_args())
    protocol = args.pop('protocol').lower()
    stab_type = args.pop('stabilizer_type').upper()
    color = args.pop('color')
    pm = args.pop('measurement_error_probability')
    pn = args.pop('network_error_probability')
    pg = args.pop('gate_error_probability')
    ltsv = args.pop('save_latex_pdf')
    sv = args.pop('save_csv')
    fn = args.pop('csv_filename')

    if stab_type not in ["X", "Z"]:
        print("ERROR: the specified stabilizer type was not recognised. Please choose between: X or Z")
        exit()

    print("\nRunning the {} protocol, with pg={}, pm={}{}.\n".format(protocol, pg, pm,
                                                                   (' and pn=' + str(pn) if protocol != 'monolithic'
                                                                    else "")))

    if protocol == "monolithic":
        monolithic(gate_name_to_array(stab_type), pg, pm, color, ltsv, sv, fn)
    elif protocol == "expedient":
        expedient(gate_name_to_array(stab_type), pg, pm, pn, color, ltsv, sv, fn)
    elif protocol == "stringent":
        stringent(gate_name_to_array(stab_type), pg, pm, pn, color, ltsv, sv, fn)
    else:
        print("ERROR: the specified protocol was not recognised. Choose between: monolithic, expedient or stringent.")
        exit()
