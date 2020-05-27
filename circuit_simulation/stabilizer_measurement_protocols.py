import os
import sys
import argparse
sys.path.insert(1, os.path.abspath(os.getcwd()))
from circuit_simulation.circuit_simulator import *
from circuit_simulation.basic_operations import gate_name_to_array


def monolithic(operation, color):
    qc = QuantumCircuit(8, 2, noise=True, pg=0.009, pm=0.009)
    qc.add_top_qubit(ket_p)
    qc.apply_2_qubit_gate(operation, 0, 1)
    qc.apply_2_qubit_gate(operation, 0, 3)
    qc.apply_2_qubit_gate(operation, 0, 5)
    qc.apply_2_qubit_gate(operation, 0, 7)
    qc.measure_first_N_qubits(1)

    qc.draw_circuit(color)
    qc.get_superoperator([0, 2, 4, 6], gate_name(operation))


def expedient(operation, color):
    qc = QuantumCircuit(8, 2, noise=True, pg=0.006, pm=0.006, pn=0.1)

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
    qc.get_superoperator([0, 2, 4, 6], gate_name(operation), no_color=color)


def stringent(operation, color):
    qc = QuantumCircuit(8, 2, noise=True, pg=0.0075, pm=0.0075, pn=0.1)

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
    qc.get_superoperator([0, 2, 4, 6], gate_name(operation), no_color=color)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--protocol',
                        help='Specified which protocol should be used. - options: {monolithic/expedient/stringent}',
                        default='monolithic')
    parser.add_argument('-s',
                        '--stabilizer_type',
                        help='Specifies what the kind of stabilizer should be. - options: {Z/X}',
                        default='Z')
    parser.add_argument('-c',
                        '--color',
                        help='Specifies if the console output should display color. Optional',
                        required=False,
                        action='store_true')
    args = vars(parser.parse_args())
    protocol = args.pop('protocol').lower()
    stab_type = args.pop('stabilizer_type').upper()
    color = args.pop('color')

    if stab_type not in ["X", "Z"]:
        print("ERROR: the specified stabilizer type was not recognised. Please choose between: X or Z")
        exit()

    if protocol == "monolithic":
        monolithic(gate_name_to_array(stab_type), color)
    elif protocol == "expedient":
        expedient(gate_name_to_array(stab_type), color)
    elif protocol == "stringent":
        stringent(gate_name_to_array(stab_type), color)
    else:
        print("ERROR: the specified protocol was not recognised. Choose between: monolithic, expedient or stringent.")
        exit()
