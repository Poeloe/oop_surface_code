import numpy as np
from .gate import TwoQubitGate, SingleQubitGate

"""
    SINGLE QUBIT GATES
"""
X_gate = SingleQubitGate("X", np.array([[0, 1], [1, 0]]), "X")
Y_gate = SingleQubitGate("Y", np.array([[0, -1j], [1j, 0]]), "Y")
Z_gate = SingleQubitGate("Z", np.array([[1, 0], [0, -1]]), "Z")
I_gate = SingleQubitGate("Identity", np.array([[1, 0], [0, 1]]), "I")
H_gate = SingleQubitGate("Hadamard", 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]]), "H")
S_gate = SingleQubitGate("Phase", np.array([[1, 0], [0, 1j]]), "S")

"""
    TWO-QUBIT GATES
"""
CNOT_gate = TwoQubitGate("CNOT",
                         np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 1],
                                   [0, 0, 1, 0]]),
                         "X",
                         1)
CZ_gate = TwoQubitGate("CPhase",
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, -1]]),
                       "Z",
                       1)
NV_two_qubit_gate = TwoQubitGate("NV two-qubit gate",
                                 np.array([[np.cos(np.pi/4), 1 * np.sin(np.pi/4), 0, 0],
                                           [-1 * np.sin(np.pi/4), np.cos(np.pi/4), 0, 0],
                                           [0, 0, np.cos(np.pi/4), -1 * np.sin(np.pi/4)],
                                           [0, 0, 1 * np.sin(np.pi/4), np.cos(np.pi/4)]]),
                                 "NV")

SWAP_gate = TwoQubitGate("Swap",
                         np.array([[1, 0, 0, 0],
                                   [0, 0, 1, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 1]]),
                         "(X)",
                         control_repr="(X)",
                         duration=0.1)

locals_gates = locals()


def set_duration_of_known_gates(gates_dict):
    for gate, duration in gates_dict.items():
        if gate in locals_gates and type(locals_gates[gate]) in [SingleQubitGate, TwoQubitGate]:
            locals_gates[gate].duration = duration


def set_gate_durations_from_file(filename):
    if filename is None:
        return
    gates_dict = {}
    with open(filename, 'r') as gate_durations:
        lines = gate_durations.read().split('\n')
        for line in lines:
            splitted_line = line.split("=")
            gate_name = splitted_line[0]
            gate_duration = float(splitted_line[1])
            gates_dict[gate_name] = gate_duration

    set_duration_of_known_gates(gates_dict)
