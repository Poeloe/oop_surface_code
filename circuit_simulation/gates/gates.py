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


np.zeros((4,4)).conj()
"""
    TWO-QUBIT GATES
"""
CNOT_gate = TwoQubitGate("CNOT",
                         np.array([[1, 0, 0, 0],
                                   [0, 1, 0, 0],
                                   [0, 0, 0, 1],
                                   [0, 0, 1, 0]]),
                         "X",
                         25.1e-3)
CZ_gate = TwoQubitGate("CPhase",
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, -1]]),
                       "Z",
                       25.1e-3)
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
                         control_repr="(X)")