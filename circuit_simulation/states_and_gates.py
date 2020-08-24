import numpy as np
from abc import ABC, abstractmethod
import circuit_simulation.circuit_simulator as cs


class Gate(ABC):

    def __init__(self, name, matrix, representation, duration=0):
        self._name = name
        self._matrix = matrix
        self._representation = representation
        self._duration = duration

    @property
    def name(self):
        return self._name

    @property
    def matrix(self):
        return self._matrix

    @property
    def representation(self):
        return self._representation

    @property
    def duration(self):
        return self._duration

    @property
    def dagger(self):
        return self.matrix.conj().T

    @duration.setter
    def duration(self, duration):
        self._duration = duration


    def __repr__(self):
        return "{}:\n{}".format(self._representation, self.matrix)

    def __eq__(self, other):
        return np.array_equal(self.matrix, other.matrix)


class SingleQubitGate(Gate):

    def __init__(self, name, matrix, representation, duration=0):
        super().__init__(name, matrix, representation, duration)

    def get_circuit_dimension_matrix(self, num_qubits, target_qubit):
        qc = cs.QuantumCircuit(num_qubits=num_qubits, init_type=0)
        return qc._create_1_qubit_gate(self, target_qubit)


class TwoQubitGate(Gate):

    def __init__(self, name, matrix, representation, duration=0, control_repr="o"):
        super().__init__(name, matrix, representation)
        self._duration = duration
        self._control_repr = control_repr
        self._one_state_matrix = matrix[2:, 2:]
        self._zero_state_matrix = matrix[:2, :2]

    @property
    def control_repr(self):
        return self._control_repr

    @property
    def zero_state_matrix(self):
        return self._zero_state_matrix

    @property
    def one_state_matrix(self):
        return self._one_state_matrix

    def get_circuit_dimension_matrix(self, num_qubits, control_qubit, target_qubit):
        qc = cs.QuantumCircuit(num_qubits=num_qubits, init_type=0)
        return qc._create_2_qubit_gate(self, control_qubit, target_qubit)


class State(object):

    def __init__(self, name, vector, representation):
        self._name = name
        self._vector = vector
        self._representation = representation

    @property
    def name(self):
        return self._name

    @property
    def vector(self):
        return self._vector

    @property
    def representation(self):
        return self._representation

    def __repr__(self):
        return self.representation

    def __eq__(self, other):
        return np.array_equal(self.vector, other.vector)

"""
    SINGLE QUBIT STATES
"""
ket_0 = State("Zero", np.array([[1, 0]]).T, "|0>")
ket_1 = State("One", np.array([[0, 1]]).T, "|1>")
ket_p = State("Plus", 1 / np.sqrt(2) * np.array([[1, 1]]).T, "|+>")
ket_m = State("Minus", 1 / np.sqrt(2) * np.array([[1, -1]]).T, "|->")

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
                         4)
CZ_gate = TwoQubitGate("CPhase",
                       np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0],
                                 [0, 0, 1, 0],
                                 [0, 0, 0, -1]]),
                       "Z",
                       4)
NV_two_qubit_gate = TwoQubitGate("NV two-qubit gate",
                                 np.array([[np.cos(np.pi/4), 1 * np.sin(np.pi/4), 0, 0],
                                           [-1 * np.sin(np.pi/4), np.cos(np.pi/4), 0, 0],
                                           [0, 0, np.cos(np.pi/4), -1 * np.sin(np.pi/4)],
                                           [0, 0, 1 * np.sin(np.pi/4), np.cos(np.pi/4)]]),
                                 "NV")
