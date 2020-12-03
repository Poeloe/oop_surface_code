import numpy as np
from abc import ABC
import scipy.sparse as sp
import circuit_simulation.circuit_simulator as cs
from circuit_simulation.states.state import State


class Gate(ABC):
    """
    Abstract class Gate

    This class should be inherited when creating a gate class. The common attributes are:

    Attributes
    ----------
    name : str
        Name that will be used to refer to the gate
    matrix : numpy array
        Matrix that represents the gate operation.
    representation : str
        Representation is used in the drawing of the circuits
    duration : float
        Duration of the gate in time unit

    """

    def __init__(self, name, matrix, representation, duration=0, duration_electron=None):
        self._name = name
        self._matrix = matrix
        self._sp_matrix = sp.csr_matrix(matrix)
        self._representation = representation
        self._duration = duration
        self._duration_electron = duration_electron if duration_electron is not None else duration

    @property
    def name(self):
        return self._name

    @property
    def matrix(self):
        return self._matrix

    @property
    def sp_matrix(self):
        return self._sp_matrix

    @property
    def representation(self):
        return self._representation

    @property
    def duration(self):
        return self._duration

    @property
    def duration_electron(self):
        return self._duration_electron

    @property
    def dagger(self):
        return self.matrix.conj().T

    @duration.setter
    def duration(self, duration):
        self._duration = duration

    @duration_electron.setter
    def duration_electron(self, duration):
        self._duration_electron = duration

    def __mul__(self, other):
        if type(other) not in [SingleQubitGate, TwoQubitGate, State, sp.csr_matrix]:
            raise ValueError('It is not possible to multiply an object of type {} with a Gate object'
                             .format(type(other)))
        if type(other) in [SingleQubitGate, TwoQubitGate]:
            other_matrix = other.sp_matrix
        elif type(other) == State:
            other_matrix = other.sp_vector
        else:
            other_matrix = other

        return self.matrix * other_matrix

    def __repr__(self):
        return "{}:\n{}".format(self._representation, self.matrix)


class SingleQubitGate(Gate):
    """
        SingleQubitGate class inherits the abstract Gate class
    """

    def __init__(self, name, matrix, representation, duration=0, duration_electron=None):
        super().__init__(name, matrix, representation, duration, duration_electron)

    def get_circuit_dimension_matrix(self, num_qubits, target_qubit):
        qc = cs.QuantumCircuit(num_qubits=num_qubits, init_type=0)
        return qc._create_1_qubit_gate(self, target_qubit, num_qubits=num_qubits)

    def __eq__(self, other):
        if type(other) != SingleQubitGate:
            return False
        return np.array_equal(self.matrix, other.matrix)


class TwoQubitGate(Gate):

    """
        TwoQubitGate class inherits from the abstract Gate class

        Extra Attributes
        ----------------
        control_repr : str, default="o"
            Representation of the control operation on the control qubit when drawing circuits. Default is "o".
        upper_left_matrix : 2x2 numpy array
            The upper left matrix of the 4x4 matrix attribute
        lower_right_matrix : 2x2 numpy array
            The lower right matrix of the 4x4 matrix attribute
        upper_right_matrix : 2x2 numpy array
            The upper right matrix of the 4x4 matrix attribute
        lower_left_matrix : 2x2 numpy array
            The lower left matrix of the 4x4 matrix attribute
        is_cntrl_gate : bool
            True if only upper left matrix and lower right matrix are non-zero
    """

    def __init__(self, name, matrix, representation, duration=0., control_repr="o", duration_electron=None):
        super().__init__(name, matrix, representation, duration, duration_electron)
        self._control_repr = control_repr
        self._upper_left_matrix = self.sp_matrix[2:, 2:]
        self._lower_right_matrix = self.sp_matrix[:2, :2]
        self._upper_right_matrix = self.sp_matrix[:2, 2:]
        self._lower_left_matrix = self.sp_matrix[2:, :2]
        self._cntrl_gate = True if np.array_equal(self._upper_right_matrix.toarray(), np.zeros((2, 2))) and \
                                   np.array_equal(self._lower_left_matrix.toarray(), np.zeros((2, 2))) else False

    @property
    def control_repr(self):
        return self._control_repr

    @property
    def lower_right_matrix(self):
        return self._lower_right_matrix

    @property
    def upper_left_matrix(self):
        return self._upper_left_matrix

    @property
    def lower_left_matrix(self):
        return self._lower_left_matrix

    @property
    def upper_right_matrix(self):
        return self._upper_right_matrix

    @property
    def is_cntrl_gate(self):
        return self._cntrl_gate

    def get_circuit_dimension_matrix(self, num_qubits, control_qubit, target_qubit):
        qc = cs.QuantumCircuit(num_qubits=num_qubits, init_type=0)
        return qc._create_2_qubit_gate(self, control_qubit, target_qubit)

    def __eq__(self, other):
        if type(other) != TwoQubitGate:
            return False
        return np.array_equal(self.matrix, other.matrix)


