from circuit_simulation.basic_operations.basic_operations import CT
from circuit_simulation.states.states import ket_0, ket_1
import numpy as np

# noinspection PyProtectedMember
class Qubit:

    def __init__(self, qc, index, qubit_type, node=None, waiting_time_idle=0, waiting_time_lde=0, T1_idle=None, \
                                                                                                      T2_idle=None,
                 T1_lde=None, T2_lde=None, is_data_qubit=False):
        self._index = index
        self._qubit_type = qubit_type
        self._node = node
        self._waiting_time_idle = waiting_time_idle
        self._waiting_time_lde = waiting_time_lde
        self._sequence_time = 0
        self._T1_idle = T1_idle
        self._T2_idle = T2_idle
        self._T1_lde = T1_lde
        self._T2_lde = T2_lde
        self._qc = qc
        self._is_data_qubit = is_data_qubit

    @property
    def index(self):
        return self._index

    @property
    def qubit_type(self):
        return self._qubit_type

    @property
    def node(self):
        return self._node

    @property
    def waiting_time_idle(self):
        return self._waiting_time_idle

    @property
    def waiting_time_lde(self):
        return self._waiting_time_lde

    @property
    def sequence_time(self):
        return self._sequence_time

    @property
    def T1_idle(self):
        return self._T1_idle

    @property
    def T2_idle(self):
        return self._T2_idle

    @property
    def T1_lde(self):
        return self._T1_lde

    @property
    def T2_lde(self):
        return self._T2_lde

    @property
    def is_data_qubit(self):
        return self._is_data_qubit

    @property
    def density_matrix(self):
        return self._qc._qubit_density_matrix_lookup[self.index][0]

    def increase_sequence_time(self, amount):
        if self.qubit_type == 'e':
            return
        self._sequence_time += amount

    def increase_waiting_time(self, amount, waiting_type='idle'):
        if waiting_type not in ['idle', 'LDE']:
            raise ValueError("Waiting type should be either 'idle' or 'LDE'")

        if waiting_type == 'idle':
            self._waiting_time_idle += amount
        elif waiting_type == 'LDE':
            self._waiting_time_lde += amount

        # When waiting time is increased, qubit is initialised and the nuclear qubit is being decoupled (pulse sequence)
        if self.qubit_type == 'n' and not self.equal_to_0_or_1_state():
            self.increase_sequence_time(amount)

    def reset_waiting_time(self):
        self._waiting_time_idle = 0
        self._waiting_time_lde = 0

    def reset_sequence_time(self):
        self._sequence_time = 0

    def equal_to_0_or_1_state(self):
        """
        If the state of te qubits is equal to |0> or |1>, then also no sequence is applied. This is checked here
        """
        dens = self.density_matrix
        zero_state = CT(ket_0).toarray()
        one_state = CT(ket_1).toarray()
        # Quick dimension check, such that no unnecessary big matrix comparison is performed
        if dens.shape != zero_state.shape:
            return False

        dens = dens.toarray()
        # State on qubit may be noisy, therefore comparison is with tolerance
        return np.allclose(dens, zero_state, 1e-3, 1e-3) or np.allclose(dens, one_state, 1e-3, 1e-3)
