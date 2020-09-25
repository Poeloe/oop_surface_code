from circuit_simulation.states.states import *
from circuit_simulation.gates.gates import *
from circuit_simulation.basic_operations.basic_operations import *
import copy


def init_density_matrix(self):
    """ Realises init_type option 0. See class description for more info. """

    density_matrices = []
    for i, qubit in enumerate(self._qubit_array):
        density_matrix = CT(qubit, qubit)
        density_matrices.append(density_matrix)
        self._qubit_density_matrix_lookup[i] = (density_matrix, [i])
    return density_matrices


def init_density_matrix_first_qubit_ket_p(self):
    """ Realises init_type option 1. See class description for more info. """

    self._qubit_array[0] = ket_p

    density_matrices = []
    for i, qubit in enumerate(self._qubit_array):
        density_matrix = CT(qubit, qubit)
        density_matrices.append(density_matrix)
        self._qubit_density_matrix_lookup[i] = (density_matrix, [i])

    return density_matrices


def init_density_matrix_bell_pair_state(self, amount_qubits=8, draw=True):
    """ Realises init_type option 2. See class description for more info. """

    density_matrices = []
    bell_pair_rho = sp.csr_matrix([[1 / 2, 0, 0, 1 / 2],
                                   [0, 0, 0, 0],
                                   [0, 0, 0, 0],
                                   [1 / 2, 0, 0, 1 / 2]])

    for i in range(0, self.num_qubits - amount_qubits):
        state = self._qubit_array[i]
        self._qubit_density_matrix_lookup[i] = (CT(state), [i])
        self._uninitialised_qubits.append(i)

    for i in range(self.num_qubits - amount_qubits, self.num_qubits, 2):
        density_matrix = copy.copy(bell_pair_rho)
        qubits = [i, i + 1]
        if draw:
            self._add_draw_operation("#", (i, i + 1))
        self._qubit_density_matrix_lookup.update({i: (density_matrix, qubits), i + 1: (density_matrix, qubits)})
        density_matrices.append(density_matrix)
    return density_matrices


def init_density_matrix_ket_p_and_CNOTS(self):
    """ Realises init_type option 3. See class description for more info. """

    # Set ket_p as first qubit of the qubit array (mainly for proper drawing of the circuit)
    self._qubit_array[0] = ket_p

    density_matrix = sp.lil_matrix((self.d, self.d))
    density_matrix[0, 0] = 1 / 2
    density_matrix[0, self.d - 1] = 1 / 2
    density_matrix[self.d - 1, 0] = 1 / 2
    density_matrix[self.d - 1, self.d - 1] = 1 / 2
    density_matrix = sp.csr_matrix(density_matrix)

    density_matrices = [density_matrix]

    qubits = [i for i, _ in enumerate(self._qubit_array)]

    for j, _ in enumerate(self._qubit_array):
        self._qubit_density_matrix_lookup[j] = (density_matrix, qubits)

    for i in range(1, self.num_qubits):
        self._add_draw_operation(CNOT_gate, (0, i))

    return density_matrices


def init_parameters_to_dict(self):
    init_params = {'num_qubits': self.num_qubits,
                   'd': self.d,
                   'init_type': self._init_type,
                   'noise': self.noise,
                   'basis_transformation_noise': self.basis_transformation_noise,
                   'pm': self.pm,
                   'pg': self.pg,
                   'pn': self.pn,
                   'qubit_array': self._qubit_array,
                   'density_matrices': self.density_matrices,
                   'qubit_density_matrix_lookup': self._qubit_density_matrix_lookup}

    return init_params