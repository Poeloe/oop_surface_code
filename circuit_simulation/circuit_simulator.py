from circuit_simulation.basic_operations import (
    CT, KP, N_dim_ket_0_or_1_density_matrix, state_repr, get_value_by_prob, trace, gate_name
)
import numpy as np
import time
from scipy import sparse as sp
import itertools as it
import copy
from scipy.linalg import eigh
import faulthandler


# These states must be in this file, since it will otherwise cause a segmentation error when diagonalising the density
# matrix for a circuit with a large amount of qubits
ket_0 = np.array([[1, 0]]).T
ket_1 = np.array([[0, 1]]).T
ket_p = 1 / np.sqrt(2) * (ket_0 + ket_1)
ket_m = 1 / np.sqrt(2) * (ket_0 - ket_1)
ket_00 = np.array([[1, 0, 0, 0]]).T
ket_11 = np.array([[0, 0, 0, 1]]).T

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])
I = np.array([[1, 0], [0, 1]])
H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])


class Circuit:

    def __init__(self, num_qubits, init_type=None, noise=False, pg=0.01, pm=0.01):
        self.num_qubits = num_qubits
        self.d = 2 ** num_qubits
        self.noise = noise
        self.pg = pg
        self.pm = pm
        self._qubit_array = num_qubits * [ket_0]
        self._draw_order = []
        self.density_matrix = None

        if init_type == 0:
            self._init_density_matrix()
        elif init_type == 1:
            self._CNOT_init_density_matrix()
        elif init_type == 2:
            self._standard_init_density_matrix()
        elif init_type == 3:
            self._init_bell_pair_state()

    """
        -------------------------
            Init Methods
        -------------------------     
    """

    def _standard_init_density_matrix(self):
        self._qubit_array[0] = ket_p

        density_matrix = sp.lil_matrix((self.d, self.d))
        density_matrix[0, 0] = 1 / 2
        density_matrix[0, self.d/2] = 1 / 2
        density_matrix[self.d/2, 0] = 1 / 2
        density_matrix[self.d/2, self.d/2] = 1 / 2

        self.density_matrix = density_matrix

    def _CNOT_init_density_matrix(self):
        self._qubit_array[0] = ket_p

        density_matrix = sp.lil_matrix((self.d, self.d))
        density_matrix[0, 0] = 1 / 2
        density_matrix[0, self.d-1] = 1 / 2
        density_matrix[self.d-1, 0] = 1 / 2
        density_matrix[self.d-1, self.d-1] = 1 / 2

        self.density_matrix = density_matrix

        for i in range(1, self.num_qubits):
            self._draw_order.append({"X": (0, i)})

    def _init_density_matrix(self):
        state_vector = KP(*self._qubit_array)
        self.density_matrix = sp.csr_matrix(CT(state_vector, state_vector))

    def _init_bell_pair_state(self):
        result = None
        bell_pair_rho = sp.lil_matrix((4, 4))
        bell_pair_rho[0, 0], bell_pair_rho[3, 0], bell_pair_rho[0, 3], bell_pair_rho[3, 3] = 1/2, 1/2, 1/2, 1/2
        for i in range(0, self.num_qubits, 2):
            if result is None:
                result = bell_pair_rho
                self._draw_order.append({"#": (i, i+1)})
                continue

            result = sp.kron(bell_pair_rho, result)
            self._draw_order.append({"#": (i, i+1)})

        self.density_matrix = result

    def set_qubit_states(self, dict):
        for tqubit, state in dict.items():
            self._qubit_array[tqubit] = state
        self._init_density_matrix()

    def get_begin_states(self):
        return KP(*self._qubit_array)

    def create_bell_pair(self, qubits):
        """
        Only usable at initialisation and when init_type of the Circuit is set to 0
        """
        for qubit1, qubit2 in qubits:
            self.H(qubit1, noise=False, draw=False)
            self.CNOT(qubit1, qubit2, noise=False, draw=False)
            self._draw_order.append({"#": (qubit1, qubit2)})

    def create_noisy_bell_pair(self, qubits, pn=0.1, new=True):
        for qubit1, qubit2 in qubits:
            if new:
                self._qubit_array.insert(0, ket_0)
                self._qubit_array.insert(0, ket_0)
                self.num_qubits += 2
                self.d = self.num_qubits**2
                rho = sp.lil_matrix((4, 4))
                rho[0, 0], rho[0, 3], rho[3, 0], rho[3, 3] = 1/2, 1/2, 1/2, 1/2

                self.density_matrix = sp.kron((1 - 4*pn/3) * rho + pn/3 * sp.eye(4, 4), self.density_matrix)

            self._draw_order.append({"@": (qubit1, qubit2)})

    """
        -----------------------------
            One-Qubit Gate Methods
        -----------------------------     
    """

    def apply_1_qubit_gate(self, gate, tqubit, noise=None, pg=None, draw=True):
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        one_qubit_gate = self._create_1_qubit_gate(gate, tqubit)
        self.density_matrix = sp.csr_matrix(one_qubit_gate.dot(CT(self.density_matrix, one_qubit_gate)))

        if noise:
            self._N_single(pg, tqubit)

        if draw:
            self._draw_order.append({gate_name(gate): tqubit})

    def _create_1_qubit_gate(self, gate, tqubit):
        if np.array_equal(gate, I):
            return sp.eye(self.d, self.d)

        first_id, second_id = self._create_identity_operations(tqubit)

        return sp.csr_matrix(KP(first_id, gate, second_id))

    def _create_identity_operations(self, tqubit):
        first_id = None
        second_id = None

        if tqubit == 0:
            second_id = sp.eye(2**(self.num_qubits - 1 - tqubit), 2**(self.num_qubits - 1 - tqubit))
        elif tqubit == self.num_qubits - 1:
            first_id = sp.eye(2**tqubit, 2**tqubit)
        else:
            first_id = sp.eye(2 ** tqubit, 2 ** tqubit)
            second_id = sp.eye(2 ** (self.num_qubits - 1 - tqubit), 2 ** (self.num_qubits - 1 - tqubit))

        return first_id, second_id

    def X(self, tqubit, times=1, noise=None, pg=None, draw=True):
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        for _ in range(times):
            self.apply_1_qubit_gate(X, tqubit, noise, pg, draw)

    def Z(self, tqubit, times=1, noise=None, pg=None, draw=True):
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        for _ in range(times):
            self.apply_1_qubit_gate(Z, tqubit, noise, pg, draw)

    def Y(self, tqubit, times=1, noise=None, pg=None, draw=True):
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        for _ in range(times):
            self.apply_1_qubit_gate(Y, tqubit, noise, pg, draw)

    def H(self, tqubit, times=1, noise=None, pg=None, draw=True):
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        for _ in range(times):
            self.apply_1_qubit_gate(H, tqubit, noise, pg, draw)

    """
        -----------------------------
            Two-Qubit Gate Methods
        -----------------------------     
    """

    def apply_2_qubit_gate(self, gate, cqubit, tqubit, noise=None, pg=None, draw=True):
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg
        two_qubit_gate = self._create_2_qubit_gate(gate, cqubit, tqubit)

        self.density_matrix = sp.csr_matrix(two_qubit_gate.dot(CT(self.density_matrix, two_qubit_gate)))

        if noise:
            self._N(pg, cqubit, tqubit)

        if draw:
            self._draw_order.append({gate_name(gate): (cqubit, tqubit)})

    def _create_2_qubit_gate(self, gate, cqubit, tqubit):
        """
        Create a controlled gate matrix for the density matrix according to the control and target qubits given.
        This is done by
                1.  first taking the Kronecker Product the identity matrix as many times as there are qubits
                    present in the system.
                2.  Then for the two sub gates formed on the place of the control qubit the identity matrix
                    is replaced for a |0><0| and |1><1| matrix respectively.
                3.  Then for the gate_2 the identity matrix on the target qubit index is replaced with the wanted gate.

        So for creating a CNOT gate with the control on the 2nd qubit and target on the first qubit on a system with 3
        qubits one will get:

                1. I*I*I + I*I*I
                2. I*|0><0|*I + I*|1><1|*I
                3. I*|0><0|*I + X_t*|1><1|*I

        (In which * is the Kronecker Product) (https://quantumcomputing.stackexchange.com/questions/4252/
        how-to-derive-the-cnot-matrix-for-a-3-qbit-system-where-the-control-target-qbi)
        """
        gate_1 = self._create_1_qubit_gate(CT(ket_0), cqubit)
        gate_2 = self._create_1_qubit_gate(CT(ket_1), cqubit)

        x_gate = self._create_1_qubit_gate(X, tqubit)
        gate_2 = x_gate.dot(gate_2)

        return sp.csr_matrix(gate_1 + gate_2)

    def CNOT(self, cqubit, tqubit, noise=None, pg=None, draw=True):
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        self.apply_2_qubit_gate(X, cqubit, tqubit, noise, pg, draw)

    def CZ(self, cqubit, tqubit, noise=None, pg=None):
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        self.apply_2_qubit_gate(Z, cqubit, tqubit, noise, pg)

    """
        --------------------------------------
            Density Matrix calculus Methods
        --------------------------------------     
    """
    def measure_first_N_qubits(self, N, noise=None, pm=None, keep_qubits=False, basis="X", measure="even"):
        if noise is None:
            noise = self.noise
        if pm is None:
            pm = self.pm

        for qubit in range(N):
            if basis == "X":
                self.H(qubit)

            # if measure != "even" and qubit % 2 == 1:
            #     self._measurement_first_qubit(measure=1, pm=pm)
            self._measurement_first_qubit(measure=0, noise=noise, pm=pm)

            self._draw_order.append({"M": qubit})

            if keep_qubits:
                if basis == "X":
                    self.H(qubit)

        density_matrix = self.density_matrix

        if keep_qubits:
            density_matrix = sp.kron(N_dim_ket_0_or_1_density_matrix(N), self.density_matrix)
            self.num_qubits += N
            self.d = 2**self.num_qubits

        self.density_matrix = density_matrix / trace(density_matrix)

    def _measurement_first_qubit(self, measure=0, noise=True, pm=0.):
        density_matrix_0 = self.density_matrix[:int(self.d/2), :int(self.d/2)]
        density_matrix_1 = self.density_matrix[int(self.d/2):, int(self.d/2):]

        if measure == 0 and noise:
            density_matrix = (1 - pm) * density_matrix_0 + pm * density_matrix_1
        elif noise:
            density_matrix = (1 - pm) * density_matrix_1 + pm * density_matrix_0
        elif measure == 0:
            density_matrix = density_matrix_0
        else:
            density_matrix = density_matrix_1

        self.density_matrix = density_matrix

        self.num_qubits -= 1
        self.d = 2**self.num_qubits

    def measure(self, qubit, measure=None, basis="X"):
        if basis == "X":
            qc.H(qubit, noise=False)

        # If no specific measurement outcome is given it is chosen by the hand of the probability
        if measure is None:
            # eigenvalues, eigenvectors = self.get_non_zero_prob_eigenvectors()
            prob1, density_matrix1 = self._measurement(qubit, measure=0)
            prob2, density_matrix2 = self._measurement(qubit, measure=1)

            self.density_matrix = get_value_by_prob([density_matrix1, density_matrix2], [prob1, prob2])
        else:
            self.density_matrix = self._measurement(qubit, measure)[1]

        self._draw_order.append({"M": qubit})

        if basis == "X":
            qc.H(qubit, noise=False)


    def _measurement(self, qubit, measure=0, eigenval=None, eigenvec=None):
        """
        This private method calculates the probability of a certain measurement outcome and calculates the
        resulting density matrix after the measurement has taken place.

        ----
        Probability calculation:

        From the eigenvectors and the eigenvalues of the density matrix before the measurement, first the probability
        of the specified outcome (0 or 1) for the given qubit is calculated. This is done by setting the opposite
        outcome for the qubit to 0 in the eigenvectors. Remember, the eigenvectors represent a system state and are thus
        the possible qubit states tensored. Thus an eigenvector is built up as:

        |a_1|   |b_1|   |c_1|   |a_1 b_1 c_1|
        |   | * |   | * |   | = |a_1 b_1 c_2| (and so on)
        |a_2|   |b_2|   |c_2|   |a_1 b_2 c_1|
                                      :

        So lets say that we measure qubit c to be 1, this means that c_1 is zero. For each eigenvector we will set the
        elements that contain c_2 to zero, which leaves us with the states (if not a zero vector) that survive after
        the measurement. While setting these elements to zero, the other elements (that contain c_2) are saved to an
        array. From this array, the non-zero array is obtained which is then absolute squared, summed and multiplied
        with the eigenvalue for that eigenvector. These values obtained from all the eigenvectors are then summed to
        obtain the probability for the given outcome.

        ----

        Density matrix calculation:

        The density matrix after the measurement is obtained by taking the CT of the adapted eigenvectors by the
        probability calculations, multiply the result with the eigenvalue for that eigenvector and add all resulting
        matrices.
        """
        if eigenvec is None:
            eigenvalues, eigenvectors = self.get_non_zero_prob_eigenvectors()
        else:
            eigenvalues, eigenvectors = eigenval, copy.copy(eigenvec)

        iterations = 2**qubit
        step = int(qc.d/(2**(qubit+1)))
        prob = 0

        # Let measurement outcome determine the states that 'survive'
        for j, eigenvector in enumerate(eigenvectors):
            prob_eigenvector = []
            for i in range(iterations):
                start = ((measure + 1) % 2) * step + (i * 2 * step)
                start2 = measure * step + (i * 2 * step)
                prob_eigenvector.append(eigenvector[start2: start2 + step, :])
                eigenvector[start:start+step, :] = 0

            # Get the probability of measurement outcome for the chosen qubit. This is the eigenvalue times the absolute
            # square of the non-zero value for the qubit present in the eigenvector
            prob_eigenvector = np.array(prob_eigenvector).flatten()
            if np.count_nonzero(prob_eigenvector) != 0:
                non_zero_items = prob_eigenvector[np.flatnonzero(prob_eigenvector)]
                prob += eigenvalues[j]*np.sum(abs(non_zero_items)**2)
        prob = np.round(prob, 10)

        # Create the new density matrix that is the result of the measurement outcome
        if prob > 0:
            result = np.zeros(self.density_matrix.shape)
            for i, eigenvalue in enumerate(eigenvalues):
                eigenvector = eigenvectors[i]
                result += eigenvalue * CT(eigenvector)

            return prob, sp.csr_matrix(np.round(result/np.trace(result), 10))

        return prob, sp.csr_matrix((self.d, self.d))
    """
        --------------------------------------
            Density Matrix calculus Methods
        --------------------------------------     
    """

    def diagonalise(self, option=2):
        if option == 0:
            return eigh(self.density_matrix.toarray(), eigvals_only=True)
        if option == 1:
            return eigh(self.density_matrix.toarray())[1]
        if option == 2:
            return eigh(self.density_matrix.toarray())

    def get_non_zero_prob_eigenvectors(self):
        eigenvalues, eigenvectors = self.diagonalise()
        non_zero_eigenvalues_index = np.argwhere(np.round(eigenvalues, 10) != 0).flatten()
        eigenvectors_list = []

        for index in non_zero_eigenvalues_index:
            eigenvector = sp.csr_matrix(np.round(eigenvectors[:, index].reshape(self.d, 1), 8))
            eigenvectors_list.append(eigenvector)

        return eigenvalues[non_zero_eigenvalues_index], eigenvectors_list

    def print_non_zero_prob_eigenvectors(self):
        eigenvalues, eigenvectors = self.get_non_zero_prob_eigenvectors()

        print_line = "\n\n ---- Eigenvalues and Eigenvectors ---- \n\n"
        for i, eigenvalue in enumerate(eigenvalues):
            print_line += "eigenvalue: {}\n\neigenvector:\n {}\n---\n".format(eigenvalue, eigenvectors[i].toarray())

        print(print_line + "\n ---- End Eigenvalues and Eigenvectors ----\n")

    def decompose_statevector(self):
        # Obtain statevector by diagonalising density matrix and finding the non-zero prob eigenvectors
        non_zero_eigenvalues, non_zero_eigenvectors = self.get_non_zero_prob_eigenvectors()

        solutions = []
        for k, eigenvector in enumerate(non_zero_eigenvectors):
            non_zero_eigenvector_value_indices = np.argwhere(eigenvector.toarray().flatten() != 0).flatten()

            eigenvector_states = []
            for index in non_zero_eigenvector_value_indices:
                eigenvector_states_split = []
                state_vector_repr = [int(bit) for bit in "{0:b}".format(index).zfill(qc.num_qubits)]
                for state in state_vector_repr:
                    if state == 0:
                        eigenvector_states_split.append(copy.copy(ket_0))
                    else:
                        eigenvector_states_split.append(copy.copy(ket_1))

                # Save the sign of the non-zero index only on one of the two states. The copy is also used to this end,
                # such that the ket_1 will not in general be altered
                eigenvector_states_split[0] *= np.sign(eigenvector[index])

                eigenvector_states.append(eigenvector_states_split)
            solutions.append(eigenvector_states)

        return solutions

    def get_kraus_operator(self, operation):
        solutions = self.decompose_statevector()
        kraus_ops = []

        for eigenvector_states in solutions:
            outcome = int(self.num_qubits/2) * [None]
            for eigenvector_states_split in eigenvector_states:
                qubit = 0
                for i in range(0, len(eigenvector_states_split), 2):
                    if outcome[qubit] is None:
                        outcome[qubit] = CT(eigenvector_states_split[i], eigenvector_states_split[i+1])
                    else:
                        outcome[qubit] += CT(eigenvector_states_split[i], eigenvector_states_split[i+1])
                    qubit += 1

            for i, op in enumerate(outcome):
                kraus_ops.append(op)

        return kraus_ops

    """
        -----------------------------
            Circuit drawing Methods
        -----------------------------     
    """

    def draw_circuit(self):
        legenda = "--- Circuit ---\n\n @ = noisy Bell-pair, # = perfect Bell-pair, o = control qubit " \
                           "(with target qubit at same level), [X,Y,Z,H] = gates, M = Measurement\n"
        init = self._draw_init()
        self._draw_gates(init)
        init[-1] += "\n\n"
        print(legenda, *init)

    def _draw_init(self):
        init_state_repr = []
        for i in self._qubit_array:
            init_state_repr.append("\n\n{} ---".format(state_repr(i)))
        return init_state_repr

    def _draw_gates(self, init):
        for gate_item in self._draw_order:
            gate = next(iter(gate_item))
            value = gate_item[gate]
            if type(value) == tuple:
                control = "o"
                if gate == "#" or gate == "@":
                    control = gate
                cqubit = value[0]
                tqubit = value[1]
                init[cqubit] += "---{}---".format(control)
                init[tqubit] += "---{}---".format(gate)
            else:
                init[value] += "---{}---".format(gate)

            for a, b in it.combinations(enumerate(init), 2):
                if len(init[a[0]]) < len(init[b[0]]):
                    init[a[0]] += "-------"
                elif len(init[a[0]]) > len(init[b[0]]):
                    init[b[0]] += "-------"

    def _add_draw_operation(self, operation, *args):
        item = {operation: args}
        self._draw_order.append(item)

    """
        -----------------------------
            Gate Noise Methods
        -----------------------------     
    """

    def _N_single(self, pg, tqubit):
        self.density_matrix = sp.csr_matrix((1 - pg) * self.density_matrix +
                                            (pg / 3) * self._sum_pauli_error_single(tqubit))

    def _N(self, pg, cqubit, tqubit):

        self.density_matrix = sp.csr_matrix((1 - pg) * self.density_matrix +
                                            (pg / 15) * self._sum_pauli_error(cqubit, tqubit))

    def _sum_pauli_error_single(self, qubit):
        matrices = [X, Y, Z]
        result = np.zeros(self.density_matrix.shape)

        for i in matrices:
            pauli_error = self._create_1_qubit_gate(i, qubit)
            result = result + pauli_error.dot(CT(self.density_matrix, pauli_error))
        return result

    def _sum_pauli_error(self, qubit1, qubit2):
        matrices = [X, Y, Z, I]
        qubit2_matrices = []

        result = sp.csr_matrix(self.density_matrix.shape)
        for i in range(len(matrices)):
            # Create the full system 1 qubit gate for qubit1
            A = self._create_1_qubit_gate(matrices[i], qubit1)
            for j in range(len(matrices)):
                # Create full system 1-qubit gate for qubit2, only once for every gate
                if i == 0:
                    qubit2_matrices.append(self._create_1_qubit_gate(matrices[j], qubit2))

                # Skip the I*I case
                if i == j == len(matrices) - 1:
                    continue

                B = qubit2_matrices[j]
                result = result + (A * B).dot(CT(self.density_matrix, (A * B)))

        return sp.csr_matrix(result)

    def __repr__(self):
        density_matrix = self.density_matrix.toarray() if self.num_qubits < 4 else self.density_matrix
        return "\nCircuit density matrix:\n\n{}\n\n".format(density_matrix)


if __name__ == "__main__":
    start = time.time()
    qc = Circuit(3, init_type=2, noise=True, pg=0.09, pm=0.09)
    for i in range(1, qc.num_qubits, 2):
        qc.create_bell_pair([(i, i+1)])
    for i in range(1, qc.num_qubits, 2):
        qc.CNOT(0, i)
    qc.measure_first_N_qubits(1)

    for operator in qc.get_kraus_operator([X]):
        print(operator.toarray())
    qc.print_non_zero_prob_eigenvectors()
    print("The run took {} seconds".format(time.time() - start))
