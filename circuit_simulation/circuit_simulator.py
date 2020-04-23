import numpy as np
from tqdm import tqdm
from scipy import sparse as sp
import itertools as it
import random
from scipy.linalg import eigh, inv
from pprint import pprint

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


def state_repr(state):
    if np.array_equal(state, ket_0):
        return "|0>"
    if np.array_equal(state, ket_1):
        return "|1>"
    if np.array_equal(state, ket_p):
        return "|+>"
    if np.array_equal(state, ket_m):
        return "|->"


def gate_name(gate):
    if np.array_equal(gate, X):
        return "X"
    if np.array_equal(gate, Y):
        return "Y"
    if np.array_equal(gate, Z):
        return "Z"
    if np.array_equal(gate, I):
        return "I"
    if np.array_equal(gate, H):
        return "H"


def KP(*args):
    result = None
    for state in args:
        if result is None:
            result = sp.csr_matrix(state)
            continue
        result = sp.csr_matrix(sp.kron(result, state))
    return sp.csr_matrix(result)


def _get_value_by_prob(array, p):
    r = random.random()
    index = 0
    while r >= 0 and index < len(p):
        r -= p[index]
        index += 1
    return array[index - 1]


def CT(state1, state2=None):
    state2 = state1 if state2 is None else state2
    return sp.csr_matrix(state1.dot(state2.conj().T))


class Circuit:

    def __init__(self, num_qubits, init_density=True, noise=False, pg=0.01, pm=0.01):
        self.num_qubits = num_qubits
        self.d = 2 ** num_qubits
        self.noise = noise
        self.pg = pg
        self.pm = pm
        self._qubit_array = num_qubits * [ket_0]
        self._draw_order = []

        if init_density:
            self._init_density_matrix()

    """
        -------------------------
            Init Methods
        -------------------------     
    """

    def _init_density_matrix(self):
        state_vector = KP(*self._qubit_array)
        self.density_matrix = sp.csr_matrix(CT(state_vector, state_vector))

    def set_qubit_states(self, dict):
        for tqubit, state in dict.items():
            self._qubit_array[tqubit] = state
        self._init_density_matrix()

    def get_begin_states(self):
        return KP(*self._qubit_array)

    """
        -----------------------------
            One-Qubit Gate Methods
        -----------------------------     
    """

    def apply_1_qubit_gate(self, gate, tqubit, noise=None, pg=None):
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        one_qubit_gate = self._create_1_qubit_gate(gate, tqubit)
        self.density_matrix = sp.csr_matrix(one_qubit_gate.dot(CT(self.density_matrix, one_qubit_gate)))

        if noise:
            self._N_single(pg, tqubit)

        self._draw_order.append({gate_name(gate): tqubit})

    def _create_1_qubit_gate(self, gate, tqubit):
        operations = self.num_qubits * [I]
        operations[tqubit] = gate

        return sp.csr_matrix(KP(*operations))

    def X(self, tqubit, times=1, noise=None, pg=None):
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        for _ in range(times):
            self.apply_1_qubit_gate(X, tqubit, noise, pg)

    def Z(self, tqubit, times=1, noise=None, pg=None):
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        for _ in range(times):
            self.apply_1_qubit_gate(Z, tqubit, noise, pg)

    def Y(self, tqubit, times=1, noise=None, pg=None):
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        for _ in range(times):
            self.apply_1_qubit_gate(Y, tqubit, noise, pg)

    def H(self, tqubit, times=1, noise=None, pg=None):
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        for _ in range(times):
            self.apply_1_qubit_gate(H, tqubit, noise, pg)

    """
        -----------------------------
            Two-Qubit Gate Methods
        -----------------------------     
    """

    def apply_2_qubit_gate(self, gate, cqubit, tqubit, noise=None, pg=None):
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg
        two_qubit_gate = self._create_2_qubit_gate(gate, cqubit, tqubit)

        self.density_matrix = sp.csr_matrix(two_qubit_gate.dot(CT(self.density_matrix, two_qubit_gate)))

        if noise:
            self._N(pg, cqubit, tqubit)

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
                3. I*|0><0|*I + X*|1><1|*X^(dagger)

        (In which * is the Kronecker Product) (https://quantumcomputing.stackexchange.com/questions/4252/
        how-to-derive-the-cnot-matrix-for-a-3-qbit-system-where-the-control-target-qbi)
        """
        gate_1 = self.num_qubits * [I]
        gate_2 = self.num_qubits * [I]
        gate_1[cqubit] = CT(ket_0, ket_0)
        gate_2[cqubit] = CT(ket_1, ket_1)
        gate_2[tqubit] = gate
        return sp.csr_matrix(KP(*gate_1) + KP(*gate_2))

    def CNOT(self, cqubit, tqubit, noise=None, pg=None):
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        self.apply_2_qubit_gate(X, cqubit, tqubit, noise, pg)

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

    def measure(self, qubit, measure=None, basis="X"):
        if basis == "X":
            qc.H(qubit, noise=False)

        # If no specific measurement outcome is given it is chosen by the hand of the probability
        if measure is None:
            prob1, density_matrix1 = self._measurement(qubit, measure=0)
            prob2, density_matrix2 = self._measurement(qubit, measure=1)

            self.density_matrix = _get_value_by_prob([density_matrix1, density_matrix2], [prob1, prob2])
        else:
            self.density_matrix = self._measurement(qubit, measure)[1]

        self._draw_order.append({"M": qubit})

        if basis == "X":
            qc.H(qubit, noise=False)

    def _measurement(self, qubit, measure=0):
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
        eigenvalues, eigenvectors = self.get_non_zero_prob_eigenvectors()

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
        result = np.zeros(self.density_matrix.shape)
        for i, eigenvalue in enumerate(eigenvalues):
            eigenvector = eigenvectors[i]
            result += eigenvalue * CT(eigenvector)

        return prob, sp.csr_matrix(np.round(result/np.trace(result), 10))
    """
        --------------------------------------
            Density Matrix calculus Methods
        --------------------------------------     
    """

    def diagonalise(self, option=3):
        if option == 0:
            return eigh(self.density_matrix.toarray())[0]
        if option == 1:
            return eigh(self.density_matrix.toarray())[1]
        else:
            return eigh(self.density_matrix.toarray())

    def decompose_statevector(self):
        # Obtain statevector by diagonalising density matrix and finding the non-zero prob eigenvectors
        non_zero_eigenvalues, non_zero_eigenvectors = self.get_non_zero_prob_eigenvectors()

        solutions = dict()
        for k, eigenvector in enumerate(non_zero_eigenvectors):
            non_zero_eigenvector_value_indices = np.argwhere(eigenvector.flatten() != 0).flatten()

            one_qubit_states = dict()
            for i, index in enumerate(non_zero_eigenvector_value_indices):
                result = []
                check = [k for k in range(0, qc.d, 2)]
                for j in range(qc.num_qubits):
                    if int(index / (2 ** (qc.num_qubits - (1 + j)))) not in check:
                        result.append(ket_1)
                    else:
                        result.append(ket_0)
                result[0] = np.sign(eigenvector[index]) * result[0]
                one_qubit_states[i] = result
            solutions[k] = one_qubit_states
        return solutions

    def get_non_zero_prob_eigenvectors(self):
        eigenvalues, eigenvectors = self.diagonalise()
        non_zero_eigenvalues_index = np.argwhere(np.round(eigenvalues, 10) != 0).flatten()
        eigenvectors_list = []

        for index in non_zero_eigenvalues_index:
            eigenvectors_list.append(eigenvectors[:, index].reshape(self.d, 1))

        return eigenvalues[non_zero_eigenvalues_index], np.round(eigenvectors_list, 10)

    def print_non_zero_prob_eigenvectors(self):
        eigenvalues, eigenvectors = self.get_non_zero_prob_eigenvectors()

        print_line = "\n\n ---- Eigenvalues and Eigenvectors ---- \n\n"
        for i, eigenvalue in enumerate(eigenvalues):
            print_line += "eigenvalue: {}\n\neigenvector:\n {}\n---\n".format(eigenvalue, eigenvectors[i])

        print(print_line + "\n ---- End Eigenvalues and Eigenvectors ----\n")

    def get_kraus_operator(self, qubit_a, qubit_b):
        states = self.decompose_statevector()
        kraus_op = None

        for key in states:
            for key2 in states[key]:
                one_qubit_state_list = states[key][key2]
                if kraus_op is None:
                    kraus_op = CT(one_qubit_state_list[qubit_a], one_qubit_state_list[qubit_b])
                    continue
                kraus_op = kraus_op + CT(one_qubit_state_list[qubit_a], one_qubit_state_list[qubit_b])

        return kraus_op

    """
        -----------------------------
            Circuit drawing Methods
        -----------------------------     
    """

    def draw_circuit(self):
        init = self._draw_init()
        self._draw_gates(init)
        init[-1] += "\n\n"
        print(*init)

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
                cqubit = value[0]
                tqubit = value[1]
                init[cqubit] += "---o---"
                init[tqubit] += "---{}---".format(gate)
            else:
                init[value] += "---{}---".format(gate)

            for a, b in it.combinations(enumerate(init), 2):
                if len(init[a[0]]) < len(init[b[0]]):
                    init[a[0]] += "-------"
                elif len(init[a[0]]) > len(init[b[0]]):
                    init[b[0]] += "-------"

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
        result = np.zeros(self.density_matrix.shape)

        for i in matrices:
            for j in matrices:
                if np.array_equal(i, j) and np.array_equal(i, I):
                    continue
                A = self._create_1_qubit_gate(i, qubit1)
                B = self._create_1_qubit_gate(j, qubit2)
                result = result + sp.csr_matrix(A * B).dot(CT(self.density_matrix, sp.csr_matrix(A * B)))

        return sp.csr_matrix(result)

    def __repr__(self):
        density_matrix = self.density_matrix.toarray() if self.num_qubits < 4 else self.density_matrix
        return "\nCircuit density matrix:\n\n{}\n\n".format(density_matrix)


if __name__ == "__main__":
    qc = Circuit(5, init_density=False, noise=True, pg=0.09)
    qc.set_qubit_states({0: ket_p})
    qc.CNOT(0, 1)
    qc.CNOT(0, 2)
    qc.CNOT(0, 3)
    qc.CNOT(0, 4)
    qc.measure(0)

    print(qc.get_kraus_operator(0, 1))
    qc.draw_circuit()
    # print(qc)
    # qc.print_non_zero_prob_eigenvectors()

