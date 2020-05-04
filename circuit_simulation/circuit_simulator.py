from circuit_simulation.basic_operations import (
    CT, KP, state_repr, get_value_by_prob, trace, gate_name, fidelity
)
import numpy as np
import time
from scipy import sparse as sp
import itertools as it
import copy
from scipy.linalg import eig
import pickle


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


class QuantumCircuit:
    """
        QuantumCircuit(num_qubits, init_type=0, noise=False, pg=0.01, pm=0.01)

            Create a QuantumCircuit object

            A QuantumCircuit consists of qubits on which various operations can be applied.
            From this information about the density matrix of the system and others can be
            gathered.

            Parameters
            ----------
            num_qubits : int
                The amount of qubits the system contains.
            init_type : int [0-3], optional, default=0
                Determines how the system is initialised. All these options do NOT include noise.
                The options are:

                0 ->    The system is initialised with all qubits being in the |0> state.
                1 ->    Almost the same as 0, but the first qubit is in the |+> state
                2 ->    The system is initialised with a perfect Bell-pair between all adjacent
                        qubits.
                3 ->    The system is initialised with the first qubit being the |+> state and the
                        rest of the qubits is in the |0> state. The on every qubit a CNOT gate is
                        applied with the first qubit being the control qubit.

            noise : bool, optional, default=False
                Will apply noise on every operation that is applied to the QuantumCircuit object,
                unless specified otherwise.
            pg : float [0-1], optional, default=0.01
                The overall amount of gate noise that will be applied when 'noise' is set to True.
            pm : float [0-1], optional, default=0.01
                The overall amount of measurement error that will be applied when 'noise' set to
                True.

            Attributes
            ----------
            num_qubits : int
                The number of qubits present in the system.
                *** NUMBER IS NOT DEFINITE AND CAN AND WILL BE CHANGED BY SOME METHODS ***
            d : int
                Dimension of the system. This is 2**num_qubits
            noise: bool, optional, default=False
                If there is general noise present in the system. This will add noise to the gate
                and measurement operations applied to the system.
            pg : float [0-1], optional, default=0.01
                The amount of gate noise present in the system. Will only be applied if 'noise' is True.
            pm: float [0-1], optional, default=0.01
                The amount of measurement noise present in the system. Will only be applied if 'noise' is True.
    """

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
            self.density_matrix = self._init_density_matrix()
        elif init_type == 1:
            self.density_matrix = self._init_density_matrix_first_qubit_ket_p()
        elif init_type == 2:
            self.density_matrix = self._init_density_matrix_bell_pair_state()
        elif init_type == 3:
            self.density_matrix = self._init_density_matrix_ket_p_and_CNOTS()

    """
        -------------------------
            Init Methods
        -------------------------     
    """

    def _init_density_matrix(self):
        """ Realises init_type option 0. See class description for more info, """

        state_vector = KP(*self._qubit_array)
        return sp.csr_matrix(CT(state_vector, state_vector))

    def _init_density_matrix_first_qubit_ket_p(self):
        """ Realises init_type option 1. See class description for more info, """

        self._qubit_array[0] = ket_p

        density_matrix = sp.lil_matrix((self.d, self.d))
        density_matrix[0, 0] = 1 / 2
        density_matrix[0, self.d/2] = 1 / 2
        density_matrix[self.d/2, 0] = 1 / 2
        density_matrix[self.d/2, self.d/2] = 1 / 2

        return density_matrix

    def _init_density_matrix_bell_pair_state(self, draw=True):
        """ Realises init_type option 2. See class description for more info, """

        bell_pair_rho = sp.lil_matrix((4, 4))
        bell_pair_rho[0, 0], bell_pair_rho[3, 0], bell_pair_rho[0, 3], bell_pair_rho[3, 3] = 1/2, 1/2, 1/2, 1/2
        density_matrix = bell_pair_rho
        for i in range(0, self.num_qubits, 2):
            if i == 0 and draw:
                self._draw_order.append({"#": (i, i+1)})
                continue

            density_matrix = sp.kron(bell_pair_rho, density_matrix)
            if draw:
                self._draw_order.append({"#": (i, i+1)})

        return density_matrix

    def _init_density_matrix_ket_p_and_CNOTS(self):
        """ Realises init_type option 3. See class description for more info, """

        # Set ket_p as first qubit of the qubit array (mainly for proper drawing of the citcuit)
        self._qubit_array[0] = ket_p

        density_matrix = sp.lil_matrix((self.d, self.d))
        density_matrix[0, 0] = 1 / 2
        density_matrix[0, self.d-1] = 1 / 2
        density_matrix[self.d-1, 0] = 1 / 2
        density_matrix[self.d-1, self.d-1] = 1 / 2

        for i in range(1, self.num_qubits):
            self._draw_order.append({"X": (0, i)})

        return density_matrix

    """
        ------------------------------
            Setter and getter Methods
        ------------------------------
    """

    def set_qubit_states(self, qubit_dict):
        """
        qc.set_qubit_states(dict)

            Sets the initial state of the specified qubits in the dict according to the specified state.
            *** METHOD SHOULD ONLY BE USED IN THE INITIALISATION PHASE OF THE CIRCUIT. SHOULD NOT BE USED
            AFTER OPERATIONS HAVE BEEN APPLIED TO THE CIRCUIT IN ORDER TO PREVENT ERRORS. ***

            Parameters
            ----------
            qubit_dict : dict
                Dictionary with the keys being the number of the qubits to be modified (first qubit is 0)
                and the value being the state the qubit should be in

            Example
            -------
            qc.set_qubit_state({0 : ket_1}) --> This sets the first qubit to the ket_1 state
        """
        for tqubit, state in qubit_dict.items():
            self._qubit_array[tqubit] = state
        self._init_density_matrix()

    def get_begin_states(self):
        """ Returns the initial state vector of the qubits """
        return KP(*self._qubit_array)

    def create_bell_pairs(self, qubits):
        """
        qc.create_bell_pair(qubits)

            Creates Bell pairs between the specified qubits.
            *** THIS WILL ONLY WORK PROPERLY WHEN THE SPECIFIED QUBITS ARE IN NO WAY ENTANGLED AND THE
            STATE OF THE QUBITS IS |0> ***

            Parameters
            ----------
            qubits : list
                List containing tuples with the pairs of qubits that should form a Bell pair

            Example
            -------
            qc.create_bell_pairs([(0, 1), (2, 3), (4,5)]) --> Creates Bell pairs between qubit 0 and 1,
            between qubit 2 and 3 and between qubit 4 and 5.
        """
        for qubit1, qubit2 in qubits:
            self.H(qubit1, noise=False, draw=False)
            self.CNOT(qubit1, qubit2, noise=False, draw=False)
            self._draw_order.append({"#": (qubit1, qubit2)})

    def create_N_noisy_bell_pairs(self, N, pn=0.1):
        """
        qc.create_bell_pair(N, pn=0.1)

            This appends noisy Bell pairs on the top of the system. The noise is based on network noise
            modeled as (paper: https://www.nature.com/articles/ncomms2773.pdf)

                rho_raw = (1 - 4/3*pn) |psi><psi| + pn/3 * I,

            in which |psi> is a perfect Bell state.
            *** THIS METHOD APPENDS THE QUBITS TO THE TOP OF THE SYSTEM. THIS MEANS THAT THE AMOUNT OF
            QUBITS IN THE SYSTEM WILL GROW WITH '2N' ***

            Parameters
            ----------
            N : int
                Number of noisy Bell pairs that should be added to the top of the system.
            pn : float [0-1], optional, default=0.1
                The amount of network noise present

            Example
            -------
            qc.create_bell_pairs([(0, 1), (2, 3), (4,5)]) --> Creates Bell pairs between qubit 0 and 1,
            between qubit 2 and 3 and between qubit 4 and 5.
        """
        for i in range(0, 2*N, 2):
            self._qubit_array.insert(0, ket_0)
            self._qubit_array.insert(0, ket_0)
            self.num_qubits += 2
            self.d = self.num_qubits**2
            rho = sp.lil_matrix((4, 4))
            rho[0, 0], rho[0, 3], rho[3, 0], rho[3, 3] = 1/2, 1/2, 1/2, 1/2

            self.density_matrix = sp.kron((1 - 4*pn/3) * rho + pn/3 * sp.eye(4, 4), self.density_matrix)

            self._draw_order.append({"@": (i, i+1)})

    def add_top_qubit(self, qubit_state=ket_0):
        """
        qc.add_top_qubit(qubit_state=ket_0)

            Method appends a qubit with a given state to the top of the system.
            *** THE METHOD APPENDS A QUBIT, WHICH MEANS THAT THE AMOUNT OF QUBITS IN THE SYSTEM WILL
            GROW WITH 1 ***

            Parameters
            ----------
            qubit_state : array, optional, default=ket_0
                Qubit state, a normalised vector of dimension 2x1
        """
        self._qubit_array.insert(0, qubit_state)
        self.num_qubits += 1
        self.d = 2**self.num_qubits

        self.density_matrix = KP(CT(qubit_state), self.density_matrix)

    """
        -----------------------------
            One-Qubit Gate Methods
        -----------------------------     
    """

    def apply_1_qubit_gate(self, gate, tqubit, noise=None, pg=None, draw=True):
        """
            qc.apply_1_qubit_gate(gate, tqubit, noise=None, pg=None, draw=True)

                Applies a one qubit gate to the specified target qubit. This will update the density
                matrix of the system accordingly.

                Parameters
                ----------
                gate : ndarray
                    Array of dimension 2x2, examples are the well-known pauli matrices (X, Y, Z)
                tqubit : int
                    Integer that indicates the target qubit. Note that the qubit counting starts at
                    0.
                noise : bool, optional, default=None
                    Determines if the gate is noisy. When the QuantumCircuit object is initialised
                    with the 'noise' parameter to True, this parameter will also evaluate to True if
                    not specified otherwise.
                pg : float [0-1], optional, default=None
                    Specifies the amount of gate noise if present. If the QuantumCircuit object is
                    initialised with a 'pg' parameter, this will be used if not specified otherwise
                draw : bool, optional, default=True
                    If true, the specified gate will appear when the circuit is visualised.
        """
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
        """
            Private method that is used to create the 1 qubit gate matrix used in for eaxmple the
            apply_1_qubit_gate method.

            Parameters
            ----------
            gate : ndarray
                Array of dimension 2x2, examples are the well-known pauli matrices (X, Y, Z)
            tqubit : int
                Integer that indicates the target qubit. Note that the qubit counting starts at
                0.

            Returns
            -------
            1_qubit_gate : sparse matrix with dimensions equal to the density_matirx attribute
                Returns a matrix with dimensions equal to the dimensions of the density matrix of
                the system.
        """
        if np.array_equal(gate, I):
            return sp.eye(self.d, self.d)

        first_id, second_id = self._create_identity_operations(tqubit)

        return sp.csr_matrix(KP(first_id, gate, second_id))

    def _create_identity_operations(self, tqubit):
        """
            Private method that is used to efficiently create identity matrices, based on the target
            qubit specified. These matrices will work on the qubits other than the target qubit

            Parameters
            ----------
            tqubit : int
                Integer that indicates the target qubit. Note that the qubit counting starts at
                0.

            Returns
            -------
            first_id : sparse identity matrix
                Sparse identity matrix that will work on the qubits prior to the target qubit. If the target
                qubit is the first qubit, the value will be 'None'
            second_id : sparse identity matrix
                Sparse identity matrix that will work on the qubits following after the target qubit. If the
                target qubit is the last qubit, the value will be 'None'
        """
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
        """ Applies the pauli X gate to the specified target qubit. See apply_1_qubit_gate for more info """
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        for _ in range(times):
            self.apply_1_qubit_gate(X, tqubit, noise, pg, draw)

    def Z(self, tqubit, times=1, noise=None, pg=None, draw=True):
        """ Applies the pauli Z gate to the specified target qubit. See apply_1_qubit_gate for more info """
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        for _ in range(times):
            self.apply_1_qubit_gate(Z, tqubit, noise, pg, draw)

    def Y(self, tqubit, times=1, noise=None, pg=None, draw=True):
        """ Applies the pauli Y gate to the specified target qubit. See apply_1_qubit_gate for more info """
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        for _ in range(times):
            self.apply_1_qubit_gate(Y, tqubit, noise, pg, draw)

    def H(self, tqubit, times=1, noise=None, pg=None, draw=True):
        """ Applies the Hadamard gate to the specified target qubit. See apply_1_qubit_gate for more info """
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
        """
            Applies a two qubit gate according to the specified control and target qubits. This will update the density
            matrix of the system accordingly.

            Parameters
            ----------
            gate : ndarray
                Array of dimension 2x2, examples are the well-known pauli matrices (X, Y, Z)
            cqubit : int
                Integer that indicates the control qubit. Note that the qubit counting starts at 0
            tqubit : int
                Integer that indicates the target qubit. Note that the qubit counting starts at 0.
            noise : bool, optional, default=None
                Determines if the gate is noisy. When the QuantumCircuit object is initialised
                with the 'noise' parameter to True, this parameter will also evaluate to True if
                not specified otherwise.
            pg : float [0-1], optional, default=None
                Specifies the amount of gate noise if present. If the QuantumCircuit object is
                initialised with a 'pg' parameter, this will be used if not specified otherwise
            draw : bool, optional, default=True
                If true, the specified gate will appear when the circuit is visualised.
        """
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

                1. I#I#I + I#I#I
                2. I#|0><0|#I + I#|1><1|#I
                3. I#|0><0|#I + X_t#|1><1|#I

        (In which '#' is the Kronecker Product) (https://quantumcomputing.stackexchange.com/questions/4252/
        how-to-derive-the-cnot-matrix-for-a-3-qbit-system-where-the-control-target-qbi)

        Parameters
        ----------
        gate : array
            Array of dimension 2x2, examples are the well-known pauli matrices (X, Y, Z)
        cqubit : int
            Integer that indicates the control qubit. Note that the qubit counting starts at 0.
        tqubit : int
            Integer that indicates the target qubit. Note that the qubit counting starts at 0.
        """
        if cqubit == tqubit:
            raise ValueError("Control qubit cannot be the same as the target qubit!")

        gate_1 = self._create_1_qubit_gate(CT(ket_0), cqubit)
        gate_2 = self._create_1_qubit_gate(CT(ket_1), cqubit)

        one_qubit_gate = self._create_1_qubit_gate(gate, tqubit)
        gate_2 = one_qubit_gate.dot(gate_2)

        return sp.csr_matrix(gate_1 + gate_2)

    def CNOT(self, cqubit, tqubit, noise=None, pg=None, draw=True):
        """ Applies the CNOT gate to the specified target qubit. See apply_2_qubit_gate for more info """
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        self.apply_2_qubit_gate(X, cqubit, tqubit, noise, pg, draw)

    def CZ(self, cqubit, tqubit, noise=None, pg=None):
        """ Applies the CZ gate to the specified target qubit. See apply_2_qubit_gate for more info """
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        self.apply_2_qubit_gate(Z, cqubit, tqubit, noise, pg)

    """
        -----------------------------
            Gate Noise Methods
        -----------------------------     
    """

    def _N_single(self, pg, tqubit):
        """
            Private method to apply noise to the single qubit gates. This is done according to the equation

                N(rho) = (1-pg) * rho + pg/3 SUM_A [A * rho * A^], --> A in {X, Y, Z}

            in which '#' is the Kronecker product and ^ is the dagger (Hermitian conjugate).

            Parameters
            ----------
            pg : float [0-1]
                Indicates the amount of gate noise applied
            tqubit: int
                Integer that indicates the target qubit. Note that the qubit counting starts at 0.
        """
        self.density_matrix = sp.csr_matrix((1 - pg) * self.density_matrix +
                                            (pg / 3) * self._sum_pauli_error_single(tqubit))

    def _N(self, pg, cqubit, tqubit):
        """
            Private method to apply noise to the single qubit gates. This is done according to the equation

                N(rho) = (1-pg)*rho + pg/15 SUM_A SUM_B [(A # B) rho (A # B)^], --> {A, B} in {X, Y, Z, I}

            in which '#' is the Kronecker product and ^ is the dagger (Hermitian conjugate).

            Parameters
            ----------
            pg : float [0-1]
                Indicates the amount of gate noise applied
            cqubit: int
                Integer that indicates the control qubit. Note that the qubit counting starts at 0.
            tqubit: int
                Integer that indicates the target qubit. Note that the qubit counting starts at 0.
        """
        self.density_matrix = sp.csr_matrix((1 - pg) * self.density_matrix +
                                            (pg / 15) * self._double_sum_pauli_error(cqubit, tqubit))

    def _sum_pauli_error_single(self, tqubit):
        """
            Private method that calculates the pauli gate sum part of the equation specified in _N_single
            method, namely

                SUM_A [A * rho * A^], --> A in {X, Y, Z}

            Parameters
            ----------
            tqubit: int
                Integer that indicates the target qubit. Note that the qubit counting starts at 0.

            Returns
            -------
            summed_matrix : sparse matrix
                Returns a sparse matrix which is the result of the equation mentioned above.
        """
        matrices = [X, Y, Z]
        summed_matrix = sp.csr_matrix((self.d, self.d))

        for i in matrices:
            pauli_error = self._create_1_qubit_gate(i, tqubit)
            summed_matrix = summed_matrix + pauli_error.dot(CT(self.density_matrix, pauli_error))
        return summed_matrix

    def _double_sum_pauli_error(self, qubit1, qubit2):
        """
            Private method that calculates the double pauli matrices sum part of the equation specified in _N
            method, namely

                SUM_B SUM_A [(A # B) * rho * (A # B)^], --> {A, B} in {X, Y, Z, I}

            in which '#' is the Kronecker product and ^ is the dagger (Hermitian conjugate).

            Parameters
            ----------
            qubit1: int
                Integer that indicates the either the target qubit or the control qubit. Note that the qubit counting
                starts at 0.
            qubit2 : int
                Integer that indicates the either the target qubit or the control qubit. Note that the qubit counting
                starts at 0.
            Returns
            -------
            summed_matrix : sparse matrix
                Returns a sparse matrix which is the result of the equation mentioned above.
        """
        matrices = [X, Y, Z, I]
        qubit2_matrices = []

        result = sp.csr_matrix(self.density_matrix.shape)
        for i in range(len(matrices)):
            # Create the full system 1-qubit gate for qubit1
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

    """
        --------------------------------------
            Measurement Methods
        --------------------------------------     
    """
    def measure_first_N_qubits(self, N, measure=0, noise=None, pm=None, basis="X"):
        """
            Method measures the first N qubits, given by the user, all in the 0 or 1 state.
            This will thus result in an even parity measurement. To also be able to enforce uneven
            parity measurements this should still be built!
            The density matrix of the system will be changed according to the measurement outcomes.
            *** MEASURED QUBITS WILL BE ERASED FROM THE SYSTEM AFTER MEASUREMENT, THIS WILL THUS
            DECREASE THE AMOUNT OF QUBITS IN THE SYSTEM WITH N ***

            Parameters
            ----------
            N : int
                Specifies the first n qubits that should be measured.
            measure : int [0 or 1], optional, default=0
                The measurement outcome for the qubits, either 0 or 1.
            noise : bool, optional, default=None
                 Whether or not the measurement contains noise.
            pm : float [0-1], optional, default=None
                The amount of measurement noise that is present (if noise is present).
            basis : str ["X" or "Z"], optional, default="X"
                Whether the measurement should be done in the X-basis or in the computational (Z) basis

        """
        if noise is None:
            noise = self.noise
        if pm is None:
            pm = self.pm

        for qubit in range(N):
            if basis == "X":
                self.H(qubit)

            self._measurement_first_qubit(measure, noise=noise, pm=pm)

            self._draw_order.append({"M": qubit})

        self.density_matrix = self.density_matrix / trace(self.density_matrix)

    def _measurement_first_qubit(self, measure=0, noise=True, pm=0.):
        """
            Private method that is used to measure the first qubit (qubit 0) in the system and removing it
            afterwards. If a 0 is measured, the upper left quarter of the density matrix 'survives'
            and if a 1 is measured the lower right quarter of the density matrix 'survives'.
            Noise is applied according to the equation

                (1-pm) * rho_p-correct + pm * rho_p-incorrect,

            where 'rho_p-correct' is the density matrix that should result after the measurement and
            'rho_p-incorrect' is the density matrix that results when the opposite measurement outcome
            is measured.

            Parameters
            ----------
            measure : int [0 or 1], optional, default=0
                The measurement outcome for the qubit, either 0 or 1.
            noise : bool, optional, default=None
                 Whether or not the measurement contains noise.
            pm : float [0-1], optional, default=0.
                The amount of measurement noise that is present (if noise is present).
        """

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

        # Remove the measured qubit from the system characteristics
        self.num_qubits -= 1
        self.d = 2**self.num_qubits

    def measure(self, qubit, measure=None, basis="X"):
        """
            Measurement that can be applied to any qubit and does NOT remove the qubit from the system.
            *** THIS METHOD IS VERY SLOW FOR LARGER SYSTEMS, SINCE IT DETERMINES THE SYSTEM STATE AFTER
            THE MEASUREMENT BY DIAGONALISING THE DENSITY MATRIX ***

            Parameters
            ----------
            qubit : int
                Indicates the qubit to be measured (qubit count starts at 0)
            measure : int [0 or 1], optional, default=None
                The measurement outcome for the qubit, either 0 or 1. If None, the method will choose
                randomly according to the probability of the outcome.
            basis : str ["X" or "Z"], optional, default="X"
                Whether the qubit is measured in the X-basis or in the computational basis (Z-basis)

        """
        if basis == "X":
            self.H(qubit, noise=False)

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
            self.H(qubit, noise=False)

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

        Parameters
        ----------
        qubit : int
            Indicates the qubit to be measured (qubit count starts at 0)
        measure : int [0 or 1], optional, default=0
            The measurement outcome for the qubit, either 0 or 1.
        eigenval : sparse matrix, optional, default=None
            For speedup purposes, the eigenvalues of the density matrix can be passed to the method. *** Keep in mind that
            this does require more memory and can therefore cause the program to stop working. ***
        eigenvec : sparse matrix, optional, deafault=None
            For speedup purposes, the eigenvectors of the density matrix can be passed to the method. *** Keep in mind that
            this does require more memory and can therefore cause the program to stop working. ***

        Returns
        -------
        prob = float [0-1]
            The probability of the specified measurement outcome.
        resulting_density_matrix : sparse matrix
            The density matrix that is the result of the specified measurement outcome
        """
        if eigenvec is None:
            eigenvalues, eigenvectors = self.get_non_zero_prob_eigenvectors()
        else:
            eigenvalues, eigenvectors = eigenval, copy.copy(eigenvec)

        iterations = 2**qubit
        step = int(self.d/(2**(qubit+1)))
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
            return eig(self.density_matrix.toarray(), eigvals_only=True)
        if option == 1:
            return eig(self.density_matrix.toarray())[1]
        if option == 2:
            return eig(self.density_matrix.toarray())

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
        # Obtain statevector by diagonalising density matrix and finding the non-zero probability eigenvectors
        non_zero_eigenvalues, non_zero_eigenvectors = self.get_non_zero_prob_eigenvectors()

        decomposed_statevector = []
        for eigenvector in non_zero_eigenvectors:
            # non_zero_eigenvector_value_indices = np.argwhere(eigenvector.toarray().flatten() != 0).flatten()
            non_zero_eigenvector_value_indices, _, values = sp.find(eigenvector)
            negative_value_indices, negative_qubit_indices = \
                self._find_negative_contributing_qubit(non_zero_eigenvector_value_indices, values)

            eigenvector_states = []
            for index in non_zero_eigenvector_value_indices:
                eigenvector_states_split = []
                eigenvector_index_value = np.sqrt(2*abs(eigenvector[index, 0]))
                state_vector_repr = [int(bit) for bit in "{0:b}".format(index).zfill(self.num_qubits)]
                for i, state in enumerate(state_vector_repr):
                    sign = -1 if i in negative_qubit_indices and index in negative_value_indices else 1
                    if state == 0:
                        eigenvector_states_split.append(sign * eigenvector_index_value * copy.copy(ket_0))
                    else:
                        eigenvector_states_split.append(sign * eigenvector_index_value * copy.copy(ket_1))

                eigenvector_states.append(eigenvector_states_split)
            decomposed_statevector.append(eigenvector_states)

        return non_zero_eigenvalues, decomposed_statevector

    def _find_negative_contributing_qubit(self, indices, values):
        negative_value_indices = np.where(values < 0)[0]
        if negative_value_indices.size == 0:
            return [], []

        bitstrings = []
        for index in indices[negative_value_indices]:
            bitstrings.append([int(bit) for bit in "{0:b}".format(index).zfill(self.num_qubits)])

        negative_indices = []
        for i in range(0, self.num_qubits, 2):
            row = np.array(bitstrings)[:, i]
            if len(set(row)) == 1:
                negative_indices.append(i)

        return indices[negative_value_indices], negative_indices

    """
        -----------------------------
            Superoperator Methods
        -----------------------------     
    """

    def get_superoperator(self):
        if self.num_qubits != 8:
            raise ValueError("Superoperator can only be determined for a system with 4 data qubits with one ancilla "
                             "qubit each. So the system should contain 8 qubits")

        operations = [X, Y, Z, I]

        with open("density_matrix.pkl", "rb") as f:
            initial_density_matrix = pickle.load(f)

        superoperator = {}

        for qubit1_op in operations:
            gate_qubit1 = self._create_1_qubit_gate(qubit1_op, 0)
            for qubit2_op in operations:
                gate_qubit2 = self._create_1_qubit_gate(qubit2_op, 2)
                for qubit3_op in operations:
                    gate_qubit3 = self._create_1_qubit_gate(qubit3_op, 4)
                    for qubit4_op in operations:
                        gate_qubit4 = self._create_1_qubit_gate(qubit4_op, 6)
                        total_error_gate = gate_qubit1 * gate_qubit2 * gate_qubit3 * gate_qubit4

                        error_density_matrix = total_error_gate * CT(initial_density_matrix, total_error_gate)

                        fid = round(fidelity(self.density_matrix, error_density_matrix).real, 6)

                        operators = [gate_name(qubit1_op), gate_name(qubit2_op), gate_name(qubit3_op),
                                     gate_name(qubit4_op)]

                        if fid != 0 and operators.count("I") >= 1:
                            if fid in superoperator:
                                current_value = superoperator[fid]
                                current_value.append(operators)
                                superoperator[fid] = current_value
                            else:
                                superoperator[fid] = [operators]

                            # uniqueness = np.unique(operators)
                            # if fid > 0.5 and uniqueness.size < 3 and I in uniqueness:
                            #     superoperator.append([fid, gate_name(qubit1_op), gate_name(qubit2_op),
                            #                           gate_name(qubit3_op), gate_name(qubit4_op)])
        for key, ops in superoperator.items():
            superoperator[key] = self._return_most_likely_option(ops)

        return superoperator

    def _return_most_likely_option(self, operators):
        id_count = []
        for op in operators:
            id_count.append(op.count("I"))

        most_likely_indices = np.argwhere(id_count == np.amax(id_count)).flatten()

        return np.array(operators)[most_likely_indices]

    def get_kraus_operator(self, operation):
        probabilities, decomposed_statevector = self.decompose_statevector()
        kraus_ops = []

        for eigenvector_states in decomposed_statevector:
            # Initialise a list that will be used to save the total operation matrix per qubit
            kraus_op_per_qubit = int(self.num_qubits/2) * [None]
            correction = 1/np.sqrt(2**int(self.num_qubits/2))

            for eigenvector_states_split in eigenvector_states:
                # For each eigenvector iterate over the one qubit state elements of the data qubits to create the
                # effective Kraus operators that happened on the specific data qubit
                for qubit, data_qubit_position in enumerate(range(0, len(eigenvector_states_split), 2)):
                    if kraus_op_per_qubit[qubit] is None:
                        kraus_op_per_qubit[qubit] = correction * CT(eigenvector_states_split[data_qubit_position],
                                                       eigenvector_states_split[data_qubit_position+1])
                        continue

                    kraus_op_per_qubit[qubit] += correction * CT(eigenvector_states_split[data_qubit_position],
                                                    eigenvector_states_split[data_qubit_position+1])

                kraus_ops.append(kraus_op_per_qubit)

        return zip(probabilities, kraus_ops)

    def print_kraus_operators(self):
        print_lines = ["\n---- Kraus operators per qubit ----\n"]
        for prob, operators_per_qubit in sorted(self.get_kraus_operator([I])):
            print_lines.append("\nProbability: {:.8}\n".format(prob.real))
            for data_qubit, operator in enumerate(operators_per_qubit):
                data_qubit_line = "Data qubit {}: \n {}\n".format(data_qubit, operator.toarray())
                operator_name = gate_name(operator.toarray())
                if operator_name is not None:
                    data_qubit_line += "which is equal to an {} operation\n\n".format(operator_name)
                print_lines.append(data_qubit_line)
        print_lines.append("\n---- End of Kraus operators per qubit ----\n\n")

        print(*print_lines)



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
                    init[a[0]] += (len(init[b[0]]) - len(init[a[0]])) * "-"
                elif len(init[a[0]]) > len(init[b[0]]):
                    init[b[0]] += (len(init[a[0]]) - len(init[b[0]])) * "-"

    def _add_draw_operation(self, operation, *args):
        item = {operation: args}
        self._draw_order.append(item)

    def __repr__(self):
        density_matrix = self.density_matrix.toarray() if self.num_qubits < 4 else self.density_matrix
        return "\nCircuit density matrix:\n\n{}\n\n".format(density_matrix)


if __name__ == "__main__":
    start = time.time()

    qc = QuantumCircuit(10, init_type=2, noise=True, pg=0.09, pm=0.09)
    for z in range(2, qc.num_qubits, 2):
        qc.CNOT(0, z)
    qc.measure_first_N_qubits(2)

    qc.draw_circuit()
    superoperator = qc.get_superoperator()
    for prob in sorted(superoperator):
        print("Probability: {}\n".format(prob/sum(superoperator.keys())))
        print(superoperator[prob])
        print("")
    print(sum(superoperator.keys()))
    print("The run took {} seconds".format(time.time() - start))