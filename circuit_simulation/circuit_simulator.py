import sys
sys.path.insert(1, '/Users/Paul/Documents/TU/Master/Afstuderen/forked_surface_code/oop_surface_code/')
from circuit_simulation.basic_operations import (
    CT, KP, state_repr, get_value_by_prob, trace, gate_name, fidelity, fidelity_elementwise, gate_name_to_array
)
import numpy as np
from scipy import sparse as sp
import itertools as it
import copy
from scipy.linalg import eig, eigh
import hashlib
import os
import re
from superoperator import SuperoperatorElement
from termcolor import colored
from itertools import combinations, permutations, product


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
                Dimension of the system. This is 2**num_qubits.
            noise: bool, optional, default=False
                If there is general noise present in the system. This will add noise to the gate
                and measurement operations applied to the system.
            pg : float [0-1], optional, default=0.01
                The amount of gate noise present in the system. Will only be applied if 'noise' is True.
            pm : float [0-1], optional, default=0.01
                The amount of measurement noise present in the system. Will only be applied if 'noise' is True.
            density_matrix : ndarray
                The density matrix of the system.
            _qubit_array : ndarray
                A list containing the initial state of the qubits.
            _draw_order : list of dict items
                A list containing dict items that specify the operations that should be drawn.
            _init_parameters : dict
                A dictionary containing the initial parameters of the system, including the '_qubit_array' and
                'density_matrix' attribute. The keys are the names of the attributes.

    """

    def __init__(self, num_qubits, init_type=None, noise=False, pg=0.01, pm=0.01, pn=0.1):
        self.num_qubits = num_qubits
        self.d = 2 ** num_qubits
        self.noise = noise
        self.pg = pg
        self.pm = pm
        self.pn = pn
        self._init_type = init_type
        self._qubit_array = num_qubits * [ket_0]
        self._draw_order = []
        self._user_operation_order = []
        self._effective_measurements = 0
        self._measured_qubits = []
        self.density_matrix = None

        if init_type == 0:
            self.density_matrix = self._init_density_matrix()
        elif init_type == 1:
            self.density_matrix = self._init_density_matrix_first_qubit_ket_p()
        elif init_type == 2:
            self.density_matrix = self._init_density_matrix_bell_pair_state()
        elif init_type == 3:
            self.density_matrix = self._init_density_matrix_ket_p_and_CNOTS()

        self._init_parameters = self._save_init_parameters()

    """
        ---------------------------------------------------------------------------------------------------------
                                                    Init Methods
        ---------------------------------------------------------------------------------------------------------     
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
        density_matrix[0, self.d / 2] = 1 / 2
        density_matrix[self.d / 2, 0] = 1 / 2
        density_matrix[self.d / 2, self.d / 2] = 1 / 2

        return density_matrix

    def _init_density_matrix_bell_pair_state(self, draw=True):
        """ Realises init_type option 2. See class description for more info, """

        bell_pair_rho = sp.lil_matrix((4, 4))
        bell_pair_rho[0, 0], bell_pair_rho[3, 0], bell_pair_rho[0, 3], bell_pair_rho[3, 3] = 1 / 2, 1 / 2, 1 / 2, 1 / 2
        density_matrix = bell_pair_rho
        if draw:
            self._add_draw_operation("#", (0, 1))

        for i in range(2, self.num_qubits, 2):
            density_matrix = sp.kron(bell_pair_rho, density_matrix)
            if draw:
                self._add_draw_operation("#", (i, i + 1))

        return density_matrix

    def _init_density_matrix_ket_p_and_CNOTS(self):
        """ Realises init_type option 3. See class description for more info, """

        # Set ket_p as first qubit of the qubit array (mainly for proper drawing of the circuit)
        self._qubit_array[0] = ket_p

        density_matrix = sp.lil_matrix((self.d, self.d))
        density_matrix[0, 0] = 1 / 2
        density_matrix[0, self.d - 1] = 1 / 2
        density_matrix[self.d - 1, 0] = 1 / 2
        density_matrix[self.d - 1, self.d - 1] = 1 / 2

        for i in range(1, self.num_qubits):
            self._add_draw_operation("X", (0, i))

        return density_matrix

    def _save_init_parameters(self):
        init_params = {'num_qubits': self.num_qubits,
                       'd': self.d,
                       'init_type': self._init_type,
                       'noise': True,
                       'pm': self.pm,
                       'pg': self.pg,
                       'qubit_array': self._qubit_array,
                       'density_matrix': self.density_matrix}

        return init_params

    """
        ---------------------------------------------------------------------------------------------------------
                                                Setter and getter Methods
        ---------------------------------------------------------------------------------------------------------
    """

    def set_qubit_states(self, qubit_dict, user_operation=True):
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
            user_operation : bool, optional, default=True
                True if the user has requested the method and (else) False if it was invoked by an internal
                method.

            Example
            -------
            qc.set_qubit_state({0 : ket_1}) --> This sets the first qubit to the ket_1 state
        """
        if user_operation:
            self._user_operation_order.append({"set_qubit_states": [qubit_dict]})

        for tqubit, state in qubit_dict.items():
            self._qubit_array[tqubit] = state
        self._init_density_matrix()

    def get_begin_states(self):
        """ Returns the initial state vector of the qubits """
        return KP(*self._qubit_array)

    def create_bell_pairs(self, qubits, user_operation=True):
        """
        qc.create_bell_pair(qubits)

            Creates Bell pairs between the specified qubits.

            *** THIS WILL ONLY WORK PROPERLY WHEN THE SPECIFIED QUBITS ARE IN NO WAY ENTANGLED AND THE
            STATE OF THE QUBITS IS |0> ***

            Parameters
            ----------
            qubits : list
                List containing tuples with the pairs of qubits that should form a Bell pair
            user_operation : bool, optional, default=True
                True if the user has requested the method and (else) False if it was invoked by an internal
                method.

            Example
            -------
            qc.create_bell_pairs([(0, 1), (2, 3), (4,5)]) --> Creates Bell pairs between qubit 0 and 1,
            between qubit 2 and 3 and between qubit 4 and 5.
        """
        if user_operation:
            self._user_operation_order.append({"create_bell_pairs": [qubits]})

        for qubit1, qubit2 in qubits:
            self.H(qubit1, noise=False, draw=False, user_operation=False)
            self.CNOT(qubit1, qubit2, noise=False, draw=False, user_operation=False)
            self._add_draw_operation("#", (qubit1, qubit2))

    def create_bell_pairs_top(self, N, new_qubit=False, noise=None, pn=None,
                              user_operation=True):
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
            new_qubit: bool, optional, default=False
                If the creation of the Bell pair adds a new qubit to the drawing scheme (True) or reuses the top qubit
                (False) (this can be done in case the top qubit has been measured)
            noise : bool, optional, default=None
                Can be specified to force the creation of the Bell pairs noisy (True) or noiseless (False).
                If not specified (None), it will take the general noise parameter of the QuantumCircuit object.
            pn : float [0-1], optional, default=0.1
                The amount of network noise present
            user_operation : bool, optional, default=True
                True if the user has requested the method and (else) False if it was invoked by an internal
                method.

            Example
            -------
            qc.create_bell_pairs([(0, 1), (2, 3), (4,5)]) --> Creates Bell pairs between qubit 0 and 1,
            between qubit 2 and 3 and between qubit 4 and 5.
        """
        if user_operation:
            self._user_operation_order.append({"create_bell_pairs_top": [N, new_qubit, noise, pn]})
        if noise is None:
            noise = self.noise
        if not noise:
            pn = 0.0
        elif pn is None:
            pn = self.pn

        for i in range(0, 2 * N, 2):
            self.num_qubits += 2
            self.d = 2 ** self.num_qubits
            rho = sp.lil_matrix((4, 4))
            rho[0, 0], rho[0, 3], rho[3, 0], rho[3, 3] = 1 / 2, 1 / 2, 1 / 2, 1 / 2
            density_matrix = (1 - (4 * pn) / 3) * rho + (pn / 3) * sp.eye(4, 4)

            self.density_matrix = sp.kron(density_matrix, self.density_matrix) if (self.density_matrix is not None) \
                else density_matrix

            # Drawing the Bell Pair
            sign = '@' if noise else '#'
            if new_qubit:
                self._qubit_array.insert(0, ket_0)
                self._qubit_array.insert(0, ket_0)
                self._correct_for_n_top_qubit_additions(n=2)
            else:
                self._effective_measurements -= 2
            self._add_draw_operation(sign, (0, 1))

    def add_top_qubit(self, qubit_state=ket_0, user_operation=True):
        """
        qc.add_top_qubit(qubit_state=ket_0)

            Method appends a qubit with a given state to the top of the system.
            *** THE METHOD APPENDS A QUBIT, WHICH MEANS THAT THE AMOUNT OF QUBITS IN THE SYSTEM WILL
            GROW WITH 1 ***

            Parameters
            ----------
            qubit_state : array, optional, default=ket_0
                Qubit state, a normalised vector of dimension 2x1
            user_operation : bool, optional, default=True
                True if the user has requested the method and (else) False if it was invoked by an internal
                method.
        """
        if user_operation:
            self._user_operation_order.append({"add_top_qubit": [qubit_state]})

        self._qubit_array.insert(0, qubit_state)
        self.num_qubits += 1
        self.d = 2 ** self.num_qubits
        self._correct_for_n_top_qubit_additions()

        self.density_matrix = KP(CT(qubit_state), self.density_matrix)

    """
        ---------------------------------------------------------------------------------------------------------
                                                One-Qubit Gate Methods
        ---------------------------------------------------------------------------------------------------------     
    """

    def apply_1_qubit_gate(self, gate, tqubit, noise=None, pg=None, draw=True, user_operation=True):
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
                user_operation : bool, optional, default=True
                True if the user has requested the method and (else) False if it was invoked by an internal
                method.
        """
        if user_operation:
            self._user_operation_order.append({"apply_1_qubit_gate": [gate, tqubit, noise, pg, draw]})
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        one_qubit_gate = self._create_1_qubit_gate(gate, tqubit)
        self.density_matrix = sp.csr_matrix(one_qubit_gate.dot(CT(self.density_matrix, one_qubit_gate)))

        if noise:
            self._N_single(pg, tqubit)

        if draw:
            gate_repr = colored("~", 'red') + gate_name(gate) if noise else gate_name(gate)
            self._add_draw_operation(gate_repr, tqubit)

    def _create_1_qubit_gate(self, gate, tqubit, num_qubits=None):
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
        if num_qubits is None:
            num_qubits = self.num_qubits

        if np.array_equal(gate, I):
            return sp.eye(2 ** num_qubits, 2 ** num_qubits)

        first_id, second_id = self._create_identity_operations(tqubit, num_qubits=num_qubits)

        return sp.csr_matrix(KP(first_id, gate, second_id))

    def _create_identity_operations(self, tqubit, num_qubits=None):
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
        if num_qubits is None:
            num_qubits = self.num_qubits

        first_id = None
        second_id = None

        if tqubit == 0:
            second_id = sp.eye(2 ** (num_qubits - 1 - tqubit), 2 ** (num_qubits - 1 - tqubit))
        elif tqubit == num_qubits - 1:
            first_id = sp.eye(2 ** tqubit, 2 ** tqubit)
        else:
            first_id = sp.eye(2 ** tqubit, 2 ** tqubit)
            second_id = sp.eye(2 ** (num_qubits - 1 - tqubit), 2 ** (num_qubits - 1 - tqubit))

        return first_id, second_id

    def X(self, tqubit, times=1, noise=None, pg=None, draw=True, user_operation=True):
        """ Applies the pauli X gate to the specified target qubit. See apply_1_qubit_gate for more info """
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        for _ in range(times):
            self.apply_1_qubit_gate(X, tqubit, noise, pg, draw, user_operation=user_operation)

    def Z(self, tqubit, times=1, noise=None, pg=None, draw=True, user_operation=True):
        """ Applies the pauli Z gate to the specified target qubit. See apply_1_qubit_gate for more info """
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        for _ in range(times):
            self.apply_1_qubit_gate(Z, tqubit, noise, pg, draw, user_operation=user_operation)

    def Y(self, tqubit, times=1, noise=None, pg=None, draw=True, user_operation=True):
        """ Applies the pauli Y gate to the specified target qubit. See apply_1_qubit_gate for more info """
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        for _ in range(times):
            self.apply_1_qubit_gate(Y, tqubit, noise, pg, draw, user_operation=user_operation)

    def H(self, tqubit, times=1, noise=None, pg=None, draw=True, user_operation=True):
        """ Applies the Hadamard gate to the specified target qubit. See apply_1_qubit_gate for more info """
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        for _ in range(times):
            self.apply_1_qubit_gate(H, tqubit, noise, pg, draw, user_operation=user_operation)

    """
        ---------------------------------------------------------------------------------------------------------
                                                Two-Qubit Gate Methods
        ---------------------------------------------------------------------------------------------------------     
    """

    def apply_2_qubit_gate(self, gate, cqubit, tqubit, noise=None, pg=None, draw=True, user_operation=True):
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
            user_operation : bool, optional, default=True
                True if the user has requested the method and (else) False if it was invoked by an internal
                method.
        """
        if user_operation:
            self._user_operation_order.append({"apply_2_qubit_gate": [gate, cqubit, tqubit, noise, pg, draw]})
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg
        two_qubit_gate = self._create_2_qubit_gate(gate, cqubit, tqubit)

        self.density_matrix = sp.csr_matrix(two_qubit_gate.dot(CT(self.density_matrix, two_qubit_gate)))

        if noise:
            self._N(pg, cqubit, tqubit)

        if draw:
            gate_repr = colored("~", 'red') + gate_name(gate) if noise else gate_name(gate)
            self._add_draw_operation(gate_repr, (cqubit, tqubit))

    def _create_2_qubit_gate(self, gate, cqubit, tqubit, num_qubits=None):
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
        if num_qubits is None:
            num_qubits = self.num_qubits
        if cqubit == tqubit:
            raise ValueError("Control qubit cannot be the same as the target qubit!")

        gate_1 = self._create_1_qubit_gate(CT(ket_0), cqubit, num_qubits=num_qubits)
        gate_2 = self._create_1_qubit_gate(CT(ket_1), cqubit, num_qubits=num_qubits)

        one_qubit_gate = self._create_1_qubit_gate(gate, tqubit, num_qubits=num_qubits)
        gate_2 = one_qubit_gate.dot(gate_2)

        return sp.csr_matrix(gate_1 + gate_2)

    def CNOT(self, cqubit, tqubit, noise=None, pg=None, draw=True, user_operation=True):
        """ Applies the CNOT gate to the specified target qubit. See apply_2_qubit_gate for more info """
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        self.apply_2_qubit_gate(X, cqubit, tqubit, noise, pg, draw, user_operation=user_operation)

    def CZ(self, cqubit, tqubit, noise=None, pg=None, user_operation=True):
        """ Applies the CZ gate to the specified target qubit. See apply_2_qubit_gate for more info """
        if noise is None:
            noise = self.noise
        if pg is None:
            pg = self.pg

        self.apply_2_qubit_gate(Z, cqubit, tqubit, noise, pg, user_operation=user_operation)

    """
        ---------------------------------------------------------------------------------------------------------
                                            Protocol gate sequences
        ---------------------------------------------------------------------------------------------------------  
    """

    def single_selection(self, operation, new_qubit=False, measure=True, noise=None, pn=None, pm=None, pg=None,
                         user_operation=True):

        self.create_bell_pairs_top(1, new_qubit=new_qubit, noise=noise, pn=pn, user_operation=user_operation)
        self.apply_2_qubit_gate(operation, 0, 2, noise=noise, pg=pg, user_operation=user_operation)
        self.apply_2_qubit_gate(operation, 1, 3, noise=noise, pg=pg, user_operation=user_operation)
        if measure:
            self.measure_first_N_qubits(2, noise=noise, pm=pm, user_operation=user_operation)

    def double_selection(self, operation, new_qubit=False, noise=None, pn=None, pm=None, pg=None, user_operation=True):

        self.single_selection(operation, new_qubit=new_qubit, measure=False, noise=noise, pn=pn, pm=pm, pg=pg,
                              user_operation=user_operation)
        self.create_bell_pairs_top(1, new_qubit=new_qubit, noise=noise, pn=pn, user_operation=user_operation)
        self.CZ(0, 2, noise=noise, pg=pg, user_operation=user_operation)
        self.CZ(1, 3, noise=noise, pg=pg, user_operation=user_operation)
        self.measure_first_N_qubits(4, noise=noise, pm=pm, user_operation=user_operation)

    def single_dot(self, operation, qubit1, qubit2, measure=True, noise=None, pn=None, pm=None,
                   pg=None, user_operation=True):

        self.create_bell_pairs_top(1, noise=noise, pn=pn, user_operation=user_operation)
        self.single_selection(X, noise=noise, pn=pn, pm=pm, pg=pg, user_operation=user_operation)
        self.single_selection(Z, noise=noise, pn=pn, pm=pm, pg=pg, user_operation=user_operation)
        self.apply_2_qubit_gate(operation, 0, qubit1, noise=noise, pg=pg, user_operation=user_operation)
        self.apply_2_qubit_gate(operation, 1, qubit2, noise=noise, pg=pg, user_operation=user_operation)
        if measure:
            self.measure_first_N_qubits(2, noise=noise, pm=pm, user_operation=user_operation)

    def double_dot(self, operation, qubit1, qubit2, noise=None, pn=None, pm=None, pg=None,
                   user_operation=True):

        self.single_dot(operation, qubit1, qubit2, measure=False, noise=noise, pn=pn, pm=pm, pg=pg,
                        user_operation=user_operation)
        self.single_selection(Z, noise=noise, pn=pn, pm=pm, pg=pg, user_operation=user_operation)
        self.measure_first_N_qubits(2, noise=noise, pm=pm, user_operation=user_operation)

    """
        ---------------------------------------------------------------------------------------------------------
                                            Gate Noise Methods
        ---------------------------------------------------------------------------------------------------------  
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

        for matrix in matrices:
            pauli_error = self._create_1_qubit_gate(matrix, tqubit)
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
        ---------------------------------------------------------------------------------------------------------
                                                Measurement Methods
        ---------------------------------------------------------------------------------------------------------   
    """

    def measure_first_N_qubits(self, N, measure=0, uneven_parity=False, noise=None, pm=None, basis="X",
                               user_operation=True):
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
            uneven_parity : bool, optional, default=False
                If True, an uneven parity measurement outcome is forced on pairs of qubits.
            noise : bool, optional, default=None
                 Whether or not the measurement contains noise.
            pm : float [0-1], optional, default=None
                The amount of measurement noise that is present (if noise is present).
            basis : str ["X" or "Z"], optional, default="X"
                Whether the measurement should be done in the X-basis or in the computational basis (Z-basis)
            user_operation : bool, optional, default=True
                True if the user has requested the method and (else) False if it was invoked by an internal
                method.

        """
        if user_operation:
            self._user_operation_order.append({"measure_first_N_qubits": [N, measure, noise, pm, basis]})
        if noise is None:
            noise = self.noise
        if pm is None:
            pm = self.pm

        for qubit in range(N):
            if basis == "X":
                # Do not let the method draw itself, since the qubit will not be removed from the circuit drawing
                self.H(0, noise=noise, draw=False, user_operation=False)

            measure_new = measure
            if uneven_parity and qubit == 0:
                measure_new = abs(measure - 1)

            self._measurement_first_qubit(measure_new, noise=noise, pm=pm)
            self._add_draw_operation("{}M_{}:{}"
                                     .format((colored("~", 'red') if noise else ""), basis, measure_new), qubit)
        self._effective_measurements += N

    def _measurement_first_qubit(self, measure=0, noise=True, pm=0.):
        """
            Private method that is used to measure the first qubit (qubit 0) in the system and removing it
            afterwards. If a 0 is measured, the upper left quarter of the density matrix 'survives'
            and if a 1 is measured the lower right quarter of the density matrix 'survives'.
            Noise is applied according to the equation

                rho_noisy = (1-pm) * rho_p-correct + pm * rho_p-incorrect,

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

        density_matrix_0 = self.density_matrix[:int(self.d / 2), :int(self.d / 2)]
        density_matrix_1 = self.density_matrix[int(self.d / 2):, int(self.d / 2):]

        if measure == 0 and noise:
            density_matrix = (1 - pm) * density_matrix_0 + pm * density_matrix_1
        elif noise:
            density_matrix = (1 - pm) * density_matrix_1 + pm * density_matrix_0
        elif measure == 0:
            density_matrix = density_matrix_0
        else:
            density_matrix = density_matrix_1

        self.density_matrix = density_matrix / trace(density_matrix)

        # Remove the measured qubit from the system characteristics
        self.num_qubits -= 1
        self.d = 2 ** self.num_qubits

    def measure(self, qubit, measure=None, basis="X", user_operation=False):
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
            user_operation : bool, optional, default=True
                True if the user has requested the method and (else) False if it was invoked by an internal
                method.

        """
        if user_operation:
            self._user_operation_order.append({"measure": [qubit, measure, basis]})
        if basis == "X":
            self.H(qubit, noise=False, user_operation=False)

        # If no specific measurement outcome is given it is chosen by the hand of the probability
        if measure is None:
            # eigenvalues, eigenvectors = self.get_non_zero_prob_eigenvectors()
            prob1, density_matrix1 = self._measurement(qubit, measure=0)
            prob2, density_matrix2 = self._measurement(qubit, measure=1)

            self.density_matrix = get_value_by_prob([density_matrix1, density_matrix2], [prob1, prob2])
        else:
            self.density_matrix = self._measurement(qubit, measure)[1]

        self._add_draw_operation("M", qubit)

        if basis == "X":
            self.H(qubit, noise=False, user_operation=False)

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
            For speedup purposes, the eigenvalues of the density matrix can be passed to the method. *** Keep in mind
            that this does require more memory and can therefore cause the program to stop working. ***
        eigenvec : sparse matrix, optional, deafault=None
            For speedup purposes, the eigenvectors of the density matrix can be passed to the method. *** Keep in mind
            that this does require more memory and can therefore cause the program to stop working. ***

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

        iterations = 2 ** qubit
        step = int(self.d / (2 ** (qubit + 1)))
        prob = 0

        # Let measurement outcome determine the states that 'survive'
        for j, eigenvector in enumerate(eigenvectors):
            prob_eigenvector = []
            for i in range(iterations):
                start = ((measure + 1) % 2) * step + (i * 2 * step)
                start2 = measure * step + (i * 2 * step)
                prob_eigenvector.append(eigenvector[start2: start2 + step, :])
                eigenvector[start:start + step, :] = 0

            # Get the probability of measurement outcome for the chosen qubit. This is the eigenvalue times the absolute
            # square of the non-zero value for the qubit present in the eigenvector
            prob_eigenvector = np.array(prob_eigenvector).flatten()
            if np.count_nonzero(prob_eigenvector) != 0:
                non_zero_items = prob_eigenvector[np.flatnonzero(prob_eigenvector)]
                prob += eigenvalues[j] * np.sum(abs(non_zero_items) ** 2)
        prob = np.round(prob, 10)

        # Create the new density matrix that is the result of the measurement outcome
        if prob > 0:
            result = np.zeros(self.density_matrix.shape)
            for i, eigenvalue in enumerate(eigenvalues):
                eigenvector = eigenvectors[i]
                result += eigenvalue * CT(eigenvector)

            return prob, sp.csr_matrix(np.round(result / np.trace(result), 10))

        return prob, sp.csr_matrix((self.d, self.d))

    """
        ---------------------------------------------------------------------------------------------------------
                                            Density Matrix calculus Methods
        ---------------------------------------------------------------------------------------------------------     
    """

    def diagonalise(self, option=0):
        """" Returns the Eigenvalues and Eigenvectors of the density matrix. option=1 returns only the Eigenvalues"""
        if option == 0:
            return eig(self.density_matrix.toarray())
        if option == 1:
            return eigh(self.density_matrix.toarray(), eigvals_only=True)

    def get_non_zero_prob_eigenvectors(self, decimals=10):
        """
            Get the eigenvectors with non-zero eigenvalues.

            Parameters
            ----------
            decimals : int, optional, default=10
                Determines how the Eigenvalues should be rounded. Based on this rounding it will also be determined
                if the Eigenvalue is non-zero.

            Returns
            -------
            non_zero_eigenvalues : list
                List containing the non-zero eigenvalues.
            corresponding_eigenvectors : list
                List containing the eigenvectors corresponding to the non-zero Eigenvalues.
        """
        eigenvalues, eigenvectors = self.diagonalise()
        non_zero_eigenvalues_index = np.argwhere(np.round(eigenvalues, decimals) != 0).flatten()
        eigenvectors_list = []

        for index in non_zero_eigenvalues_index:
            eigenvector = sp.csr_matrix(np.round(eigenvectors[:, index].reshape(self.d, 1), 8))
            eigenvectors_list.append(eigenvector)

        return eigenvalues[non_zero_eigenvalues_index], eigenvectors_list

    def print_non_zero_prob_eigenvectors(self):
        """ Prints a clear overview of the non-zero Eigenvalues and their Eigenvectors to the console """
        eigenvalues, eigenvectors = self.get_non_zero_prob_eigenvectors()

        print_line = "\n\n ---- Eigenvalues and Eigenvectors ---- \n\n"
        for i, eigenvalue in enumerate(eigenvalues):
            print_line += "eigenvalue: {}\n\neigenvector:\n {}\n---\n".format(eigenvalue, eigenvectors[i].toarray())

        print(print_line + "\n ---- End Eigenvalues and Eigenvectors ----\n")

    def decompose_non_zero_eigenvectors(self):
        """
            Method to decompose the eigenvectors, with non-zero eigenvalues, into N-qubit states (in which N is
            the number of qubits present in the system) which on themselves are again decomposed in one-qubit states.
            Visualised for a random eigenvector of a 6 qubit system

            Eigenvector --> |000100> + |100000> + ... --> |0>#|0>#|0>#|1>#|0>#|0> + |1>#|0>#|0>#|0>#|0>#|0> + ...

            in which '#' is the Kronecker product.

            *** DOES NOT WORK PROPERLY WHEN MULTIPLE QUBITS OBTAINED AN EFFECTIVE PHASE, SINCE IT IS NOT YET
            FIGURED OUT HOW THESE MULTIPLE NEGATIVE CONTRIBUTIONS CAN BE TRACED BACK --> SEE MORE INFORMATION AT
            THE _FIND_NEGATIVE_CONTRIBUTING_QUBIT' METHOD ***

            Returns
            -------
            non_zero_eigenvalues : list
                List containing the non-zero eigenvalues.
            decomposed_eigenvectors : list
                A list containing each eigenvector (with a non-zero Eigenvalue) decomposed into a list of
                N-qubit states which is yet again decomposed into one-qubit states

        """
        non_zero_eigenvalues, non_zero_eigenvectors = self.get_non_zero_prob_eigenvectors()

        decomposed_eigenvectors = []
        for eigenvector in non_zero_eigenvectors:
            # Find all the values and indices of the non-zero elements in the eigenvector. Each of these elements
            # represents an N-qubit state. The N-qubit state corresponding to the index of the non-zero element of the
            # eigenvector is found by expressing the index in binary with the amount of bits equal to the amount
            # of qubits.
            non_zero_eigenvector_value_indices, _, values = sp.find(eigenvector)
            negative_value_indices, negative_qubit_indices = \
                self._find_negative_contributing_qubit(non_zero_eigenvector_value_indices, values)

            eigenvector_in_n_qubit_states = []
            for index in non_zero_eigenvector_value_indices:
                one_qubit_states_in_n_qubit_state = []
                eigenvector_index_value = np.sqrt(2 * abs(eigenvector[index, 0]))
                state_vector_repr = [int(bit) for bit in "{0:b}".format(index).zfill(self.num_qubits)]
                for i, state in enumerate(state_vector_repr):
                    sign = -1 if i in negative_qubit_indices and index in negative_value_indices else 1
                    if state == 0:
                        one_qubit_states_in_n_qubit_state.append(sign * eigenvector_index_value * copy.copy(ket_0))
                    else:
                        one_qubit_states_in_n_qubit_state.append(sign * eigenvector_index_value * copy.copy(ket_1))

                eigenvector_in_n_qubit_states.append(one_qubit_states_in_n_qubit_state)
            decomposed_eigenvectors.append(eigenvector_in_n_qubit_states)

        return non_zero_eigenvalues, decomposed_eigenvectors

    def _find_negative_contributing_qubit(self, non_zero_eigenvector_elements_indices,
                                          non_zero_eigenvector_elements_values):
        """
            returns the index of the qubit that obtained a phase (negative value). So for a
            4 qubit system (2 data qubits (_d), 2 ancilla qubits (_a))

            (|0_d, 0_a> -|1_d, 1_a>) # (|0_d, 0_a> + |1_d, 1_a>) = |0000> + |0011> - |1100> - |1111>

            Comparing the data qubits of the negative N-qubit states, we see that the first data qubit
            is always in the |1>, which is indeed the qubit that obtained the phase.

            *** THIS ONLY WORKS WHEN ONE QUBIT HAS OBTAINED A PHASE. SO ONLY ONE EFFECTIVE
            Z (OR Y) ON ONE OF THE QUBITS IN THE SYSTEM. SHOULD BE CHECKED IF IT IS POSSIBLE
            TO DETERMINE THIS IN EVERY SITUATION ***

            Parameters
            ----------
            non_zero_eigenvector_elements_indices : list
                List with the indices of non-zero elements of the eigenvector.
            non_zero_eigenvector_elements_values : list
                List that contains the values of the elements that are non-zero.

            Returns
            -------
            negative_value_indices : list
                List of indices that correspond to the negative elements in the Eigenvector
            negative_qubit_indices : list
                List of qubits that obtained a phase (negative value). For now this will only
                contain one qubit or no qubit index
        """
        # Get the indices of the negative values in the eigenvector
        negative_value_indices = np.where(non_zero_eigenvector_elements_values < 0)[0]
        if negative_value_indices.size == 0:
            return [], []

        # Get the N-qubit states that corresponds to the negative value indices
        bitstrings = []
        for negative_value_index in non_zero_eigenvector_elements_indices[negative_value_indices]:
            bitstrings.append([int(bit) for bit in "{0:b}".format(negative_value_index).zfill(self.num_qubits)])

        # Check for each data qubits (all the even qubits) if it is in the same state in each N-qubit state.
        # If this is the case then this data qubit is the negative contributing qubits (if only one qubit
        # has obtained an effective phase).
        negative_qubit_indices = []
        for i in range(0, self.num_qubits, 2):
            row = np.array(bitstrings)[:, i]
            if len(set(row)) == 1:
                negative_qubit_indices.append(i)

        return non_zero_eigenvector_elements_indices[negative_value_indices], negative_qubit_indices

    """
        ---------------------------------------------------------------------------------------------------------
                                                Superoperator Methods
        ---------------------------------------------------------------------------------------------------------     
    """

    def get_superoperator(self, qubits, proj_type, save_noiseless_density_matrix=False, combine=True, most_likely=True,
                          print_to_console=True, file_name_noiseless=None, file_name_measerror=None):
        """
            Returns the superoperator for the system. The superoperator is determined by taking the fidelities
            of the density matrix of the system [rho_real] and the density matrices obtained with any possible
            combination of error on the 4 data qubits in a noiseless version of the system
            [(ABCD) rho_ideal (ABCD)^]. Thus in equation form

            F[rho_real, (ABCD) * rho_ideal * (ABCD)^], {A, B, C, D} in {X, Y, Z, I}

            The fidelity is equal to the probability of this specific error, the combination of (ABCD), happening.

            Parameters
            __________
            qubits : list
                List of qubits of which the superoperator should be calculated. Only for these qubits it will be
                checked if certain errors occured on them. This is necessary to specify in case the circuit contains
                ancilla qubits that should not be evaluated. **The index of the qubits should be the index of the
                resulting density matrix, thus in case of measurements this can differ from the initial indices!!**
            proj_type : str, options: "X" or "Z"
                Specifies the type of stabilizer for which the superoperator should be calculated. This value is
                necessary for the postprocessing of the superoperator results if 'combine' is set to True.
            save_noiseless_density_matrix : bool, optional, default=True
                Whether or not the calculated noiseless (ideal) version of the circuit should be saved.
                This saved matrix will a next time be used for speedup if the same system is analysed with this method.
            combine : bool, optional, default=True
                Combines the error configuration on the data qubits that are equal up to permutation. This effectively
                means that for example [I, I, I, X] and [X, I, I, I] will be combined to one term [I, I, I, X] with the
                probabilities summed.
            most_likely : bool, optional, default=True
                Will choose the most likely configuration of degenerate configurations. This effectively means that the
                configuration with the highest amount of identity operators will be chosen. Only works if 'combine' is
                also set to True.
            print_to_console : bool, optional, default=True
                Whether the result should be printed in a clear overview to the console.
            file_name_noiseless : str, optional, default=None
                file name of the noiseless variant of the density matrix of the noisy system. Use this option if density
                matrix has been named manually and this one should be used for the calculations.
            file_name_measerror : str, optional, default=None
                file name of the noiseless variant with measurement error of the density matrix of the noisy system.
                Use this option if density matrix has been named manually and this one should be used for the
                calculations.
        """
        noiseless_density_matrix = self._get_noiseless_density_matrix(save=save_noiseless_density_matrix,
                                                                      file_name=file_name_noiseless)
        measerror_density_matrix = self._get_noiseless_density_matrix(measure_error=True,
                                                                      save=save_noiseless_density_matrix,
                                                                      file_name=file_name_measerror)

        measurement_applied = not np.array_equal(noiseless_density_matrix.toarray(), measerror_density_matrix.toarray())

        superoperator = []

        # Get all combinations of gates ([X, Y, Z, I]) possible on the given qubits
        all_gate_combinations = self._all_single_qubit_gate_possibilities(qubits)

        for combination in all_gate_combinations:
            total_error_gate = None
            for gate_dict in combination:
                gate = list(gate_dict.values())[0]
                if total_error_gate is None:
                    total_error_gate = gate
                    continue
                total_error_gate = total_error_gate * gate

            error_density_matrix = total_error_gate * CT(noiseless_density_matrix, total_error_gate)
            me_error_density_matrix = total_error_gate * CT(measerror_density_matrix, total_error_gate)

            fid_no_me = round(fidelity_elementwise(error_density_matrix, self.density_matrix), 12)
            fid_me = round(fidelity_elementwise(me_error_density_matrix, self.density_matrix), 12) \
                if measurement_applied else 0

            operators = [list(applied_gate.keys())[0] for applied_gate in combination]

            superoperator.append(SuperoperatorElement(fid_me, True, operators))
            superoperator.append(SuperoperatorElement(fid_no_me, False, operators))

        # Possible post-processing options for the superoperator
        if combine:
            superoperator = self._fuse_equal_config_up_to_permutation(superoperator, proj_type)
        if combine and most_likely:
            superoperator = self._remove_not_likely_configurations(superoperator)

        if print_to_console:
            self._print_superoperator(superoperator)

        return superoperator

    def _get_noiseless_density_matrix(self, measure_error=False, save=True, file_name=None):
        """
            Private method to calculate the noiseless variant of the density matrix.
            It traverses the operations on the system by the hand of the '_user_operation_order' attribute.

            Parameters
            ----------
            measure_error: bool, optional, default=False
                Specifies if the measurement outcome should be opposite of the ideal circuit.
            save : bool
                Whether or not the calculated noiseless version of the circuit should be saved.
                This saved matrix will a next time be used if the same system is analysed wth this method.
            file_name : str
                File name of the density matrix file that should be used as noiseless density matrix. Note that
                specifying this with an existing file name will directly return this density matrix.

            Returns
            -------
            noiseless_density_matrix : sparse matrix
                The density matrix of the current system, but without noise
        """
        if file_name is None:
            file_name = self._file_name_from_circuit(measure_error)

        # Check if the noiseless system has been calculated before
        if os.path.exists(file_name):
            return sp.load_npz(file_name)

        # Get the initial parameters of the current QuantumCircuit object
        init_type = self._init_parameters['init_type']
        num_qubits = self._init_parameters['num_qubits']

        qc_noiseless = QuantumCircuit(num_qubits, init_type)

        for i, user_operation in enumerate(self._user_operation_order):
            operation = list(user_operation.keys())[0]
            parameters = list(user_operation.values())[0]

            if operation == "create_bell_pairs_top":
                qc_noiseless.create_bell_pairs_top(parameters[0], parameters[1])
            elif operation == "apply_1_qubit_gate":
                qc_noiseless.apply_1_qubit_gate(parameters[0], parameters[1])
            elif operation == "apply_2_qubit_gate":
                qc_noiseless.apply_2_qubit_gate(parameters[0], parameters[1], parameters[2])
            elif operation == "add_top_qubit":
                qc_noiseless.add_top_qubit(parameters[0])
            elif operation == "measure_first_N_qubits":
                uneven_parity = True if measure_error and i == (len(self._user_operation_order) - 1) else False
                qc_noiseless.measure_first_N_qubits(parameters[0], parameters[1], uneven_parity)

        qc_noiseless.draw_circuit()

        if save:
            sp.save_npz(file_name, qc_noiseless.density_matrix)

        return qc_noiseless.density_matrix

    def _file_name_from_circuit(self, measure_error):
        # Create an hash id, based on the operation and there order on the system and use this for the filename
        system_id = "".join(["{}{}".format(list(d.keys())[0], list(d.values())[0]) for d in self._user_operation_order])
        hash_id = hashlib.sha1(system_id.encode("UTF-8")).hexdigest()[:10]
        file_name = "density_matrix{}_{}.npz".format(("_me" if measure_error else ""), hash_id)
        return file_name

    def save_density_matrix(self, measure_error):
        file_name = self._file_name_from_circuit(measure_error)

        sp.save_npz(file_name, self.density_matrix)

    def _all_single_qubit_gate_possibilities(self, qubits):
        """
            Method returns a list containing all the possible combinations of Pauli matrix gates
            that can be applied to the specified qubits.

            Parameters
            ----------
            qubits : list
                A list of the qubit indices for which all the possible combinations of Pauli matrix gates
                should be returned.

            Returns
            -------
            all_gate_combinations : list
                list of all the qubit gate arrangements that are possible for the specified qubits.

            Examples
            --------
            self._all_single_qubit_gate_possibilities([0, 1]), then the method will return

            [[X, X], [X, Y], [X, Z], [X, I], [Y, X], [Y, Y], [Y, Z] ....]

            in which, in general, A -> {"A": single_qubit_A_gate_object} where A in {X, Y, Z, I}.
        """
        operations = [X, Y, Z, I]
        gate_combinations = []

        for qubit in qubits:
            gates = []
            for operation in operations:
                gates.append({gate_name(operation): self._create_1_qubit_gate(operation, qubit)})
            gate_combinations.append(gates)

        return list(product(*gate_combinations))

    @staticmethod
    def _fuse_equal_config_up_to_permutation(superoperator, proj_type="Z"):
        checked = []
        sorted_superoperator = []
        count = None
        new_value = None
        old_value = None

        # Check for same configurations up to permutations by comparing the sorted error_arrays of each
        # SuperOperatorElement and the lie attribute.
        for supop_el_a, supop_el_b in permutations(superoperator, 2):
            if supop_el_b.id in checked or supop_el_a.id in checked: continue
            if supop_el_a != old_value:
                if old_value is not None:
                    if old_value.error_array.count("I") == old_value.error_array.count(proj_type):
                        new_value = new_value/2
                    sorted_superoperator.append(SuperoperatorElement(new_value, old_value.lie, old_value.error_array))
                count = 1
                new_value = supop_el_a.p
            if supop_el_a.error_array_lie_equals(supop_el_b):
                count += 1
                new_value += supop_el_b.p
                checked.append(supop_el_b.id)
            old_value = supop_el_a

        sorted_superoperator.append(SuperoperatorElement(new_value, old_value.lie, old_value.error_array))

        return sorted_superoperator

    @staticmethod
    def _remove_not_likely_configurations(superoperator):
        for supop_el_a, supop_el_b in combinations(superoperator, 2):
            if supop_el_a.probability_lie_equals(supop_el_b):
                if supop_el_a.error_array.count("I") > supop_el_b.error_array.count("I") \
                        and supop_el_b in superoperator:
                    superoperator.remove(supop_el_b)
                elif supop_el_a.error_array.count("I") < supop_el_b.error_array.count("I") \
                        and supop_el_a in superoperator:
                    superoperator.remove(supop_el_a)

        return superoperator

    @staticmethod
    def _print_superoperator(superoperator):
        print("\n---- Superoperator ----\n")

        total = sum([supop_el.p for supop_el in superoperator])
        for supop_el in sorted(superoperator):
            probability = supop_el.p
            print("\nProbability: {}".format(probability))
            config = ""
            for gate in supop_el.error_array:
                if gate == "X":
                    config += (colored(gate, 'red') + " ")
                elif gate == "Z":
                    config += (colored(gate, 'cyan') + " ")
                elif gate == "Y":
                    config += (colored(gate, 'magenta') + " ")
                elif gate == "I":
                    config += (colored(gate, 'yellow') + " ")
                else:
                    config += (gate + " ")
            me = "me" if supop_el.lie else "no me"
            print("{} - {}".format(config, me))
        print("\nSum of the probabilities is: {}\n".format(total))
        print("\n---- End of Superoperator ----\n")

    def get_kraus_operator(self):
        """
            Returns the effective operator per qubit. Works only for a system that is initially in the maximally
            entangled state (data qubits in a perfect Bell state with their corresponding ancilla qubit). This is
            because it is based on the Choi-Jamiolkowski Isomorphism.

            *** METHOD ONLY WORKS PROPERLY WHEN ONLY ONE QUBIT OBTAINED AN EFFECTIVE PHASE. FOR MORE INFORMATION
            ON WHY, SEE THE 'DECOMPOSE_NON_ZERO_EIGENVECTORS' METHOD ***

            Returns
            -------
            probabilities_operators : zip
                Zip containing the probabilities with the corresponding operators on the qubits
        """
        probabilities, decomposed_statevector = self.decompose_non_zero_eigenvectors()
        kraus_ops = []

        for eigenvector_states in decomposed_statevector:
            # Initialise a list that will be used to save the total operation matrix per qubit
            kraus_op_per_qubit = int(self.num_qubits / 2) * [None]
            correction = 1 / np.sqrt(2 ** int(self.num_qubits / 2))

            for eigenvector_states_split in eigenvector_states:
                # For each eigenvector iterate over the one qubit state elements of the data qubits to create the
                # effective Kraus operators that happened on the specific data qubit
                for qubit, data_qubit_position in enumerate(range(0, len(eigenvector_states_split), 2)):
                    if kraus_op_per_qubit[qubit] is None:
                        kraus_op_per_qubit[qubit] = correction * CT(eigenvector_states_split[data_qubit_position],
                                                                    eigenvector_states_split[data_qubit_position + 1])
                        continue

                    kraus_op_per_qubit[qubit] += correction * CT(eigenvector_states_split[data_qubit_position],
                                                                 eigenvector_states_split[data_qubit_position + 1])

                kraus_ops.append(kraus_op_per_qubit)

        return zip(probabilities, kraus_ops)

    def print_kraus_operators(self):
        """ Prints a clear overview of the effective operations that have happened on the individual qubits """
        print_lines = ["\n---- Kraus operators per qubit ----\n"]
        for prob, operators_per_qubit in sorted(self.get_kraus_operator()):
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
        ----------------------------------------------------------------------------------------------------------
                                            Circuit drawing Methods
        ----------------------------------------------------------------------------------------------------------     
    """

    def draw_circuit(self):
        """ Draws the circuit that corresponds to the operation that have been applied on the system,
        up until the moment of calling. """
        legenda = "--- Circuit ---\n\n @: noisy Bell-pair, #: perfect Bell-pair, o: control qubit " \
                  "(with target qubit at same level), [X,Y,Z,H]: gates, M: measurement, " + colored("~", 'red') + \
                  ": noisy operation (gate/measurement)\n"
        init = self._draw_init()
        self._draw_gates(init)
        init[-1] += "\n\n"
        print(legenda, *init)

    def _draw_init(self):
        """ Returns an array containing the visual representation of the initial state of the qubits. """
        init_state_repr = []
        for i in self._qubit_array:
            init_state_repr.append("\n\n{} ---".format(state_repr(i)))
        return init_state_repr

    def _draw_gates(self, init):
        """ Adds the visual representation of the operations applied on the qubits """
        for gate_item in self._draw_order:
            gate = next(iter(gate_item))
            value = gate_item[gate]
            if type(value) == tuple:
                control = "o" if "~" not in gate else colored("~", 'red') + "o"
                if gate == "#" or gate == "@":
                    control = gate
                cqubit = value[0]
                tqubit = value[1]
                init[cqubit] += "---{}---".format(control)
                init[tqubit] += "---{}---".format(gate)
            else:
                init[value] += "---{}---".format(gate)

            for a, b in it.combinations(enumerate(init), 2):
                # Since colored ansi code is shown as color and not text it should be stripped for length comparison
                ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
                a_stripped = ansi_escape.sub("", init[a[0]])
                b_stripped = ansi_escape.sub("", init[b[0]])

                if (diff := len(b_stripped) - len(a_stripped)) > 0:
                    init[a[0]] += diff * "-"
                elif (diff := len(a_stripped) - len(b_stripped)) > 0:
                    init[b[0]] += diff * "-"

    def _add_draw_operation(self, operation, qubits):
        """ Add an operation to the draw order list """
        if self._effective_measurements != 0:
            if type(qubits) is tuple:
                cqubit = qubits[0] + self._effective_measurements
                tqubit = qubits[1] + self._effective_measurements

                if self._measured_qubits != [] and cqubit >= min(self._measured_qubits):
                    cqubit += len(self._measured_qubits)
                if self._measured_qubits != [] and tqubit >= min(self._measured_qubits):
                    tqubit += len(self._measured_qubits)

                qubits = (cqubit, tqubit)
            else:
                qubits += int(self._effective_measurements)

                if self._measured_qubits != [] and qubits >= min(self._measured_qubits):
                    qubits += len(self._measured_qubits)
        item = {operation: qubits}
        self._draw_order.append(item)

    def _correct_for_n_top_qubit_additions(self, n=1):
        """ Corrects the draw order list for addition of n top qubits """
        self._measured_qubits.extend([i for i in range(self._effective_measurements)])
        self._measured_qubits = [(x + n) for x in self._measured_qubits]
        self._effective_measurements = 0
        for i, draw_item in enumerate(self._draw_order):
            operation = list(draw_item.keys())[0]
            qubits = list(draw_item.values())[0]
            if type(qubits) == tuple:
                self._draw_order[i] = {operation: (qubits[0] + n, qubits[1] + n)}
            else:
                self._draw_order[i] = {operation: qubits + n}

    def __repr__(self):
        density_matrix = self.density_matrix.toarray() if self.num_qubits < 4 else self.density_matrix
        return "\nCircuit density matrix:\n\n{}\n\n".format(density_matrix)
