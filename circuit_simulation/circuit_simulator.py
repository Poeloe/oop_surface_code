import os
import sys
sys.path.insert(1, os.path.abspath(os.getcwd()))
from circuit_simulation.basic_operations import (
    CT, KP, state_repr, get_value_by_prob, trace, gate_name, fidelity_elementwise
)
from circuit_simulation.states_and_gates import *
import numpy as np
from scipy import sparse as sp
import itertools as it
import copy
from scipy.linalg import eig, eigh
import hashlib
import re
from oopsc.superoperator.superoperator import SuperoperatorElement
from termcolor import colored
from itertools import combinations, permutations, product
from circuit_simulation.latex_circuit.qasm_to_pdf import create_pdf_from_qasm
import pandas as pd
from fractions import Fraction as Fr
import math
import random
from operator import itemgetter


# Uncomment this if a segmentation error when diagonalising the density matrix for a circuit with a large amount of
# qubits occurs:
# ket_0 = np.array([[1, 0]]).T
# ket_1 = np.array([[0, 1]]).T
# ket_p = 1 / np.sqrt(2) * (ket_0 + ket_1)
# ket_m = 1 / np.sqrt(2) * (ket_0 - ket_1)
#
# X = np.array([[0, 1], [1, 0]])
# Y = np.array([[0, -1j], [1j, 0]])
# Z = np.array([[1, 0], [0, -1]])
# I = np.array([[1, 0], [0, 1]])
# H = 1 / np.sqrt(2) * np.array([[1, 1], [1, -1]])
# S = np.array([[1, 0], [0, 1j]])


class QuantumCircuit:
    """
        QuantumCircuit(num_qubits, init_type=0, noise=False, pg=0.01, pm=0.01)

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
                        rest of the qubits is in the |0> state. On every qubit a CNOT gate is
                        applied with the first qubit being the control qubit.

            noise : bool, optional, default=False
                Will apply noise on every operation that is applied to the QuantumCircuit object,
                unless specified otherwise.
            pg : float [0-1], optional, default=0.01
                The overall amount of gate noise that will be applied when 'noise' is set to True.
            pm : float [0-1], optional, default=0.01
                The overall amount of measurement error that will be applied when 'noise' set to
                True.
            pn : float [0-1], optional, default=None
                The overall amount of network noise that will be applied when 'noise is set to True.
            p_dec : float [0-1], optional, default=0
                The overall amount of decoherence in the system. This is only applied when noise is True and
                the value is greater than 0.
            p_bell_success : float [0-1], optional, default=1
                Specifies the success rate of the creation of Bell pairs. Default value is 1, which equals the case
                that a Bell pair creation always instantly succeeds.
            basis_transformation_noise : bool, optional, default = None
                Set to true if the transformation from the computational basis to the X-basis for a
                measurement should be noisy.
            probabilistic : bool, optional, default=False
                In case measurements should be probabilistic of nature, this can be set to True. Measurement
                outcomes will then be determined based on their probabilities if not differently specified
            measurement_duration : float, optional, default=4
                In case of decoherence, the measurement duration is used to determine the amount of decoherence that
                should be applied for a measurement operation
            bell_creation_duration : float, optional, default=4
                In case of decoherence, the bell creation duration is used to determine the amount of decoherence that
                should be applied for a measurement operation
            network_noise_type : int, optional, default=0
                The type of network noise that should be used. At this point in time, two variants are
                available:

                0 ->    NV centre specific noise for the creation of a Bell pair
                1 ->    Noise specified by Naomi Nickerson in her master thesis
            no_single_qubit_error : bool, optional, default=False
                When single qubit gates are free of noise, but noise in general is present, this boolean
                is set to True. It prevents the addition of noise when applying a single qubit gate
            thread_safe_printing : bool, optional, deafult=False
                If working with threas, this can be set to True. This prevents print statements from being
                printed in real-time. Instead the lines will be saved and can at all time be printed all in once
                when running the 'print' method. Print lines are always saved in the _print_lines array until printing


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
            basis_transformation_noise : bool, optional, default=False
                Whether the H-gate that is applied to transform the basis in which the qubit is measured should be
                noisy (True) or noiseless (False) in general. If not specified, it will have the same value as the
                'noise' attribute.
            pg : float [0-1], optional, default=0.01
                The amount of gate noise present in the system. Will only be applied if 'noise' is True.
            pm : float [0-1], optional, default=0.01
                The amount of measurement noise present in the system. Will only be applied if 'noise' is True.
            _qubit_density_matrix_lookup : dict
                The density matrix of the entire system is split into separate density matrices where ever possible
                (density matrices will be fused when two-qubit gate is applied). This dictionary is used to lookup
                to which density matrix a qubit belongs
            _qubit_array : ndarray
                A list containing the initial state of the qubits.
            _draw_order : list of dict items
                A list containing dict items that specify the operations that should be drawn.
            _user_operation_order : list
                List containing the actions on the circuit applied by the user.
            _effective_measurements : int, default=0
                Integer keeping track of the amount of effectively measured qubits. Used for more clear circuit
                drawings.
            _measured_qubits : list
                List containing the indices of the qubits that have been measured and are therefore not used after.
                Used for more clear circuit drawings.
            _init_parameters : dict
                A dictionary containing the initial parameters of the system, including the '_qubit_array' and
                'density_matrix' attribute. The keys are the names of the attributes.

    """

    def __init__(self, num_qubits, init_type=0, noise=False, basis_transformation_noise=None, pg=0.001, pm=0.001,
                 pn=None, p_dec=0, p_bell_success=1, time_step=1, measurement_duration=4, bell_creation_duration=4,
                 probabilistic=False, network_noise_type=0, no_single_qubit_error=False, thread_safe_printing=False):
        self.num_qubits = num_qubits
        self.d = 2 ** num_qubits
        self.noise = noise
        self.pg = pg
        self.pm = pm
        self.pn = pn
        self.p_dec = p_dec
        self.p_bell_success = p_bell_success
        self.bell_creation_duration = bell_creation_duration
        self.network_noise_type = network_noise_type
        self.time_step = time_step
        self.measurement_duration = measurement_duration
        self.probabilistic = probabilistic
        self.no_single_qubit_error = no_single_qubit_error
        self.total_duration = 0
        self._init_type = init_type
        self._qubit_array = num_qubits * [ket_0]
        self._draw_order = []
        self._user_operation_order = []
        self._effective_measurements = 0
        self._measured_qubits = []
        self.density_matrices = None
        self._qubit_density_matrix_lookup = {}
        self._print_lines = []
        self._thread_safe_printing=thread_safe_printing
        self._fused = False

        self.basis_transformation_noise = noise if basis_transformation_noise is None else basis_transformation_noise

        if init_type == 0:
            self.density_matrices = self._init_density_matrix()
        elif init_type == 1:
            self.density_matrices = self._init_density_matrix_first_qubit_ket_p()
        elif init_type == 2:
            self.density_matrices = self._init_density_matrix_bell_pair_state()
        elif init_type == 3:
            self.density_matrices = self._init_density_matrix_ket_p_and_CNOTS()

        self._init_parameters = self._init_parameters_to_dict()

    """
        ---------------------------------------------------------------------------------------------------------
                                                    Init Methods
        ---------------------------------------------------------------------------------------------------------     
    """

    def _init_density_matrix(self):
        """ Realises init_type option 0. See class description for more info. """

        density_matrices = []
        for i, qubit in enumerate(self._qubit_array):
            density_matrix = CT(qubit, qubit)
            density_matrices.append(density_matrix)
            self._qubit_density_matrix_lookup[i] = (density_matrix, [i])
        return density_matrices

    def _init_density_matrix_first_qubit_ket_p(self):
        """ Realises init_type option 1. See class description for more info. """

        self._qubit_array[0] = ket_p

        density_matrices = []
        for i, qubit in enumerate(self._qubit_array):
            density_matrix = CT(qubit, qubit)
            density_matrices.append(density_matrix)
            self._qubit_density_matrix_lookup[i] = (density_matrix, [i])

        return density_matrices

    def _init_density_matrix_bell_pair_state(self, amount_qubits=8, draw=True):
        """ Realises init_type option 2. See class description for more info. """

        density_matrices = []
        bell_pair_rho = sp.lil_matrix((4, 4))
        bell_pair_rho[0, 0], bell_pair_rho[3, 0], bell_pair_rho[0, 3], bell_pair_rho[3, 3] = 1 / 2, 1 / 2, 1 / 2, 1 / 2

        for i in range(0, self.num_qubits-amount_qubits):
            state = self._qubit_array[i]
            self._qubit_density_matrix_lookup[i] = (CT(state), [i])

        for i in range(self.num_qubits-amount_qubits, self.num_qubits, 2):
            density_matrix = sp.csr_matrix(bell_pair_rho)
            qubits = [i, i+1]
            if draw:
                self._add_draw_operation("#", (i, i + 1))
            self._qubit_density_matrix_lookup.update({i: (density_matrix, qubits), i+1: (density_matrix, qubits)})
            density_matrices.append(density_matrix)
        return density_matrices

    def _init_density_matrix_ket_p_and_CNOTS(self):
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

    def _init_parameters_to_dict(self):
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
    """
        ---------------------------------------------------------------------------------------------------------
                                                Separated Density Matrices Methods
        ---------------------------------------------------------------------------------------------------------
    """
    def _correct_lookup_for_addition(self, amount_qubits=1, position='top'):
        """
            Method corrects the qubit_density_matrix_lookup dictionary for the addition of a top or bottom qubit.

            Parameters
            ----------
            amount_qubits : int
                Amount of qubits that is added to the top (or bottom) of the system.
            position : str['top', 'bottom'], optional, default='top'
                String value that indicates if the qubit is added to the top or the bottom of the system
        """
        if position.lower() == 'top':
            position = 0
        elif position.lower() == 'bottom':
            position = -1
        else:
            raise ValueError("position argument can only be 'top' or 'bottom'.")

        new_lookup_dict = {}
        for qubit, (density_matrix, qubits) in sorted(self._qubit_density_matrix_lookup.items()):
            new_lookup_dict[qubit+amount_qubits] = (density_matrix, [q + amount_qubits for q in qubits])
        self._qubit_density_matrix_lookup = new_lookup_dict

        qubit_indices = [i for i in range(amount_qubits)]
        for qubit_num in range(amount_qubits):
            self._qubit_density_matrix_lookup[qubit_num] = (self.density_matrices[position], qubit_indices)

    def _correct_lookup_for_two_qubit_gate(self, cqubit, tqubit):
        """
            Method corrects the qubit_density_matrix_lookup dictionary when a two-qubit gate is applied.
            Due to two-qubit gates, the density matrices of the involved qubits should be fused (if not already).

            Parameters
            ----------
            cqubit : int
                Qubit number of the control qubit
            tqubit : int
                Qubit number of the control qubit
        """
        cqubit_density_matrix, c_qubits = self._qubit_density_matrix_lookup[cqubit]
        tqubit_density_matrix, t_qubits = self._qubit_density_matrix_lookup[tqubit]
        fused_density_matrix = KP(cqubit_density_matrix, tqubit_density_matrix)
        fused_qubits = c_qubits + t_qubits

        for qubit in fused_qubits:
            self._qubit_density_matrix_lookup[qubit] = (fused_density_matrix, fused_qubits)

    def _get_qubit_relative_objects(self, qubit):
        """
            Method returns for the given qubit the following relative objects:
             - relative density matrix,
             - qubits order that is present in the density matrix,
             - the qubit index for the density matrix
             - the amount of qubits that is present in the density matrix

            Parameters
            ----------
            qubit : int
                Qubit number of the qubit that the relative objects are requested for
        """
        density_matrix, qubits = self._qubit_density_matrix_lookup[qubit]
        relative_qubit_index = qubits.index(qubit)
        relative_num_qubits = len(qubits)

        return density_matrix, qubits, relative_qubit_index, relative_num_qubits

    def _correct_lookup_for_measurement_top(self):
        """
            Method corrects the qubit_density_matrix_lookup dictionary for the (destructive) measurement of the top
            qubit

            **NOTE: Qubits involved in the same density matrix should all point to the same density matrix object
            in memory and the same involved qubits list object in memory. This is why the qubits list is adapted in the
            qubits[:] way, this ensures that the same memory address is used.**
        """
        new_lookup_dict = {}
        _, qubits_old = self._qubit_density_matrix_lookup[0]
        del qubits_old[-1]
        for qubit, (density_matrix, qubits) in sorted(self._qubit_density_matrix_lookup.items()):
            if qubit == 0:
                qubits_old = qubits
                continue
            if qubits_old is qubits:
                qubits = qubits_old
            else:
                qubits[:] = [i - 1 for i in qubits]
                qubits_old = qubits

            new_lookup_dict[qubit - 1] = density_matrix, qubits

        self._qubit_density_matrix_lookup = new_lookup_dict

    def _correct_lookup_for_measurement_any(self, qubit, qubits, density_matrix_measured, new_density_matrix):
        self._qubit_density_matrix_lookup[qubit] = (density_matrix_measured, [qubit])
        qubits.remove(qubit)
        for q in qubits:
            self._qubit_density_matrix_lookup[q] = (new_density_matrix, qubits)

    def _correct_lookup_for_circuit_fusion(self, lookup_other):
        num_qubits_other = len(lookup_other)
        new_lookup = lookup_other
        prev_qubits = None
        for qubit, (density_matrix, qubits) in sorted(self._qubit_density_matrix_lookup.items()):
            if prev_qubits is not qubits:
                qubits[:] = [i + num_qubits_other for i in qubits]
                prev_qubits = qubits
            new_lookup[qubit + num_qubits_other] = (density_matrix, qubits)
        self._qubit_density_matrix_lookup = new_lookup

    def _set_density_matrix(self, qubit, new_density_matrix):
        """
            Method sets the density matrix for the given qubit and all qubits that are involved in the same density
            matrix

            *** NOTE: density matrices have to be set with this method in order to guarantee proper functioning of the
            program. It ensures that qubits involved in the same density matrix will point to the same density matrix
            object in memory (such that when the matrix changes, it changes for each involved qubit) ***

            Parameters
            ----------
            qubit : int
                Qubit number for which the density matrix should be set
            new_density_matrix : csr_matrix
                The new density matrix that should be set
        """
        _, qubits, _, _ = self._get_qubit_relative_objects(qubit)
        for qubit in qubits:
            self._qubit_density_matrix_lookup[qubit] = (new_density_matrix, qubits)

    def get_combined_density_matrix(self, qubits):
        density_matrices = []
        skip_qubits = []
        for qubit in qubits:
            if qubit not in skip_qubits:
                density_matrix, involved_qubits, _, _ = self._get_qubit_relative_objects(qubit)
                density_matrices.append(density_matrix)
                skip_qubits.extend(involved_qubits)
        return KP(*density_matrices)

    def total_density_matrix(self):
        """
            Get the total density matrix of the system
        """
        density_matrices = []
        skip_qubits = []
        for qubit, (density_matrix, qubits) in sorted(self._qubit_density_matrix_lookup.items()):
            if qubit not in skip_qubits:
                density_matrices.append(density_matrix)
                skip_qubits.extend(qubits)
        return KP(*density_matrices)

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
            _, _, _, rel_num_qubits = self._get_qubit_relative_objects(tqubit)
            if rel_num_qubits > 1 or tqubit >= self.num_qubits:
                raise ValueError("Qubit is not suitable to set state for.")

            self._qubit_array[tqubit] = state
            self._qubit_density_matrix_lookup[tqubit] = (CT(state), [tqubit])

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

    def create_bell_pairs_top(self, N, new_qubit=False, noise=None, pn=None, network_noise_type=None, bell_state_type=1,
                              probabilistic=None, p_bell_success=None, bell_creation_duration=None, user_operation=True):
        """
        qc.create_bell_pair(N, pn=0.1)

            This appends noisy Bell pairs on the top of the system. The noise is based on network noise
            modeled as (paper: https://www.nature.com/articles/ncomms2773.pdf)

                rho_raw = (1 - 4/3*pn) |psi><psi| + pn/3 * I,

            in which |psi> is a perfect Bell state.

            *** THIS METHOD APPENDS THE QUBITS TO THE TOP OF THE SYSTEM. THIS MEANS THAT THE AMOUNT OF
            QUBITS IN THE SYSTEM WILL GROW WITH '2N' AND THE INDICES OF THE EXISTING QUBITS INCREASE WITH 2N AS WELL,
            WHICH IS IMPORTANT FOR FUTURE OPERATIONS ***

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
            network_noise_type : int, optional, default=None
                Type of network noise that should be used. If not specified, the network noise type known for the
                QuantumCircuit object is used
            bell_state_type : int [1-4], optional, default=1
                Choose the Bell state type which should be created, types are:
                    1 : |00> + |11>
                    2 : |00> - |11>
                    3 : |01> + |10>
                    4 : |01> - |10>
            probabilistic : bool, optional, default=None
                In case of a probabilistic, the method will keep trying to create the bell state untill success. When
                decoherence is present, this adds decoherence after each try. If not specified, the value kwnown for
                the QuantumCircuit object is used
            p_bell_success : float [0-1], optional, default=None
                The success rate of the bell state creation when probabilistic. If not specified, the value known for
                the QuantumCircuit object is used.
            bell_creation_duration : float, optional, defualt=None,
                The duration of a Bell pair creation relative to the time-step. If not specified, the value known for
                the QuantumCircuit object is used.

            Example
            -------
            qc.create_bell_pairs([(0, 1), (2, 3), (4,5)]) --> Creates Bell pairs between qubit 0 and 1,
            between qubit 2 and 3 and between qubit 4 and 5.
        """
        if user_operation:
            self._user_operation_order.append({"create_bell_pairs_top": [N, new_qubit, noise, pn]})
        if noise is None:
            noise = self.noise
        if pn is None:
            pn = self.pn
        if network_noise_type is None:
            network_noise_type = self.network_noise_type
        if probabilistic is None:
            probabilistic = self.probabilistic
        if p_bell_success is None:
            p_bell_success = self.p_bell_success
        if bell_creation_duration is None:
            bell_creation_duration = self.bell_creation_duration

        for i in range(0, 2 * N, 2):
            times = 1
            while probabilistic and random.random() > p_bell_success:
                times += 1

            # print("\nBell Pair creation took {} time{}".format(times, "s" if times > 1 else ""))

            self.num_qubits += 2
            self.d = 2 ** self.num_qubits
            density_matrix = self._get_bell_state_by_type(bell_state_type)

            if noise:
                density_matrix = self._N_network(density_matrix, pn, network_noise_type)

            self.density_matrices.insert(0, density_matrix)
            self._correct_lookup_for_addition(amount_qubits=2)

            # Drawing the Bell Pair
            if new_qubit:
                self._qubit_array.insert(0, ket_0)
                self._qubit_array.insert(0, ket_0)
                self._correct_drawing_for_n_top_qubit_additions(n=2)
            else:
                self._effective_measurements -= 2
            self._add_draw_operation("#", (0, 1), noise)

            if noise and self.p_dec > 0:
                times_total = times * int(math.ceil(bell_creation_duration / self.time_step))
                self._N_decoherence([i, i + 1], times=times_total)

    def create_bell_pair(self, qubit1, qubit2, noise=None, pn=None, network_noise_type=None, bell_state_type=1, probabilistic=None,
                         p_bell_success=None, bell_creation_duration=None, user_operation=True):
        if user_operation:
            self._user_operation_order.append({"create_bell_pair": [qubit1, qubit2, noise, pn]})
        if noise is None:
            noise = self.noise
        if pn is None:
            pn = self.pn
        if network_noise_type is None:
            network_noise_type = self.network_noise_type
        if probabilistic is None:
            probabilistic = self.probabilistic
        if p_bell_success is None:
            p_bell_success = self.p_bell_success
        if bell_creation_duration is None:
            bell_creation_duration = self.bell_creation_duration

        times = 1
        while probabilistic and random.random() > p_bell_success:
            times += 1

        _, _, _, num_qubits_1 = self._get_qubit_relative_objects(qubit1)
        _, _, _, num_qubits_2 = self._get_qubit_relative_objects(qubit2)

        if num_qubits_1 > 1 or num_qubits_2 > 1:
            raise ValueError("Qubits are not suitable to create a Bell pair this way.")

        new_density_matrix = self._get_bell_state_by_type(bell_state_type)

        if noise:
            new_density_matrix = self._N_network(new_density_matrix, pn, network_noise_type)

        self._qubit_density_matrix_lookup.update({qubit1: (new_density_matrix, [qubit2, qubit1]),
                                                  qubit2: (new_density_matrix, [qubit2, qubit1])})

        self._add_draw_operation("#", (qubit1, qubit2), noise)

        if noise and self.p_dec > 0:
            times_total = times * int(math.ceil(bell_creation_duration / self.time_step))
            self._N_decoherence([qubit1, qubit2], times=times_total)

    @staticmethod
    def _get_bell_state_by_type(bell_state_type=1):
        """
            Returns a Bell state density matrix based on the type provided. types are:
                    1 : |00> + |11>
                    2 : |00> - |11>
                    3 : |01> + |10>
                    4 : |01> - |10>
        """
        rho = sp.lil_matrix((4, 4))
        if bell_state_type == 1:
            rho[0, 0], rho[0, 3], rho[3, 0], rho[3, 3] = 1 / 2, 1 / 2, 1 / 2, 1 / 2
        elif bell_state_type == 2:
            rho[0, 0], rho[0, 3], rho[3, 0], rho[3, 3] = 1 / 2, -1 / 2, -1 / 2, 1 / 2
        elif bell_state_type == 3:
            rho[1, 1], rho[1, 2], rho[2, 1], rho[2, 2] = 1 / 2, -1 / 2, -1 / 2, 1 / 2
        elif bell_state_type == 4:
            rho[1, 1], rho[1, 2], rho[2, 1], rho[2, 2] = 1 / 2, 1 / 2, 1 / 2, 1 / 2
        else:
            raise ValueError("A non-valid Bell state type was requested. Known types are 1, 2, 3, and 4.")
        return rho

    def add_top_qubit(self, qubit_state=ket_0, p_prep=0, user_operation=True):
        """
        qc.add_top_qubit(qubit_state=ket_0)

            Method appends a qubit with a given state to the top of the system.
            *** THE METHOD APPENDS A QUBIT, WHICH MEANS THAT THE AMOUNT OF QUBITS IN THE SYSTEM WILL
            GROW WITH 1 AND THE INDICES OF THE EXISTING QUBITS WILL INCREASE WITH 1 AS WELL***

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
        if self.noise and p_prep > 0:
            qubit_state = self._N_preparation(state=qubit_state, p_prep=p_prep)

        self._qubit_array.insert(0, qubit_state)
        self.num_qubits += 1
        self.d = 2 ** self.num_qubits
        self._correct_drawing_for_n_top_qubit_additions()

        self.density_matrices.insert(0, CT(qubit_state))
        self._correct_lookup_for_addition()

    """
        ---------------------------------------------------------------------------------------------------------
                                                One-Qubit Gate Methods
        ---------------------------------------------------------------------------------------------------------     
    """

    def apply_1_qubit_gate(self, gate, tqubit, conj=False, noise=None, pg=None, draw=True, user_operation=True):
        """
            qc.apply_1_qubit_gate(gate, tqubit, noise=None, pg=None, draw=True)

                Applies a single-qubit gate to the specified target qubit. This will update the density
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

        tqubit_density_matrix, _, relative_tqubit_index, relative_num_qubits = self._get_qubit_relative_objects(tqubit)

        one_qubit_gate = self._create_1_qubit_gate(gate.matrix if not conj else gate.dagger,
                                                   relative_tqubit_index,
                                                   relative_num_qubits)
        new_density_matrix = sp.csr_matrix(one_qubit_gate.dot(CT(tqubit_density_matrix, one_qubit_gate)))

        if noise and not self.no_single_qubit_error:
            new_density_matrix = self._N_single(pg, relative_tqubit_index, new_density_matrix, relative_num_qubits)

        if noise and self.p_dec != 0:
            new_density_matrix = self._N_decoherence([tqubit], gate)

        self._set_density_matrix(tqubit, new_density_matrix)

        if draw:
            self._add_draw_operation(gate, tqubit, noise)

    def _create_1_qubit_gate(self, gate, tqubit, num_qubits=None):
        """
            Private method that is used to create the single-qubit gate matrix used in for example the
            apply_1_qubit_gate method.

            Parameters
            ----------
            gate : ndarray
                Array of dimension 2x2, examples are the well-known pauli matrices (X, Y, Z)
            tqubit : int
                Integer that indicates the target qubit. Note that the qubit counting starts at
                0.
            num_qubits : int, optional, default=None
                Determines the size of the resulting one-qubit gate matrix. If not specified, the
                num_qubits known for the entire QuantumCircuit object is used

            Returns
            -------
            1_qubit_gate : sparse matrix with dimensions equal to the density_matirx attribute
                Returns a matrix with dimensions equal to the dimensions of the density matrix of
                the system.
        """
        if num_qubits is None:
            num_qubits = self.num_qubits
        if type(gate) == SingleQubitGate:
            gate = gate.matrix

        if num_qubits == 1:
            return sp.csr_matrix(gate)
        if np.array_equal(gate, I_gate.matrix):
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
            num_qubits : int, optional, default=None
                Amount of qubits that is present in the specific density matrix that the identity operations
                are requested for. If not specified, the amount of qubits of the QuantumCircuit object is used


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

        for _ in range(times):
            self.apply_1_qubit_gate(X_gate, tqubit, noise=noise, pg=pg, draw=draw, user_operation=user_operation)

    def Z(self, tqubit, times=1, noise=None, pg=None, draw=True, user_operation=True):
        """ Applies the pauli Z gate to the specified target qubit. See apply_1_qubit_gate for more info """

        for _ in range(times):
            self.apply_1_qubit_gate(Z_gate, tqubit, noise=noise, pg=pg, draw=draw, user_operation=user_operation)

    def Y(self, tqubit, times=1, noise=None, pg=None, draw=True, user_operation=True):
        """ Applies the pauli Y gate to the specified target qubit. See apply_1_qubit_gate for more info """

        for _ in range(times):
            self.apply_1_qubit_gate(Y_gate, tqubit, noise=noise, pg=pg, draw=draw, user_operation=user_operation)

    def H(self, tqubit, times=1, noise=None, pg=None, draw=True, user_operation=True):
        """ Applies the Hadamard gate to the specified target qubit. See apply_1_qubit_gate for more info """

        for _ in range(times):
            self.apply_1_qubit_gate(H_gate, tqubit, noise=noise, pg=pg, draw=draw, user_operation=user_operation)

    def S(self, tqubit, conj=False, times=1, noise=None, pg=None, draw=True, user_operation=True):

        for _ in range(times):
            self.apply_1_qubit_gate(S_gate, tqubit, conj=conj, noise=noise, pg=pg, draw=draw,
                                    user_operation=user_operation)

    def Rx(self, tqubit, theta, times=1, noise=None, pg=None, draw=True, user_operation=True):
        """ Applies a rotation gate around the x-axis to the specified target qubit with the specified angle.

            Parameters
            ----------
            theta : float (radians)
                Angle of rotation that should be applied. Value should be specified in radians
        """
        R_gate = SingleQubitGate("Rotation gate",
                      np.array([[np.cos(theta/2), -1j * np.sin(theta/2)],
                                [-1j * np.sin(theta/2), np.cos(theta/2)]]),
                      "Rx({})".format(str(Fr(theta/np.pi)) + "\u03C0"))

        for _ in range(times):
            self.apply_1_qubit_gate(R_gate, tqubit, noise=noise, pg=pg, draw=draw, user_operation=user_operation)

    def Ry(self, tqubit, theta, times=1, noise=None, pg=None, draw=True, user_operation=True):
        """ Applies a rotation gate around the y-axis to the specified target qubit with the specified angle.

            Parameters
            ----------
            theta : float (radians)
                Angle of rotation that should be applied. Value should be specified in radians
        """
        R_gate = SingleQubitGate("Rotation gate",
                      np.array([[np.cos(theta / 2), -1 * np.sin(theta / 2)],
                                [1 * np.sin(theta / 2), np.cos(theta / 2)]]),
                      "Ry({})".format(str(Fr(theta/np.pi)) + "\u03C0"))

        for _ in range(times):
            self.apply_1_qubit_gate(R_gate, tqubit, noise=noise, pg=pg, draw=draw, user_operation=user_operation)

    def Rz(self, tqubit, theta, times=1, noise=None, pg=None, draw=True, user_operation=True):
        """ Applies a rotation gate around the x axis to the specified target qubit with the specified angle.

            Parameters
            ----------
            theta : float (radians)
                Angle of rotation that should be applied. Value should be specified in radians

        """
        R_gate = SingleQubitGate("Rotation gate",
                      np.array([np.exp(-1j * theta / 2), 0],
                               [0, np.exp(1j * theta / 2)]),
                      "Rz({})".format(str(Fr(theta/np.pi)) + "\u03C0"))

        for _ in range(times):
            self.apply_1_qubit_gate(R_gate, tqubit, noise=noise, pg=pg, draw=draw, user_operation=user_operation)

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
            gate : TwoQubitGate class
                Gate class object, predefined Gate objects are available such as the X, Y and Z gates
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
            gate_2 : array, optional, default=None
                Array of dimension 2x2. This parameter can be used to specify a gate that is applied to the
                target qubit for the case that the control qubit is in the |0> state.
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

        cqubit_density_matrix, _ = self._qubit_density_matrix_lookup[cqubit]
        tqubit_density_matrix, _ = self._qubit_density_matrix_lookup[tqubit]

        # Check if cqubit and tqubit belong to the same density matrix. If not they should fuse
        if not cqubit_density_matrix is tqubit_density_matrix:
            self._correct_lookup_for_two_qubit_gate(cqubit, tqubit)

        # Since density matrices are fused if not equal, it is only necessary to get the (new) density matrix from
        # the lookup table by either one of the qubit indices
        density_matrix, qubits, rel_cqubit, rel_num_qubits = self._get_qubit_relative_objects(cqubit)
        rel_tqubit = qubits.index(tqubit)

        two_qubit_gate = self._create_2_qubit_gate(gate,
                                                   rel_cqubit,
                                                   rel_tqubit,
                                                   num_qubits=rel_num_qubits)

        new_density_matrix = sp.csr_matrix(two_qubit_gate.dot(CT(density_matrix, two_qubit_gate)))

        if noise:
            new_density_matrix = self._N(pg, rel_cqubit, rel_tqubit, new_density_matrix, num_qubits=rel_num_qubits)
        if draw:
            self._add_draw_operation(gate, (cqubit, tqubit), noise)

        self._set_density_matrix(cqubit, new_density_matrix)

        if noise and self.p_dec != 0:
            self._N_decoherence([tqubit, cqubit], gate)

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

                1. I#I#I + I#I#I + I#I#I + I#I#I
                2. I#|0><0|#I + I#|1><1|#I + 0#|0><1|#I + 0#|1><0|#I
                3. I#|0><0|#I + X_t#|1><1|#I + 0#|0><1|#I + 0#|1><0|#I

        (In which '#' is the Kronecker Product, and '0' is the zero matrix)
        (https://quantumcomputing.stackexchange.com/questions/4252/
        how-to-derive-the-cnot-matrix-for-a-3-qbit-system-where-the-control-target-qbi and
        https://quantumcomputing.stackexchange.com/questions/9181/swap-gate-on-2-qubits-in-3-entangled-qubit-system)

        The 'create_component_2_qubit_gate' method defined within creates one of the 4 components that is shown in
        step 3 above. Thus 'first_part = create_component_2_qubit_gate(CT(ket_0), zero_state_matrix)' creates the first
        component namely I#|0><0|#I in case of the CNOT mentioned.

        Parameters
        ----------
        gate : TwoQubitGate object
            TwoQubitGate object representing a 2-qubit gate
        cqubit : int
            Integer that indicates the control qubit. Note that the qubit counting starts at 0.
        tqubit : int
            Integer that indicates the target qubit. Note that the qubit counting starts at 0.
        num_qubits : int, optional, default=None
            Determines the size of the resulting two-qubit gate matrix. If not specified, the
            num_qubits known for the entire QuantumCircuit object is used

        """
        if num_qubits is None:
            num_qubits = self.num_qubits
        if cqubit == tqubit:
            raise ValueError("Control qubit cannot be the same as the target qubit!")

        def create_component_2_qubit_gate(control_qubit_matrix, target_qubit_matrix):
            # Initialise the only identity case with on the place of the control qubit the identity replaced
            # with the specified control_qubit_matrix
            control_gate = self._create_1_qubit_gate(control_qubit_matrix, cqubit, num_qubits=num_qubits)

            # Initialise the only identity case with on the place of the target qubit the identity replaced
            # with the specified target_qubit_matrix
            if not np.array_equal(target_qubit_matrix, I_gate):
                target_gate = self._create_1_qubit_gate(target_qubit_matrix, tqubit, num_qubits=num_qubits)

                # Matrix multiply the two cases to obtain the total gate
                return target_gate.dot(control_gate)

            return control_gate

        one_state_matrix = gate.matrix if type(gate) == SingleQubitGate else gate.upper_left_matrix
        zero_state_matrix = I_gate.matrix if type(gate) == SingleQubitGate else gate.lower_right_matrix

        first_part = create_component_2_qubit_gate(CT(ket_0), zero_state_matrix)
        second_part = create_component_2_qubit_gate(CT(ket_1), one_state_matrix)

        if type(gate) == TwoQubitGate and not gate.is_cntrl_gate:
            third_part = create_component_2_qubit_gate(CT(ket_0, ket_1), gate.upper_right_matrix)
            fourth_part = create_component_2_qubit_gate(CT(ket_1, ket_0), gate.lower_left_matrix)

            return sp.csr_matrix(first_part + second_part + third_part + fourth_part)

        return sp.csr_matrix(first_part + second_part)

    def CNOT(self, cqubit, tqubit, noise=None, pg=None, draw=True, user_operation=True):
        """ Applies the CNOT gate to the specified target qubit. See apply_2_qubit_gate for more info """

        self.apply_2_qubit_gate(CNOT_gate, cqubit, tqubit, noise, pg, draw, user_operation=user_operation)

    def CZ(self, cqubit, tqubit, noise=None, pg=None, draw=True, user_operation=True):
        """ Applies the CZ gate to the specified target qubit. See apply_2_qubit_gate for more info """

        self.apply_2_qubit_gate(CZ_gate, cqubit, tqubit, noise, pg, draw, user_operation=user_operation)

    def SWAP(self, cqubit, tqubit, noise=None, pg=None, draw=True, user_operation=True):

        self.apply_2_qubit_gate(SWAP_gate, cqubit, tqubit, noise, pg, draw, user_operation=user_operation)

    def two_qubit_gate_NV(self, cqubit, tqubit, noise=None, pg=None, draw=True, user_operation=True):
        """ Applies the two-qubit gate that is specific to the actual NV center"""

        self.apply_2_qubit_gate(NV_two_qubit_gate, cqubit, tqubit, noise, pg, draw, user_operation=user_operation)

    def CNOT_NV(self, cqubit, tqubit, noise=None, pg=None, draw=True, user_operation=True):

        self.Z(cqubit, noise=noise, pg=pg, draw=draw, user_operation=user_operation)
        self.S(cqubit, noise=noise, pg=pg, draw=draw, user_operation=user_operation)
        self.Ry(tqubit, np.pi/2, noise=noise, pg=pg, draw=draw, user_operation=user_operation)
        self.S(tqubit, noise=noise, pg=pg, draw=draw, user_operation=user_operation)
        self.two_qubit_gate_NV(cqubit, tqubit, noise=noise, pg=pg, draw=draw, user_operation=user_operation)
        self.S(tqubit, conj=True, noise=noise, pg=pg, draw=draw, user_operation=user_operation)

    """
        ---------------------------------------------------------------------------------------------------------
                                            Protocol gate sequences
        ---------------------------------------------------------------------------------------------------------  
    """

    def single_selection(self, operation, bell_qubit_1, bell_qubit_2, measure=True, noise=None, pn=None, pm=None,
                         pg=None, user_operation=True):
        """ Single selection as specified by Naomi Nickerson in https://www.nature.com/articles/ncomms2773.pdf """
        success = False
        while not success:
            self.create_bell_pair(bell_qubit_1, bell_qubit_2, noise=noise, pn=pn, user_operation=user_operation)
            self.apply_2_qubit_gate(operation, bell_qubit_1, bell_qubit_1 + 1, noise=noise, pg=pg,
                                    user_operation=user_operation)
            self.apply_2_qubit_gate(operation, bell_qubit_2, bell_qubit_2 + 1, noise=noise, pg=pg,
                                    user_operation=user_operation)
            if measure:
                success = self.measure([bell_qubit_2, bell_qubit_1], noise=noise, pm=pm, user_operation=user_operation)
            else:
                success = True

    def double_selection(self, operation, bell_qubit_1, bell_qubit_2, noise=None, pn=None, pm=None, pg=None,
                         user_operation=True):
        """ Double selection as specified by Naomi Nickerson in https://www.nature.com/articles/ncomms2773.pdf """
        success = False
        while not success:
            self.single_selection(operation, bell_qubit_1, bell_qubit_2, measure=False, noise=noise, pn=pn, pm=pm, pg=pg,
                                  user_operation=user_operation)
            self.create_bell_pair(bell_qubit_1 - 1, bell_qubit_2 - 1, noise=noise, pn=pn, user_operation=user_operation)
            self.CZ(bell_qubit_1 - 1, bell_qubit_1, noise=noise, pg=pg, user_operation=user_operation)
            self.CZ(bell_qubit_2 - 1, bell_qubit_2, noise=noise, pg=pg, user_operation=user_operation)
            success = self.measure([bell_qubit_2 - 1, bell_qubit_1 - 1, bell_qubit_2, bell_qubit_1], noise=noise, pm=pm,
                                   user_operation=user_operation)

    def single_dot(self, operation, bell_qubit_1, bell_qubit_2, measure=True, noise=None, pn=None, pm=None,
                   pg=None, user_operation=True):
        """ single dot as specified by Naomi Nickerson in https://www.nature.com/articles/ncomms2773.pdf """
        success = False
        while not success:
            self.create_bell_pair(bell_qubit_1, bell_qubit_2, noise=noise, pn=pn, user_operation=user_operation)
            self.single_selection(X_gate, bell_qubit_1 - 1, bell_qubit_2 - 1, noise=noise, pn=pn, pm=pm, pg=pg,
                                  user_operation=user_operation)
            self.single_selection(Z_gate, bell_qubit_1 - 1, bell_qubit_2 - 1, noise=noise, pn=pn, pm=pm, pg=pg,
                                  user_operation=user_operation)
            self.apply_2_qubit_gate(operation, bell_qubit_1, bell_qubit_1 + 1, noise=noise, pg=pg,
                                    user_operation=user_operation)
            self.apply_2_qubit_gate(operation, bell_qubit_2, bell_qubit_2 + 1, noise=noise, pg=pg,
                                    user_operation=user_operation)
            if measure:
                success = self.measure([bell_qubit_2, bell_qubit_1], noise=noise, pm=pm, user_operation=user_operation)
            else:
                success = True

    def double_dot(self, operation, bell_qubit_1, bell_qubit_2, noise=None, pn=None, pm=None, pg=None,
                   user_operation=True):
        """ double dot as specified by Naomi Nickerson in https://www.nature.com/articles/ncomms2773.pdf """
        success = False
        while not success:
            self.single_dot(operation, bell_qubit_1, bell_qubit_2, measure=False, noise=noise, pn=pn, pm=pm, pg=pg,
                            user_operation=user_operation)
            self.single_selection(Z_gate, bell_qubit_1 - 1, bell_qubit_2 - 1, noise=noise, pn=pn, pm=pm, pg=pg,
                                  user_operation=user_operation)
            success = self.measure([bell_qubit_2, bell_qubit_1], noise=noise, pm=pm, user_operation=user_operation)

    """
        ---------------------------------------------------------------------------------------------------------
                                            Gate Noise Methods
        ---------------------------------------------------------------------------------------------------------  
    """

    def _N_single(self, pg, tqubit, density_matrix, num_qubits):
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
            density_matrix : csr_matrix
                Density matrix to which the noise should be applied to.
            num_qubits : int
                Number of qubits of which the density matrix is composed.
        """
        new_density_matrix = sp.csr_matrix((1-pg) * density_matrix +
                                           (pg / 3) * self._sum_pauli_error_single(tqubit,
                                                                                   density_matrix,
                                                                                   num_qubits=num_qubits))
        return new_density_matrix

    def _N(self, pg, cqubit, tqubit, density_matrix, num_qubits):
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
            density_matrix : csr_matrix
                Density matrix to which the noise should be applied to.
            num_qubits : int
                Number of qubits of which the density matrix is composed.
        """
        new_density_matrix = sp.csr_matrix((1 - pg) * density_matrix +
                                           (pg / 15) * self._double_sum_pauli_error(cqubit,
                                                                                    tqubit,
                                                                                    density_matrix,
                                                                                    num_qubits=num_qubits))
        return new_density_matrix

    @staticmethod
    def _N_network(density_matrix, pn, network_noise_type):
        """
            Parameters
            ----------
            density_matrix : sparse matrix
                Density matrix of the ideal Bell-pair.
            pn : float [0-1]
                Amount of network noise present in the system.
        """
        if network_noise_type == 1:
            return sp.csr_matrix((1-(4/3)*pn) * density_matrix + pn/3 * sp.eye(4, 4))
        else:
            error_density = sp.lil_matrix(4, 4)
            error_density[3, 3] = 1
            return sp.csr_matrix((1-pn) * density_matrix + pn * error_density)

    @staticmethod
    def _N_preparation(state, p_prep):
        opp_state = state
        if state == ket_0:
            opp_state = ket_1
        if state == ket_1:
            opp_state = ket_0
        if state == ket_p:
            opp_state = ket_m
        if state == ket_m:
            opp_state = ket_p

        error_state = State("Prep error state",
                            (1-p_prep) * state.vector + p_prep * opp_state.vector,
                            colored("~", 'red') + state.representation)

        return error_state

    def _N_decoherence(self, excluded_qubits, gate=None, times=None, p_dec=None):
        if gate and times is None:
            times = int(math.ceil(gate.duration/self.time_step))
        elif times is None:
            times = 1
        if p_dec is None:
            p_dec = self.p_dec

        # apply decoherence to the qubits not involved in the operation. REMOVING OF ANCILLA QUBITS THAT ARE USED
        # TO CALCULATE THE SUPEROPERATOR IS HARDCODED NOW FOR THE CASE OF A GHZ WITH 4 NODES
        included_qubits = set([i for i in range(self.num_qubits)]).difference(excluded_qubits)
        included_qubits = included_qubits.difference([(self.num_qubits - 1) - (2*i) for i in range(4)])

        drawn = False
        for _ in range(times):
            for inc_qubit in included_qubits:
                density_matrix, qubits, rel_qubit, rel_num_qubits = self._get_qubit_relative_objects(inc_qubit)
                new_density_matrix = self._N_single(p_dec, rel_qubit, density_matrix, num_qubits=rel_num_qubits)
                self._set_density_matrix(inc_qubit, new_density_matrix)
                if not drawn:
                    self._add_draw_operation("{}xD".format(times), inc_qubit, noise=True)
            drawn = True

    def _N_decoherence_fused(self, excluded_qubits, gate=None, times=None, p_dec=None):
        if gate and times is None:
            times = int(math.ceil(gate.duration/self.time_step))
        elif times is None:
            times = 1
        if p_dec is None:
            p_dec = self.p_dec

        # apply decoherence to the qubits not involved in the operation. REMOVING OF ANCILLA QUBITS THAT ARE USED
        # TO CALCULATE THE SUPEROPERATOR IS HARDCODED NOW FOR THE CASE OF A GHZ WITH 4 NODES
        included_qubits = set([i for i in range(self.num_qubits)]).difference(excluded_qubits)
        included_qubits = included_qubits.difference([(self.num_qubits - 1) - (2*i) for i in range(4)])

        skip_qubits = []
        loop_qubits = []
        for inc_qubit in included_qubits:
            if inc_qubit not in skip_qubits:
                _, qubits = self._qubit_density_matrix_lookup[inc_qubit]
                loop_qubits.append(inc_qubit)
                skip_qubits.extend(qubits)

        for qubit in loop_qubits:
            density_matrix, qubits, rel_qubit, rel_num_qubits = self._get_qubit_relative_objects(qubit)
            if rel_num_qubits == 1:
                for _ in range(times):
                    new_density_matrix = self._N_single(p_dec, rel_qubit, density_matrix, num_qubits=rel_num_qubits)
                    self._set_density_matrix(qubit, new_density_matrix)
                    density_matrix = new_density_matrix
            else:
                gates = [X_gate, Y_gate, Z_gate]
                total_gates = []
                for gate in gates:
                    total_gates.append(self._create_fused_single_qubit_gates(gate, included_qubits, qubits))

                for _ in range(times):
                    summed_matrix = sp.csr_matrix((2 ** rel_num_qubits, 2 ** rel_num_qubits))
                    for total_gate in total_gates:
                        summed_matrix = summed_matrix + total_gate.dot(CT(density_matrix, total_gate))

                    new_density_matrix = (1 - p_dec) * density_matrix + (p_dec/3) * summed_matrix
                    self._set_density_matrix(qubit, new_density_matrix)
                    density_matrix = new_density_matrix

        for qubit in included_qubits:
            self._add_draw_operation("{}xD".format(times), qubit, noise=True)

    def _create_fused_single_qubit_gates(self, gate, included_qubits, dm_qubits):
        gate_qubits = [qubit for qubit in included_qubits if qubit in dm_qubits]
        identity_qubits = [qubit for qubit in dm_qubits if qubit not in gate_qubits]
        grouped_gates = [list(map(itemgetter(1), g)) for k, g in it.groupby(enumerate(gate_qubits),
                                                                            lambda x: x[0]-x[1])]

        if identity_qubits != []:
            grouped_identity = [list(map(itemgetter(1), g))
                                for k, g in it.groupby(enumerate(identity_qubits), lambda x: x[0]-x[1])]
            start, first, second = ("g", grouped_gates, grouped_identity) if gate_qubits[0] < identity_qubits[0] \
                else ("i", grouped_identity, grouped_gates)
            all_grouped_sorted = list(it.chain.from_iterable(it.zip_longest(first, second)))
        else:
            start = "-"
            all_grouped_sorted = grouped_gates

        # def create_grouped_gate(gate, amount_qubits):
        #     if gate == "I":
        #         return sp.eye(2**amount_qubits, 2**amount_qubits)
        #     elif gate == "X":
        #         pass
        #     elif gate == "Z":
        #         total_gate = sp.lil_matrix(2**amount_qubits, 2**amount_qubits)
        #         diagonals = None
        #         non_zero_elements = [1, -1]
        #         for i in range(amount_qubits):
        #             if diagonals is None:
        #                 diagonals = non_zero_elements
        #             else:
        #                 diagonals = np.append(diagonals, (-1*diagonals))
        #         total_gate.setdiag(diagonals)
        #         return total_gate

        total_gate = None
        for i, group in enumerate(all_grouped_sorted):
            if group is None:
                continue
            current_gate = I_gate if (start == "i" and i % 2 == 0) or (start == "g" and i % 2 == 1) else gate
            gates = [current_gate for _ in group]
            if total_gate is None:
                total_gate = KP(*gates)
            else:
                total_gate = KP(total_gate, *gates)

        return total_gate

    def _sum_pauli_error_single(self, tqubit, density_matrix, num_qubits):
        """
            Private method that calculates the pauli gate sum part of the equation specified in _N_single
            method, namely

                SUM_A [A * rho * A^], --> A in {X, Y, Z}

            Parameters
            ----------
            tqubit: int
                Integer that indicates the target qubit. Note that the qubit counting starts at 0.
            density_matrix : csr_matrix
                Density matrix to which the noise should be applied to.
            num_qubits : int
                Number of qubits of which the density matrix is composed.

            Returns
            -------
            summed_matrix : sparse matrix
                Returns a sparse matrix which is the result of the equation mentioned above.
        """

        gates = [X_gate, Y_gate, Z_gate]
        summed_matrix = sp.csr_matrix((2**num_qubits, 2**num_qubits))

        for gate in gates:
            pauli_error = self._create_1_qubit_gate(gate, tqubit, num_qubits)
            summed_matrix = summed_matrix + pauli_error.dot(CT(density_matrix, pauli_error))
        return summed_matrix

    def _double_sum_pauli_error(self, qubit1, qubit2, density_matrix, num_qubits):
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
            density_matrix : csr_matrix
                Density matrix to which the noise should be applied to.
            num_qubits : int
                Number of qubits of which the density matrix is composed.

            Returns
            -------
            summed_matrix : sparse matrix
                Returns a sparse matrix which is the result of the equation mentioned above.
        """
        gates = [X_gate, Y_gate, Z_gate, I_gate]
        qubit2_matrices = []

        result = sp.csr_matrix(density_matrix.shape)
        for i, gate_1 in enumerate(gates):
            # Create the full system 1-qubit gate for qubit1
            A = self._create_1_qubit_gate(gate_1.matrix, qubit1, num_qubits=num_qubits)
            for j, gate_2 in enumerate(gates):
                # Create full system 1-qubit gate for qubit2, only once for every gate
                if i == 0:
                    qubit2_matrices.append(self._create_1_qubit_gate(gate_2.matrix, qubit2, num_qubits=num_qubits))

                # Skip the I*I case
                if i == j == len(gates) - 1:
                    continue

                B = qubit2_matrices[j]
                result = result + (A * B).dot(CT(density_matrix, (A * B)))

        return sp.csr_matrix(result)

    @staticmethod
    def _sum_bell_pairs():
        sum_bell_states = 1/2 * sp.lil_matrix(sp.eye(4, 4))
        sum_bell_states[0, 3], sum_bell_states[3, 0] = -1/2, -1/2
        sum_bell_states[1, 1], sum_bell_states[2, 2] = 1, 1
        return sum_bell_states

    """
        ---------------------------------------------------------------------------------------------------------
                                                Measurement Methods
        ---------------------------------------------------------------------------------------------------------   
    """

    def measure_first_N_qubits(self, N, qubit=None, measure=0, uneven_parity=False, noise=None, pm=None, p_dec=None, basis="X",
                               basis_transformation_noise=None, probabilistic=None, user_operation=True):
        """
            Method measures the first N qubits, given by the user, all in the 0 or 1 state.
            This will thus result in an even parity measurement. To also be able to enforce uneven
            parity measurements this should still be built!
            The density matrix of the system will be changed according to the measurement outcomes.

            *** MEASURED QUBITS WILL BE ERASED FROM THE SYSTEM AFTER MEASUREMENT, THIS WILL THUS
            DECREASE THE AMOUNT OF QUBITS IN THE SYSTEM WITH 'N' AS WELL. THE QUBIT INDICES WILL THEREFORE ALSO
            INCREASE WITH 'N', WHICH IS IMPORTANT FOR FUTURE OPERATIONS ***

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
                Whether the measurement should be done in the X-basis or in the computational basis (Z-basis)
            basis_transformation_noise : bool, optional, default=False
                Whether the H-gate that is applied to transform the basis in which the qubit is measured should be
                noisy (True) or noiseless (False)
            probabilistic : bool, optional, default=False
                Whether the measurement should be probabilistic. In case of an uneven parity in the outcome of the
                measurements, the method will return False else it returns True
            user_operation : bool, optional, default=True
                True if the user has requested the method and (else) False if it was invoked by an internal
                method.
        """
        if user_operation:
            self._user_operation_order.append({"measure_first_N_qubits": [N, measure, noise, pm, basis,
                                                                          basis_transformation_noise]})
        if noise is None:
            noise = self.noise
        if pm is None:
            pm = self.pm
        if p_dec is None:
            p_dec = self.p_dec
        if basis_transformation_noise is None:
            basis_transformation_noise = self.basis_transformation_noise
        if probabilistic is None:
            probabilistic = self.probabilistic
        if qubit is None:
            qubit = 0

        measurement_outcomes = []

        for qubit in range(N):
            if basis == "X":
                # Do not let the method draw itself, since the qubit will not be removed from the circuit drawing
                self.H(0, noise=basis_transformation_noise, draw=False, user_operation=False)

            qubit_density_matrix, _ = self._qubit_density_matrix_lookup[qubit]

            if probabilistic:
                prob_0, density_matrix_0 = self._measurement_first_qubit(qubit_density_matrix, measure=0, noise=noise,
                                                                         pm=pm)
                prob_1, density_matrix_1 = self._measurement_first_qubit(qubit_density_matrix, measure=1, noise=noise,
                                                                         pm=pm)

                density_matrices = [density_matrix_0, density_matrix_1]
                outcome = get_value_by_prob([0, 1], [prob_0, prob_1])
                new_density_matrix = density_matrices[outcome]
            else:
                outcome = measure
                if uneven_parity and qubit == 0:
                    outcome = abs(measure - 1)

                new_density_matrix = self._measurement_first_qubit(qubit_density_matrix, outcome, noise=noise,
                                                                   pm=pm)[1]

            self._set_density_matrix(0, new_density_matrix)
            self._correct_lookup_for_measurement_top()
            measurement_outcomes.append(outcome)
            # Remove the measured qubit from the system characteristics and add the operation to the draw_list
            self.num_qubits -= 1
            self.d = 2 ** self.num_qubits
            self._add_draw_operation("M_{}:{}".format(basis, outcome), qubit, noise)

            if noise and p_dec != 0:
                self._effective_measurements += (1+qubit)
                times = int(math.ceil(self.measurement_duration/self.time_step))
                self._N_decoherence([], times=times)
                self._effective_measurements -= (1+qubit)

        self._effective_measurements += N
        measurement_outcomes = iter(measurement_outcomes)
        parity_outcome = [True if i == j else False for i, j in zip(measurement_outcomes, measurement_outcomes)]
        return all(parity_outcome)

    def _measurement_first_qubit(self, density_matrix, measure=0, noise=True, pm=0.):
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
            density_matrix : csr_matrix
                Density matrix to which the top qubit should be measured.
            measure : int [0 or 1], optional, default=0
                The measurement outcome for the qubit, either 0 or 1.
            noise : bool, optional, default=None
                 Whether or not the measurement contains noise.
            pm : float [0-1], optional, default=0.
                The amount of measurement noise that is present (if noise is present).
        """
        d = density_matrix.shape[0]

        density_matrix_0 = density_matrix[:int(d / 2), :int(d / 2)]
        density_matrix_1 = density_matrix[int(d / 2):, int(d / 2):]

        if measure == 0 and noise:
            temp_density_matrix = (1 - pm) * density_matrix_0 + pm * density_matrix_1
        elif noise:
            temp_density_matrix = (1 - pm) * density_matrix_1 + pm * density_matrix_0
        elif measure == 0:
            temp_density_matrix = density_matrix_0
        else:
            temp_density_matrix = density_matrix_1

        prob = trace(temp_density_matrix)
        temp_density_matrix = temp_density_matrix / prob

        return prob, temp_density_matrix

    def measure(self, measure_qubits, outcome=0, uneven_parity=False, basis="X", noise=None, pm=None, probabilistic=None,
                basis_transformation_noise=None, user_operation=False):
        """
            Measurement that can be applied to any qubit.

            Parameters
            ----------
            qubit : int
                Indicates the qubit to be measured (qubit count starts at 0)
            outcome : int [0 or 1], optional, default=None
                The measurement outcome for the qubit, either 0 or 1. If None, the method will choose
                randomly according to the probability of the outcome.
            basis : str ["X" or "Z"], optional, default="X"
                Whether the qubit is measured in the X-basis or in the computational basis (Z-basis)
            basis_transformation_noise : bool, optional, default=False
                Whether the H-gate that is applied to transform the basis in which the qubit is measured should be
                noisy (True) or noiseless (False)
            user_operation : bool, optional, default=True
                True if the user has requested the method and (else) False if it was invoked by an internal
                method.
        """
        if user_operation:
            self._user_operation_order.append({"measure": [measure_qubits, outcome, basis]})
        if noise is None:
            noise = self.noise
        if pm is None:
            pm = self.pm
        if probabilistic is None:
            probabilistic = self.probabilistic
        if basis_transformation_noise is None:
            basis_transformation_noise = self.basis_transformation_noise

        if type(measure_qubits) == int:
            measure_qubits = [measure_qubits]

        measurement_outcomes = []

        for i, qubit in enumerate(measure_qubits):
            if basis == "X":
                self.H(qubit, noise=basis_transformation_noise, user_operation=False, draw=False)

            density_matrix, qubits, rel_qubit, rel_num_qubits = self._get_qubit_relative_objects(qubit)

            # If no specific measurement outcome is given it is chosen by the hand of the probability
            if probabilistic:
                if rel_qubit == 0:
                    prob1, density_matrix1 = self._measurement_first_qubit(density_matrix, measure=0, noise=False,
                                                                           pm=pm)
                    prob2, density_matrix2 = self._measurement_first_qubit(density_matrix, measure=1, noise=False,
                                                                           pm=pm)
                else:
                    prob1, density_matrix1 = self._get_measurement_outcome_probability(rel_qubit, density_matrix,
                                                                                       outcome=0,
                                                                                       keep_qubit=False)
                    prob2, density_matrix2 = self._get_measurement_outcome_probability(rel_qubit, density_matrix,
                                                                                       outcome=1,
                                                                                       keep_qubit=False)

                density_matrices = [density_matrix1, density_matrix2]
                outcome_new = get_value_by_prob([0, 1], [prob1, prob2])

                new_density_matrix = density_matrices[outcome_new]

                if noise:
                    new_density_matrix = (1 - pm) * new_density_matrix + pm * density_matrices[outcome_new ^ 1]

            else:
                outcome_new = outcome
                if uneven_parity and i == 0:
                    outcome_new = outcome ^ 1

                if rel_qubit == 0:
                    _, new_density_matrix = self._measurement_first_qubit(density_matrix, measure=outcome_new,
                                                                          noise=noise, pm=pm)
                else:
                    _, new_density_matrix = self._get_measurement_outcome_probability(rel_qubit, density_matrix,
                                                                                      outcome=outcome_new,
                                                                                      keep_qubit=False)
                    if noise:
                        _, wrong_density_matrix = self._get_measurement_outcome_probability(rel_qubit, density_matrix,
                                                                                            outcome=outcome_new ^ 1,
                                                                                            keep_qubit=False)
                        new_density_matrix = (1 - pm) * new_density_matrix + pm * wrong_density_matrix

            if basis == "X":
                density_matrix_measured = CT(ket_p) if outcome == 0 else CT(ket_m)
                self._correct_lookup_for_measurement_any(qubit, qubits, density_matrix_measured, new_density_matrix)
                self.H(qubit, noise=basis_transformation_noise, user_operation=False, draw=False)
            else:
                density_matrix_measured = CT(ket_0) if outcome == 0 else CT(ket_1)
                self._correct_lookup_for_measurement_any(qubit, qubits, density_matrix_measured, new_density_matrix)

            measurement_outcomes.append(outcome_new)
            self._add_draw_operation("M_{}:{}".format(basis, outcome_new), qubit, noise)


        measurement_outcomes = iter(measurement_outcomes)
        parity_outcome = [True if i == j else False for i, j in zip(measurement_outcomes, measurement_outcomes)]
        return all(parity_outcome)

    @staticmethod
    def _get_measurement_outcome_probability(qubit, density_matrix, outcome, keep_qubit=True):
        """
            Method returns the probability and new density matrix for the given measurement outcome of the given qubit.

            *** THIS METHOD IS VERY SLOW FOR LARGER SYSTEMS, SINCE IT DETERMINES THE SYSTEM STATE AFTER
            THE MEASUREMENT BY DIAGONALISING THE DENSITY MATRIX ***

            To explain the approach taken, consider that:
                    |a_1|   |b_1|   |c_1|   |a_1 b_1 c_1|                        |a_1 b_1 c_1 a_1 b_1 c_1 ... |
                    |   | * |   | * |   | = |a_1 b_1 c_2|  ---> density matrix:  |a_1 b_1 c_1 a_1 b_1 c_2 ... |
                    |a_2|   |b_2|   |c_2|   |a_1 b_2 c_1|                        |a_1 b_1 c_1 a_1 b_2 c_1 ... |
                                            |    ...    |                        |          ...               |

            When the second qubit (with the elements b_1 and b_2) is measured and the outcome is a 1, it means
            that b_1 is 0 and b_2 is 1. This thus means that all elements of the density matrix that are built up
            out of b_1 elements are 0 and only the elements not containing b_1 elements survive. This way a new
            density matrix can be constructed of which the trace is equal to the probability of this outcome occurring.
            Pattern of the elements across the density matrix can be compared with a chess pattern, where the square
            dimension reduce by a factor of 2 with the qubit number.

            Parameters
            ----------
            qubit : int
                qubit for which the measurement outcome probability should be measured
            density_matrix : csr_matrix
                Density matrix to which the qubit belongs
            outcome : int [0,1]
                Outcome for which the probability and resulting density matrix should be calculated
        """
        d = density_matrix.shape[0]
        dimension_block = int(d / (2 ** (qubit + 1)))
        non_zero_rows = density_matrix.nonzero()[0]
        non_zero_columns = density_matrix.nonzero()[1]

        if keep_qubit:
            new_density_matrix = sp.lil_matrix(copy.copy(density_matrix))
            start = 0 if outcome == 1 else dimension_block
            rows_columns_to_zero = [i+j for i in range(start, d, dimension_block * 2)
                                    for j in range(dimension_block)]
            non_zero_rows_unique = np.array(list(set(rows_columns_to_zero).intersection(non_zero_rows)))
            non_zero_columns_unique = np.array(list(set(rows_columns_to_zero).intersection(non_zero_columns)))
            if non_zero_columns_unique.size != 0:
                for row in non_zero_rows_unique:
                    column_indices = [i for i, e in enumerate(non_zero_rows) if e == row]
                    new_density_matrix[row, non_zero_columns[column_indices]] = 0
            if non_zero_columns_unique.size != 0:
                for column in non_zero_columns_unique:
                    row_indices = [i for i, e in enumerate(non_zero_columns) if e == column]
                    new_density_matrix[non_zero_rows[row_indices], column] = 0

            new_density_matrix = sp.csr_matrix(new_density_matrix)
        else:
            new_density_matrix = sp.lil_matrix((int(d/2), int(d/2)), dtype=density_matrix.dtype)
            start = 0 if outcome == 0 else dimension_block
            surviving_columns_rows = [i+j for i in range(start, d, dimension_block * 2)
                                    for j in range(dimension_block)]
            non_zero_rows_unique = np.array(list(set(surviving_columns_rows).intersection(non_zero_rows)))
            non_zero_columns_unique = np.array(list(set(surviving_columns_rows).intersection(non_zero_columns)))
            if non_zero_columns_unique.size != 0:
                for row in non_zero_rows_unique:
                    new_row = int(row/2) + (row % 2)
                    column_indices = [i for i, e in enumerate(non_zero_rows) if e == row]
                    valid_columns = [c for c in non_zero_columns[column_indices] if c in surviving_columns_rows]
                    valid_columns_new = [(int(c/2) + (c % 2)) for c in valid_columns]
                    new_density_matrix[new_row, valid_columns_new] = density_matrix[row, valid_columns]

        prob = trace(new_density_matrix)
        new_density_matrix = new_density_matrix / trace(new_density_matrix)

        return prob, new_density_matrix

    def _measurement_by_diagonalising(self, qubit, density_matrix, measure=0, eigenval=None, eigenvec=None):
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
        density_matrix : csr_matrix
                Density matrix to which the qubit belongs.
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

        d = density_matrix.shape[0]
        iterations = 2 ** qubit
        step = int(d / (2 ** (qubit + 1)))
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
            result = np.zeros(density_matrix.shape)
            for i, eigenvalue in enumerate(eigenvalues):
                eigenvector = eigenvectors[i]
                result += eigenvalue * CT(eigenvector)

            return prob, sp.csr_matrix(np.round(result / np.trace(result), 10))

        return prob, sp.csr_matrix((d, d))

    """
        ---------------------------------------------------------------------------------------------------------
                                            Density Matrix calculus Methods
        ---------------------------------------------------------------------------------------------------------     
    """

    @staticmethod
    def diagonalise(density_matrix, option=0):
        """" Returns the Eigenvalues and Eigenvectors of the density matrix. option=1 returns only the Eigenvalues"""
        if option == 0:
            return eig(density_matrix.toarray())
        if option == 1:
            return eigh(density_matrix.toarray(), eigvals_only=True)

    def get_non_zero_prob_eigenvectors(self, density_matrix, d, decimals=10):
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
        eigenvalues, eigenvectors = self.diagonalise(density_matrix)
        non_zero_eigenvalues_index = np.argwhere(np.round(eigenvalues, decimals) != 0).flatten()
        eigenvectors_list = []

        for index in non_zero_eigenvalues_index:
            eigenvector = sp.csr_matrix(np.round(eigenvectors[:, index].reshape(d, 1), 8))
            eigenvectors_list.append(eigenvector)

        return eigenvalues[non_zero_eigenvalues_index], eigenvectors_list

    def print_non_zero_prob_eigenvectors(self):
        """ Prints a clear overview of the non-zero Eigenvalues and their Eigenvectors to the console """
        eigenvalues, eigenvectors = self.get_non_zero_prob_eigenvectors()

        print_line = "\n\n ---- Eigenvalues and Eigenvectors ---- \n\n"
        for i, eigenvalue in enumerate(eigenvalues):
            print_line += "eigenvalue: {}\n\neigenvector:\n {}\n---\n".format(eigenvalue, eigenvectors[i].toarray())

        self._print_lines.append(print_line + "\n ---- End Eigenvalues and Eigenvectors ----\n")
        if not self._thread_safe_printing:
            self.print()

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
                        one_qubit_states_in_n_qubit_state.append(sign * eigenvector_index_value
                                                                 * copy.copy(ket_0.vector))
                    else:
                        one_qubit_states_in_n_qubit_state.append(sign * eigenvector_index_value
                                                                 * copy.copy(ket_1.vector))

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

    def get_superoperator(self, qubits, proj_type, stabilizer_protocol=False, save_noiseless_density_matrix=False,
                          combine=True, most_likely=True, print_to_console=True, file_name_noiseless=None,
                          file_name_measerror=None, no_color=False, to_csv=False, csv_file_name=None):
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
                necessary for the postprocessing of the superoperator results if 'combine' is set to True and used if
                stabilizer_protocol is set to True.
            stabilizer_protocol : bool, optional, default=False
                If the superoperator is calculated for a stabilizer measurement protocol (for example Stringent or
                Expedient).
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
                qasm_file name of the noiseless variant of the density matrix of the noisy system. Use this option if
                density matrix has been named manually and this one should be used for the calculations.
            file_name_measerror : str, optional, default=None
                qasm_file name of the noiseless variant with measurement error of the density matrix of the noisy
                system. Use this option if density matrix has been named manually and this one should be used for the
                calculations.
            no_color : bool, optional, default=False
                Indicates if the output of the superoperator to the console should not contain color, when for example
                the used console does not support color codes.
            to_csv : bool, optional, default=False
                Whether the results of the superoperator should be saved to a csv file.
            csv_file_name : str, optional, default=None
                The file name that should be used for the csv file. If not supplied, the system will use generic naming
                and the file will be saved to the 'oopsc/superoperator/csv_files' folder.
        """
        noiseless_density_matrix = self._get_noiseless_density_matrix(stabilizer_protocol=stabilizer_protocol,
                                                                      proj_type=proj_type,
                                                                      save=save_noiseless_density_matrix,
                                                                      file_name=file_name_noiseless)
        measerror_density_matrix = self._get_noiseless_density_matrix(measure_error=True,
                                                                      stabilizer_protocol=stabilizer_protocol,
                                                                      proj_type=proj_type,
                                                                      save=save_noiseless_density_matrix,
                                                                      file_name=file_name_measerror)
        superoperator = []

        # Get all combinations of gates ([X, Y, Z, I]) possible on the given qubits
        total_density_matrix = self.get_combined_density_matrix(qubits)
        num_qubits = int(math.log(total_density_matrix.shape[0])/math.log(2))
        all_gate_combinations = self._all_single_qubit_gate_possibilities(qubits, num_qubits=num_qubits)

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

            fid_no_me = fidelity_elementwise(error_density_matrix, total_density_matrix)
            fid_me = fidelity_elementwise(me_error_density_matrix, total_density_matrix)

            operators = [list(applied_gate.keys())[0] for applied_gate in combination]

            superoperator.append(SuperoperatorElement(fid_me, True, operators))
            superoperator.append(SuperoperatorElement(fid_no_me, False, operators))

        # Possible post-processing options for the superoperator
        if combine:
            superoperator = self._fuse_equal_config_up_to_permutation(superoperator, proj_type)
        if combine and most_likely:
            superoperator = self._remove_not_likely_configurations(superoperator)

        if to_csv:
            self._superoperator_to_csv(superoperator, proj_type, file_name=csv_file_name)
        if print_to_console:
            self._print_superoperator(superoperator, no_color)
        return superoperator

    def _get_noiseless_density_matrix(self, stabilizer_protocol, proj_type, measure_error=False, save=True,
                                      file_name=None):
        """
            Private method to calculate the noiseless variant of the density matrix.
            It traverses the operations on the system by the hand of the '_user_operation_order' attribute. If the
            noiseless matrix is present in the 'saved_density_matrices' folder, the method will use this instead
            of recalculating the circuits. When no file name is given, the noiseless density matrix is searched for
            based on the user operations applied to the noisy circuit (see method '_absolute_file_path_from_circuit').

            Parameters
            ----------
            stabilizer_protocol : bool
                If the noiseless density matrix is one of a stabilizer measurement protocol (for example Stringent or
                Expedient). This leads to a speed-up, since the noiseless density matrix can be assumed equal to the
                noiseless density matrix of a stabilizer measurement in a monolithic architecture.
            proj_type : str, options: "X" or "Z"
                Specifies the type of stabilizer for which the superoperator should be calculated.
            measure_error: bool, optional, default=False
                Specifies if the measurement outcome should be opposite of the ideal circuit.
            save : bool
                Whether or not the calculated noiseless version of the circuit should be saved.
                This saved matrix will a next time be used if the same system is analysed wth this method.
            file_name : str
                File name of the density matrix qasm_file that should be used as noiseless density matrix. Note that
                specifying this with an existing qasm_file name will directly return this density matrix.

            Returns
            -------
            noiseless_density_matrix : sparse matrix
                The density matrix of the current system, but without noise
        """
        if stabilizer_protocol:
            return self._noiseless_stabilizer_protocol_density_matrix(proj_type, measure_error)
        if file_name is None:
            file_name = self._absolute_file_path_from_circuit(measure_error)

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
            sp.save_npz(file_name, qc_noiseless.total_density_matrix())

        return qc_noiseless.total_density_matrix()

    @staticmethod
    def _noiseless_stabilizer_protocol_density_matrix(proj_type, measure_error):
        """
            Method returns the noiseless density matrix of a stabilizer measurement in the monolithic architecture.
            Since this density matrix is equal for all equal kinds of stabilizer measurement protocols, this method
            can be used to gain a speed-up in obtaining the noiseless density matrix.

            Parameters
            ----------
            proj_type : str, options: "X" or "Z"
                Specifies the type of stabilizer for which the superoperator should be calculated.
            measure_error : bool
                True if the noiseless density matrix should contain a measurement error.
        """
        qc = QuantumCircuit(9, 2)
        qc.set_qubit_states({0: ket_p})
        gate = Z_gate if proj_type == "Z" else X_gate
        for i in range(1, qc.num_qubits, 2):
            qc.apply_2_qubit_gate(gate, 0, i)

        qc.measure([0], outcome=0 if not measure_error else 1)

        return qc.get_combined_density_matrix([1])

    def _file_name_from_circuit(self, measure_error=False, general_name="circuit", extension=""):
        """
            Returns the file name of the Quantum Circuit based on the initial parameters and the user operations
            applied to the circuit.

            Parameters
            ----------
            measure_error : bool, optional, default=False
                This variable is used for the case of density matrix naming for the noiseless density matrices.
                This ensures explicit naming of a density matrix containing a measurement error. For more info see
                the 'get_superoperator' and '_get_noiseless_density_matrix'.
            general_name : str, optional, default="circuit"
                To specify the file name more, one can add a custom start of the file name. Default is 'circuit'.
            extension : str, optional, default=""
                Use this argument if the file name needs a specific type of extension. By default, it will NOT append
                an extension.
        """
        # Create an hash id, based on the operation and there order on the system and use this for the filename
        init_params_id = str(self._init_parameters)
        user_operation_id = "".join(["{}{}".format(list(d.keys())[0], list(d.values())[0])
                              for d in self._user_operation_order])
        total_id = init_params_id + user_operation_id
        hash_id = hashlib.sha1(total_id.encode("UTF-8")).hexdigest()[:10]
        file_name = "{}{}_{}{}".format(general_name, ("_me" if measure_error else ""), hash_id, extension)

        return file_name

    def _absolute_file_path_from_circuit(self, measure_error, kind="dm"):
        """
            Returns a file path to a file based on what kind of object needs to be saved. The kind of files that
            are supported, including their standard directory can be found below in the parameters section.

            Parameters
            ----------
            measure_error : bool
                True if the ideal density matrix containing a measurement error should be returned.
            kind : str, optional, default="dm"
                Kind of file of which the absolute file path should be obtained. In this moment in time the options are
                    * "dm"
                        Density matrix file. Directory will be the 'saved_density_matrix' folder.
                    * "qasm"
                        Qasm file. Directory will be the 'latex_circuit' folder.
                    * "os"
                        Superoperator file. Directory will be the 'oopsc/superoperator/csv_files/' folder.

            Returns
            -------
            file_name : str
                Returns the file_name of the ideal (or ideal up to measurement error if parameter 'measure_error' is set
                to True) density matrix of the noisy QuantumCircuit object.
        """
        if kind == "dm":
            file_name = self._file_name_from_circuit(measure_error, general_name="density_matrix", extension=".npz")
            file_path = os.path.join(os.path.dirname(__file__), "saved_density_matrices", file_name)
        elif kind == "qasm":
            file_name = self._file_name_from_circuit(measure_error, extension=".qasm")
            file_path = os.path.join(os.path.dirname(__file__), "latex_circuit", file_name)
        elif kind == "so":
            file_name = self._file_name_from_circuit(measure_error, general_name="superoperator", extension=".csv")
            file_path = os.path.join(SuperoperatorElement.file_path(), "csv_files", file_name)
        else:
            file_name = self._file_name_from_circuit(measure_error, extension=".npz")
            file_path = os.path.join(os.getcwd(), file_name)
            self._print_lines.append("\nkind: '{}' was not recognized. Please see method documentation for supported kinds. "
                  "File path is now: '{}'".format(kind, file_path))

        return file_path

    def _all_single_qubit_gate_possibilities(self, qubits, num_qubits):
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
        operations = [X_gate, Y_gate, Z_gate, I_gate]
        gate_combinations = []

        for qubit in qubits:
            _, _, rel_qubit, _ = self._get_qubit_relative_objects(qubit)
            gates = []
            for operation in operations:
                gates.append({operation.representation: self._create_1_qubit_gate(operation.matrix, rel_qubit,
                                                                                  num_qubits=num_qubits)})
            gate_combinations.append(gates)

        return list(product(*gate_combinations))

    @staticmethod
    def _fuse_equal_config_up_to_permutation(superoperator, proj_type):
        """
            Post-processing method for the superoperator which fuses similar Pauli-error configurations inside the
            superoperator up to permutation. This is done by sorting the error configurations and comparing them after.
            If equal, the probabilities will be summed and saved as one new entry.

            Parameters
            ----------
            superoperator : list
                Superoperator obtained in the 'get_superoperator' method. Containing all the probabilities of the
                possible Pauli-error configurations on the data qubits.
            proj_type : str ['Z' or 'X']
                The stabilizer type of the to be analysed superoperator. This is necessary in order to determine the
                degenerate configurations, for example [I,I,Z,Z] and [Z,Z,I,I] that on first sight look as if they have
                to be treated equally, but in fact they are degenerate and the probabilities should not be summed (since
                this will cause the total probability to exceed 1).

            Returns
            -------
            sorted_superoperator : list
                New superoperator that now contains only one entry per similar Pauli-error configurations up to
                permutations. The new probability of this one entry is the summed probability of all the similar
                configurations that were fused.

            Example
            -------
            The superoperator contains, among others, the configurations [X,I,I,I], [I,X,I,I], [I,I,X,I] and [I,I,I,X].
            These Pauli-error configurations on the data qubits are similar up to permutations. The method will
            eventually end up making one entry, namely [I,I,I,X], in the returned new superoperator. The according
            probability will be equal to the sum of the probabilities of the 4 configurations.
        """
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
        """
            Post-processing method for the superoperator which removes the degenerate configurations of the
            superoperator based on the fact that the Pauli-error configuration with the most 'I' operations is the most
            likely to have occurred.

            Parameters
            ----------
            superoperator : list
                Superoperator obtained in the 'get_superoperator' method. Containing all the probabilities of the
                possible Pauli-error configurations on the data qubits.

            Returns
            -------
            sorted_superoperator : list
                Returns the superopertor with the not-likely degenerate configurations entries removed. Note that is a
                full removal, thus the probability is removed from the list (and not summed as in the 'fuse'
                post-processing).

            Example
            -------
            Consider the superoperator with, among others, the degenerate entries [Z,Z,Z,X] and [I,I,I,X]. In this
            method, it is assumed that the configuration [I,I,I,X] is more likely to have occurred than the other and
            therefore only this configuration is kept in the returned superoperator. Effectively, this means that the
            [Z,Z,Z,X] is removed from the superoperator together with the according probability.
        """
        for supop_el_a, supop_el_b in combinations(superoperator, 2):
            if supop_el_a.probability_lie_equals(supop_el_b):
                if supop_el_a.error_array.count("I") > supop_el_b.error_array.count("I") \
                        and supop_el_b in superoperator:
                    superoperator.remove(supop_el_b)
                elif supop_el_a.error_array.count("I") < supop_el_b.error_array.count("I") \
                        and supop_el_a in superoperator:
                    superoperator.remove(supop_el_a)

        return superoperator

    def _print_superoperator(self, superoperator, no_color):
        """ Prints the superoperator in a clear way to the console """
        self._print_lines.append("\n---- Superoperator ----\n")

        total = sum([supop_el.p for supop_el in superoperator])
        for supop_el in sorted(superoperator):
            probability = supop_el.p
            self._print_lines.append("\nProbability: {}".format(probability))
            config = ""
            for gate in supop_el.error_array:
                if gate == "X":
                    config += (colored(gate, 'red') + " ") if not no_color else gate
                elif gate == "Z":
                    config += (colored(gate, 'cyan') + " ") if not no_color else gate
                elif gate == "Y":
                    config += (colored(gate, 'magenta') + " ") if not no_color else gate
                elif gate == "I":
                    config += (colored(gate, 'yellow') + " ") if not no_color else gate
                else:
                    config += (gate + " ")
            me = "me" if supop_el.lie else "no me"
            self._print_lines.append("\n{} - {}".format(config, me))
        self._print_lines.append("\n\nSum of the probabilities is: {}\n".format(total))
        self._print_lines.append("\n---- End of Superoperator ----\n")

        if not self._thread_safe_printing:
            self.print()

    def _superoperator_to_csv(self, superoperator, proj_type, file_name=None):
        """
            Save the obtained superoperator results to a csv file format that is suitable with the superoperator
            format that is used in the (distributed) surface code simulations.

            *** IN THIS METHOD IT IS ASSUMED Z AND X ERRORS ARE EQUALLY LIKELY TO OCCUR, SUCH THAT THE RESULTS FOR THE
             OPPOSITE PROJECTION TYPES (PLAQUETTE IF STAR AND VICE VERSA) ONLY DIFFER BY A HADAMARD TRANSFORM ON THE
             ERROR CONFIGURATIONS (SO IIIX -> IIIY) AND APPLYING THIS WILL LEAD TOT RESULTS OF THE OPPOSITE PROJECTION
             TYPE. ***

            superoperator : list
                The superoperator results, a list containing the SuperoperatorElement objects.
            proj_type : str, options: {"X", "Z"}
                The stabilizer type that has been analysed, options are "X" or "Z"
            file_name : str, optional, default=None
                User specified file name that should be used to save the csv file with. The file will always be stored
                in the 'csv_files' directory, so the string should NOT contain any '/'. These will be removed.
        """
        probs = []
        lies = []
        p_error_arrays = []
        s_error_arrays = []
        for supop_el in sorted(superoperator):
            probs.append(supop_el.p)
            lies.append(supop_el.lie)
            error_array = "".join(supop_el.error_array)
            p_error_arrays.append(error_array)
            # When Z and X errors are equally likely, symmetry between proj_type and only H gate difference in
            # error_array
            s_error_arrays.append(error_array.translate(str.maketrans({'X': 'Z', 'Z': 'X'})))

        stab_type = 'p' if proj_type == "Z" else 's'
        opp_stab = 's' if proj_type == "Z" else 'p'

        df_values = pd.DataFrame({(stab_type + '_prob'): probs,
                           (stab_type + '_lie'): lies,
                           (stab_type + '_error'): p_error_arrays,
                           (opp_stab + '_prob'): probs,
                           (opp_stab + '_lie'): lies,
                           (opp_stab + '_error'): s_error_arrays})
        df_parameters = pd.DataFrame({"pg": [self.pg],
                                      "pm": [self.pm]})

        if self.pn and self.pn != 0.0:
            df_parameters.append({"pn": [self.pn]})

        df = pd.concat([df_values, df_parameters], axis=1)

        path_to_file = self._absolute_file_path_from_circuit(measure_error=False, kind="so")
        if file_name is None:
            self._print_lines.append("\nFile name was created manually and is: {}\n".format(path_to_file))
        else:
            path_to_file = os.path.join(path_to_file.rpartition(os.sep)[0], file_name.replace(os.sep, "") + ".csv")
            self._print_lines.append("\nCSV file has been saved at: {}\n".format(path_to_file))
        df.to_csv(path_to_file, sep=';', index=False)
        if not self._thread_safe_printing:
            self.print()

    def get_kraus_operator(self, print_to_console=True):
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

        kraus_decomposition = zip(probabilities, kraus_ops)

        if print_to_console:
            self._print_kraus_operators(kraus_decomposition)

        return kraus_decomposition

    def _print_kraus_operators(self, kraus_decomposition):
        """ Prints a clear overview of the effective operations that have happened on the individual qubits """
        print_lines = ["\n---- Kraus operators per qubit ----\n"]
        for prob, operators_per_qubit in kraus_decomposition:
            print_lines.append("\nProbability: {:.8}\n".format(prob.real))
            for data_qubit, operator in enumerate(operators_per_qubit):
                data_qubit_line = "Data qubit {}: \n {}\n".format(data_qubit, operator.toarray())
                operator_name = gate_name(operator.toarray().round(1))
                if operator_name is not None:
                    data_qubit_line += "which is equal to an {} operation\n\n".format(operator_name)
                print_lines.append(data_qubit_line)
        print_lines.append("\n---- End of Kraus operators per qubit ----\n\n")

        self._print_lines.append(*print_lines)

        if not self._thread_safe_printing:
            self.print()

    """
        ----------------------------------------------------------------------------------------------------------
                                            Circuit drawing Methods
        ----------------------------------------------------------------------------------------------------------     
    """

    def draw_circuit(self, no_color=False):
        """ Draws the circuit that corresponds to the operation that have been applied on the system,
        up until the moment of calling. """
        legenda = "\n--- Circuit ---\n\n @: noisy Bell-pair, #: perfect Bell-pair, o: control qubit " \
                  "(with target qubit at same level), [X,Y,Z,H]: gates, M: measurement,"\
                  " {}: noisy operation (gate/measurement)\n".format("~" if no_color else colored("~", 'red'))
        init = self._draw_init(no_color)
        self._draw_gates(init, no_color)
        init[-1] += "\n\n"
        self._print_lines.append(legenda)
        self._print_lines.extend(init)
        if not self._thread_safe_printing:
            self.print()

    def draw_circuit_latex(self, meas_error=False):
        qasm_file_name = self._create_qasm_file(meas_error)
        create_pdf_from_qasm(qasm_file_name, qasm_file_name.replace(".qasm", ".tex"))

    def _draw_init(self, no_color):
        """ Returns an array containing the visual representation of the initial state of the qubits. """
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        init_state_repr = []
        for state in self._qubit_array:
            init_state_repr.append("\n\n{} ---".format(ansi_escape.sub("", state.representation) if no_color else
                                                       state.representation))

        for a, b in it.combinations(enumerate(init_state_repr), 2):
            # Since colored ansi code is shown as color and not text it should be stripped for length comparison
            a_stripped = ansi_escape.sub("", init_state_repr[a[0]])
            b_stripped = ansi_escape.sub("", init_state_repr[b[0]])

            if (diff := len(b_stripped) - len(a_stripped)) > 0:
                state_repr_split = init_state_repr[a[0]].split(" ")
                init_state_repr[a[0]] = state_repr_split[0] + ((diff+1) * " ") + state_repr_split[1]
            elif (diff := len(a_stripped) - len(b_stripped)) > 0:
                state_repr_split = init_state_repr[b[0]].split(" ")
                init_state_repr[b[0]] = state_repr_split[0] + ((diff+1) * " ") + state_repr_split[1]

        return init_state_repr

    def _draw_gates(self, init, no_color):
        """ Adds the visual representation of the operations applied on the qubits """
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

        for draw_item in self._draw_order:
            gate = draw_item[0]
            qubits = draw_item[1]
            noise = draw_item[2]

            if type(qubits) == tuple:
                if type(gate) in [SingleQubitGate, TwoQubitGate]:
                    control = gate.control_repr if type(gate) == TwoQubitGate else "o"
                    gate = gate.representation
                elif gate == "#":
                    control = gate
                else:
                    control = "o"

                if noise:
                    control = "~" + control if no_color else colored("~", 'red') + control
                    gate = "~" + gate if no_color else colored('~', 'red') + gate

                cqubit = qubits[0]
                tqubit = qubits[1]
                init[cqubit] += "---{}---".format(control)
                init[tqubit] += "---{}---".format(gate)
            else:
                if type(gate) == SingleQubitGate:
                    gate = gate.representation
                if noise:
                    gate = "~" + gate if no_color else colored("~", 'red') + gate
                init[qubits] += "---{}---".format(gate)

            for a, b in it.combinations(enumerate(init), 2):
                # Since colored ansi code is shown as color and not text it should be stripped for length comparison
                a_stripped = ansi_escape.sub("", init[a[0]])
                b_stripped = ansi_escape.sub("", init[b[0]])

                if (diff := len(b_stripped) - len(a_stripped)) > 0:
                    init[a[0]] += diff * "-"
                elif (diff := len(a_stripped) - len(b_stripped)) > 0:
                    init[b[0]] += diff * "-"

    def _create_qasm_file(self, meas_error):
        """
            Method constructs a qasm file based on the 'self._draw_order' list. It returns the file path to the
            constructed qasm file.

            Parameters
            ----------
            meas_error : bool
                Specify if there has been introduced an measurement error on purpose to the QuantumCircuit object.
                This is needed to create the proper file name.
        """
        file_path = self._absolute_file_path_from_circuit(meas_error, kind="qasm")
        ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
        file = open(file_path, 'w')

        file.write("\tdef meas,0,'M'\n")
        file.write("\tdef n-meas,0,'\widetilde{M}'\n")
        file.write("\tdef bell,1,'B'\n")
        file.write("\tdef n-bell,1,'\widetilde{B}'\n\n")
        file.write("\tdef n-cnot,1,'\widetilde{X}'\n")
        file.write("\tdef n-cz,1,'\widetilde{Z}'\n")
        file.write("\tdef n-cnot,1,'\widetilde{X}'\n")
        file.write("\tdef n-x,0,'\widetilde{X}'\n")
        file.write("\tdef n-h,0,'\widetilde{H}'\n")
        file.write("\tdef n-y,0,'\widetilde{Y}'\n")

        for i in range(len(self._qubit_array)):
            file.write("\tqubit " + str(i) + "\n")

        file.write("\n")

        for draw_item in self._draw_order:
            gate = draw_item[0]
            qubits = draw_item[1]
            noise = draw_item[2]

            if type(gate) in [SingleQubitGate, TwoQubitGate]:
                gate = gate.representation

            gate = ansi_escape.sub("", gate)
            gate = gate.lower()
            if type(qubits) == tuple:
                if 'z' in gate:
                    gate = "c-z" if not noise else "n-cz"
                elif 'x' in gate:
                    gate = 'cnot' if not noise else "n-cnot"
                elif '#' in gate:
                    gate = 'bell' if not noise else "n-bell"
                cqubit = qubits[0]
                tqubit = qubits[1]
                file.write("\t" + gate + " " + str(cqubit) + "," + str(tqubit) + "\n")
            elif "m" in gate:
                gate = "meas " if "~" not in gate else "n-meas "
                file.write("\t" + gate + str(qubits) + "\n")
            else:
                gate = gate if "~" not in gate or not noise else "n-"+gate
                file.write("\t" + gate + " " + str(qubits) + "\n")

        file.close()

        return file_path

    def _add_draw_operation(self, operation, qubits, noise=False):
        """
            Adds an operation to the draw order list.

            Notes
            -----
            **Note** :
                Since measurements and additions of qubits change the qubit indices dynamically, this will be
                accounted for in this method when adding a draw operation. The '_effective_measurement' attribute keeps
                track of how many qubits have effectively been measured, which means they have not been reinitialised
                after measurement (by creating a Bell-pair at the top or adding a top qubit). The '_measured_qubits'
                attribute contains all the qubits that have been measured and are not used anymore after (in means of
                the drawing scheme).

            **2nd Note** :
                Please consider that the drawing of the circuit can differ from reality due to this dynamic
                way of changing the qubit indices with measurement and/or qubit addition operations. THIS EFFECTIVELY
                MEANS THAT THE CIRCUIT REPRESENTATION MAY NOT ALWAYS PROPERLY REPRESENT THE APPLIED CIRCUIT WHEN USING
                MEASUREMENTS AND QUBIT ADDITIONS.
        """

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
        item = [operation, qubits, noise]
        self._draw_order.append(item)

    def _correct_drawing_for_n_top_qubit_additions(self, n=1):
        """
            Corrects the self._draw_order list for addition of n top qubits.

            When a qubit gets added to the top of the stack, it gets the index 0. This means that the indices of the
            already existing qubits increase by 1. This should be corrected for in the self._draw_order list, since
            the qubit references used the 'old' qubit index.

            *** Note that for the actual qubit operations that already have been applied to the system the addition of
            a top qubit is not of importance, but after addition the user should know this index change for future
            operations ***

            Parameters
            ----------
            n : int, optional, default=1
                Amount of added top qubits that should be corrected for.
        """
        self._measured_qubits.extend([i for i in range(self._effective_measurements)])
        self._measured_qubits = [(x + n) for x in self._measured_qubits]
        self._effective_measurements = 0
        for i, draw_item in enumerate(self._draw_order):
            operation = draw_item[0]
            qubits = draw_item[1]
            noise = draw_item[2]
            if type(qubits) == tuple:
                self._draw_order[i] = [operation, (qubits[0] + n, qubits[1] + n), noise]
            else:
                self._draw_order[i] = [operation, qubits + n, noise]

    def _correct_drawing_for_circuit_fusion(self, other_draw_order, num_qubits_other):
        new_draw_order = other_draw_order
        for draw_item in self._draw_order:
            operation = draw_item[0]
            if type(draw_item[1]) == tuple:
                qubits = tuple([i + num_qubits_other for i in draw_item[1]])
            else:
                qubits = draw_item[1] + num_qubits_other
            noise = draw_item[2]
            new_draw_item = [operation, qubits, noise]
            new_draw_order.append(new_draw_item)
        self._draw_order = new_draw_order

    def save_density_matrix(self, filename=None):
        if filename is None:
            filename = self._absolute_file_path_from_circuit(measure_error=False, kind='dm')

        sp.save_npz(filename, self.total_density_matrix())

        self._print_lines.append("\nFile successfully saved at: {}".format(filename))

    def fuse_circuits(self, other):
        if type(other) != QuantumCircuit:
            raise ValueError("Other should be of type QuantumCircuit, not {}".format(type(other)))

        if self.noise and self.p_dec > 0:
            duration_difference = self.total_duration - other.total_duration
            if duration_difference < 0:
                times = int(math.ceil(abs(duration_difference)/self.time_step))
                self._N_decoherence([], times)
            elif duration_difference > 0:
                times = int(math.ceil(abs(duration_difference)/other.time_step))
                other._N_decoherence([], times)

        self._fused = True
        self.num_qubits = self.num_qubits + other.num_qubits
        self.d = 2 ** self.num_qubits
        self._correct_lookup_for_circuit_fusion(other._qubit_density_matrix_lookup)
        self._correct_drawing_for_circuit_fusion(other._draw_order, len(other._qubit_array))
        self._effective_measurements = other._effective_measurements + self._effective_measurements
        self._measured_qubits = other._measured_qubits + self._measured_qubits
        self._print_lines = other._print_lines + self._print_lines
        self._qubit_array = other._qubit_array + self._qubit_array

    def __repr__(self):
        return "\nQuantumCircuit object containing {} qubits\n".format(self.num_qubits)

    def __copy__(self):
        new_circuit = QuantumCircuit(self.num_qubits)
        new_circuit.density_matrix = self.density_matrix.copy()
        new_circuit.noise = self.noise
        new_circuit.pg = self.pg
        new_circuit.pm = self.pm
        new_circuit.pn = self.pn
        new_circuit._user_operation_order = self._user_operation_order.copy()
        new_circuit._measured_qubits = self._measured_qubits.copy()
        new_circuit._effective_measurements = self._effective_measurements
        new_circuit._draw_order = self._draw_order.copy()
        new_circuit._qubit_array = self._qubit_array.copy()
        new_circuit._init_type = self._init_type

        return new_circuit

    def copy(self):
        return self.__copy__()

    def print(self, empty_print_lines=True):
        print(*self._print_lines)
        if empty_print_lines:
            self._print_lines.clear()
