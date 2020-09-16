import os
import sys
sys.path.insert(1, os.path.abspath(os.getcwd()))
from circuit_simulation.basic_operations.basic_operations import (
    CT, KP, get_value_by_prob, fidelity_elementwise
)
from circuit_simulation.states.states import *
from circuit_simulation.gates.gates import *
from circuit_simulation.gates.gate import SingleQubitGate
from circuit_simulation.qubit.qubit import Qubit
from circuit_simulation.sub_circuit.sub_quantum_circuit import SubQuantumCircuit
from scipy import sparse as sp
import hashlib
from oopsc.superoperator.superoperator import SuperoperatorElement
from termcolor import colored
from circuit_simulation._draw.qasm_to_pdf import create_pdf_from_qasm
from fractions import Fraction as Fr
import math
import random


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
                True. In case pm_1 is specified, this value holds as the measurement error when a 0-state is
                supposed to be measured.
            pm_1 : float [0-1], optional, default=None
                The amount of measurement error when a 1-state is supposed to be measured. This can be used in case
                there is a difference in measurement error between an 0-state and an 1-state.
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
            thread_safe_printing : bool, optional, default=False
                If working with threads, this can be set to True. This prevents print statements from being
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
                 pm_1=None, pn=None, p_dec=0, p_bell_success=1, time_step=1, measurement_duration=1,
                 bell_creation_duration=1, probabilistic=False, network_noise_type=0, no_single_qubit_error=False,
                 thread_safe_printing=False, single_qubit_gate_lookup=None, two_qubit_gate_lookup=None):
        self.num_qubits = num_qubits
        self.d = 2 ** num_qubits
        self.noise = noise
        self.pg = pg
        self.pm = pm
        self.pm_1 = pm_1
        self.pn = pn
        self.p_dec = p_dec
        self.p_bell_success = p_bell_success
        self.bell_creation_duration = bell_creation_duration
        self.network_noise_type = network_noise_type
        self.time_step = time_step
        self.measurement_duration = measurement_duration
        self.probabilistic = probabilistic
        self.no_single_qubit_error = no_single_qubit_error
        self.density_matrices = None
        self.total_duration = 0
        self._init_type = init_type
        self._qubit_array = num_qubits * [ket_0]
        self._draw_order = []
        self._user_operation_order = []
        self._effective_measurements = 0
        self._measured_qubits = []
        self._uninitialised_qubits = []
        self._qubit_density_matrix_lookup = {}
        self._print_lines = []
        self._thread_safe_printing = thread_safe_printing
        self._fused = False
        self._sub_circuits = {}
        self._current_sub_circuit = None
        self.nodes = None
        self.qubits = None
        self._single_qubit_gate_lookup = single_qubit_gate_lookup if single_qubit_gate_lookup is not None else {}
        self._two_qubit_gate_lookup = two_qubit_gate_lookup if two_qubit_gate_lookup is not None else {}

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

    from . import _noise
    from . import _quantum_circuit_init
    from . import _superoperator
    from . import _draw
    from . import _operations
    """
        ---------------------------------------------------------------------------------------------------------
                                                    Init Methods
        ---------------------------------------------------------------------------------------------------------     
    """

    def _init_density_matrix(self):
        """ Realises init_type option 0. See class description for more info. """
        return self._quantum_circuit_init.quantum_circuit_init.init_density_matrix(self)

    def _init_density_matrix_first_qubit_ket_p(self):
        """ Realises init_type option 1. See class description for more info. """

        return self._quantum_circuit_init.quantum_circuit_init.init_density_matrix_first_qubit_ket_p(self)

    def _init_density_matrix_bell_pair_state(self, amount_qubits=8, draw=True):
        """ Realises init_type option 2. See class description for more info. """

        return self._quantum_circuit_init.quantum_circuit_init.init_density_matrix_bell_pair_state(self,
                                                                                                   amount_qubits,
                                                                                                   draw)

    def _init_density_matrix_ket_p_and_CNOTS(self):
        """ Realises init_type option 3. See class description for more info. """

        return self._quantum_circuit_init.quantum_circuit_init.init_density_matrix_ket_p_and_CNOTS(self)

    def _init_parameters_to_dict(self):
        return self._quantum_circuit_init.quantum_circuit_init.init_parameters_to_dict(self)
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
                                                SubQuantumCircuit Methods
        ---------------------------------------------------------------------------------------------------------
    """
    def start_sub_circuit(self, name, qubits, waiting_qubits=None):
        if waiting_qubits is None:
            waiting_qubits = qubits
        sub_circuit = SubQuantumCircuit(name, qubits, waiting_qubits)
        self._sub_circuits[name] = sub_circuit

        self._current_sub_circuit = sub_circuit

    def end_current_sub_circuit(self):
        if self._current_sub_circuit is not None:
            self.total_duration += self._current_sub_circuit.total_duration
        self._current_sub_circuit = None

    def define_node(self, name, qubits, electron_qubits=None):
        if self.nodes is None:
            self.nodes = {}
        if self.qubits is None:
            self.qubits = {}

        if electron_qubits is None:
            electron_qubits = []
        elif type(electron_qubits) == int:
            electron_qubits = [electron_qubits]

        self.nodes.update({name: qubits})
        for qubit in qubits:
            qubit_type = 'e' if qubit in electron_qubits else 'n'
            q = Qubit(self, qubit, qubit_type)
            self.qubits[qubit] = q

    def get_node_qubits(self, qubit):
        if self.nodes is None:
            return None
        for node_qubits in self.nodes.values():
            if qubit in node_qubits:
                return node_qubits
        return None

    def get_node_name_from_qubit(self, qubit):
        if self.nodes is None:
            return
        for key, values in self.nodes.items():
            if qubit in values:
                return key

    def apply_decoherence_to_fastest_sub_circuit(self, name_sub_circuit_1, name_sub_circuit_2):
        if self.p_dec == 0:
            sub_circuit_1 = self._sub_circuits[name_sub_circuit_1]
            self.total_duration += sub_circuit_1.total_duration
            return
        sub_circuit_1 = self._sub_circuits[name_sub_circuit_1]
        sub_circuit_2 = self._sub_circuits[name_sub_circuit_2]

        if (diff := sub_circuit_1.total_duration - sub_circuit_2.total_duration) < 0:
            times = int(math.ceil(abs(diff)/self.time_step))
            self._N_decoherence([], involved_qubits=sub_circuit_1.waiting_qubits, times=times)
            self.total_duration += sub_circuit_2.total_duration
        elif (diff := sub_circuit_2.total_duration - sub_circuit_1.total_duration) < 0:
            times = int(math.ceil(abs(diff)/self.time_step))
            self._N_decoherence([], involved_qubits=sub_circuit_2.waiting_qubits, times=times)
            self.total_duration += sub_circuit_1.total_duration
        else:
            self.total_duration += sub_circuit_1.total_duration

    def _increase_duration(self, amount, excluded_qubits, included_qubits=None, kind='idle'):
        if self._current_sub_circuit is None:
            self.total_duration += amount
        else:
            current_sub_circuit = self._current_sub_circuit
            current_sub_circuit.increase_duration(amount)

        if self.qubits is not None:
            if included_qubits is None:
                if self._current_sub_circuit is not None:
                    involved_qubits = self._current_sub_circuit.qubits
                else:
                    involved_qubits = [i for i in range(self.num_qubits)]

                excluded_qubits.extend(self._uninitialised_qubits)
                # apply waiting time to the qubits not involved in the operation.
                included_qubits = sorted(list(set(involved_qubits).difference(excluded_qubits)))

            for qubit in included_qubits:
                current_qubit = self.qubits[qubit]
                current_qubit.increase_waiting_time(amount, waiting_type=kind)

    def _update_uninitialised_qubit_register(self, qubits, update_type):
        if update_type.lower() not in ["remove", "add"]:
            raise ValueError("Type can only be 'remove' or 'add'.")

        if update_type.lower() == 'remove':
            self._uninitialised_qubits = list(set(self._uninitialised_qubits) ^ set(qubits))
        if update_type.lower() == 'add':
            self._uninitialised_qubits.extend(qubits)
            self._uninitialised_qubits = list(set(self._uninitialised_qubits))
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

    def create_bell_pairs_circuit(self, qubits, user_operation=True):
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
            self._user_operation_order.append({"create_bell_pairs_circuit": [qubits]})

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

            self._update_uninitialised_qubit_register([i, i+1], update_type='remove')

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
                self._increase_duration(bell_creation_duration)

    def create_bell_pair(self, qubit1, qubit2, noise=None, pn=None, network_noise_type=None, bell_state_type=1,
                         probabilistic=None, p_bell_success=None, bell_creation_duration=None, user_operation=True):
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

        if noise and self.p_dec > 0:
            self._N_decoherence_new([qubit1, qubit2])

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

        self._update_uninitialised_qubit_register([qubit1, qubit2], update_type="remove")
        self._increase_duration(times * bell_creation_duration, [qubit1, qubit2], kind='LDE')

        self._add_draw_operation("#", (qubit1, qubit2), noise)

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
        return sp.csr_matrix(rho)

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

        if noise and self.p_dec > 0:
            self._N_decoherence_new([tqubit])

        tqubit_density_matrix, _, relative_tqubit_index, relative_num_qubits = self._get_qubit_relative_objects(tqubit)

        one_qubit_gate = self._create_1_qubit_gate(gate,
                                                   relative_tqubit_index,
                                                   relative_num_qubits, conj=conj)
        new_density_matrix = one_qubit_gate.dot(CT(tqubit_density_matrix, one_qubit_gate))

        if noise and not self.no_single_qubit_error:
            new_density_matrix = self._N_depolarising_channel(pg, relative_tqubit_index, new_density_matrix,
                                                              relative_num_qubits)

        self._set_density_matrix(tqubit, new_density_matrix)

        self._increase_duration(gate.duration, [tqubit])

        if draw:
            self._add_draw_operation(gate, tqubit, noise)

    def _create_1_qubit_gate(self, gate, tqubit, num_qubits=None, conj=False):
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
        return self._operations.gate_operations.create_1_qubit_gate(self, gate, tqubit, num_qubits, conj)

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

        if noise and self.p_dec > 0:
            self._N_decoherence_new([cqubit, tqubit])

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

        new_density_matrix = two_qubit_gate.dot(CT(density_matrix, two_qubit_gate))

        if noise:
            new_density_matrix = self._N_two_qubit_gate(pg, rel_cqubit, rel_tqubit, new_density_matrix, num_qubits=rel_num_qubits)
        if draw:
            self._add_draw_operation(gate, (cqubit, tqubit), noise)

        self._set_density_matrix(cqubit, new_density_matrix)

        self._increase_duration(gate.duration, [cqubit, tqubit])

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
        return self._operations.gate_operations.create_2_qubit_gate(self, gate, cqubit, tqubit, num_qubits)

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
            self.single_selection(CNOT_gate, bell_qubit_1 - 1, bell_qubit_2 - 1, noise=noise, pn=pn, pm=pm, pg=pg,
                                  user_operation=user_operation)
            self.single_selection(CZ_gate, bell_qubit_1 - 1, bell_qubit_2 - 1, noise=noise, pn=pn, pm=pm, pg=pg,
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
            self.single_selection(CZ_gate, bell_qubit_1 - 1, bell_qubit_2 - 1, noise=noise, pn=pn, pm=pm, pg=pg,
                                  user_operation=user_operation)
            success = self.measure([bell_qubit_2, bell_qubit_1], noise=noise, pm=pm, user_operation=user_operation)

    """
        ---------------------------------------------------------------------------------------------------------
                                            Gate Noise Methods
        ---------------------------------------------------------------------------------------------------------  
    """

    def _N_depolarising_channel(self, pg, tqubit, density_matrix, num_qubits, times=1):
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
        return self._noise.noise_maps.N_depolarising_channel(self, pg, tqubit, density_matrix, num_qubits, times)

    def _N_two_qubit_gate(self, pg, cqubit, tqubit, density_matrix, num_qubits):
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
        return self._noise.noise_maps.N_two_qubit_gate(self, pg, cqubit, tqubit, density_matrix, num_qubits)

    def _N_network(self, density_matrix, pn, network_noise_type):
        """
            Parameters
            ----------
            density_matrix : sparse matrix
                Density matrix of the ideal Bell-pair.
            pn : float [0-1]
                Amount of network noise present in the system.
            network_noise_type: int {0, 1}
                Type of network noise that is requested
        """
        return self._noise.noise_maps.N_network(density_matrix, pn, network_noise_type)

    def _N_preparation(self, state, p_prep):
        return self._noise.noise_maps.N_preparation(state, p_prep)

    def _N_decoherence_new(self, qubits, p_dec=None):
        self._noise.decoherence.N_decoherence_new(self, qubits, p_dec)

    def _N_amplitude_damping_channel(self, tqubit, density_matrix, num_qubits, p):
        return self._noise.noise_maps.N_amplitude_damping_channel(self, tqubit, density_matrix, num_qubits, p)

    def _N_dephasing_channel(self, tqubit, density_matrix, num_qubits, p):
        return self._noise.noise_maps.N_dephasing_channel(self, tqubit, density_matrix, num_qubits, p)

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
            self._update_uninitialised_qubit_register([qubit], update_type="add")
            measurement_outcomes.append(outcome)
            # Remove the measured qubit from the system characteristics and add the operation to the draw_list
            self.num_qubits -= 1
            self.d = 2 ** self.num_qubits
            self._add_draw_operation("M_{}:{}".format(basis, outcome), qubit, noise)

            if noise and p_dec != 0:
                self._effective_measurements += (1+qubit)
                times = int(math.ceil(self.measurement_duration/self.time_step))
                self._N_decoherence([], times=times)
                self._increase_duration(self.measurement_duration)
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
        return self._operations.measurement_operations.measurement_first_qubit(density_matrix, measure, noise, pm)

    def measure(self, measure_qubits, outcome=0, uneven_parity=False, basis="X", noise=None, pm=None, pm_1=None,
                p_dec=None, probabilistic=None, basis_transformation_noise=None, user_operation=False):
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
        if pm_1 is None:
            pm_1 =self.pm_1
        if p_dec is None:
            p_dec = self.p_dec
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

            if noise and self.p_dec > 0:
                self._N_decoherence_new([qubit])

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

                if pm_1 is not None and outcome == 1:
                    pm = pm_1

                if noise:
                    new_density_matrix = (1 - pm) * new_density_matrix + pm * density_matrices[outcome_new ^ 1]

            else:
                outcome_new = outcome
                if uneven_parity and i == 0:
                    outcome_new = outcome ^ 1

                if rel_qubit == 0:
                    prob, new_density_matrix = self._measurement_first_qubit(density_matrix, measure=outcome_new,
                                                                             noise=noise, pm=pm)
                else:
                    prob, new_density_matrix = self._get_measurement_outcome_probability(rel_qubit, density_matrix,
                                                                                         outcome=outcome_new,
                                                                                         keep_qubit=False)
                    if noise:
                        _, wrong_density_matrix = self._get_measurement_outcome_probability(rel_qubit, density_matrix,
                                                                                            outcome=outcome_new ^ 1,
                                                                                            keep_qubit=False)
                        new_density_matrix = (1 - pm) * new_density_matrix + pm * wrong_density_matrix

                if prob == 0:
                    raise ValueError("Measuring a state with 0 probability cannot be dealt with. Please write"
                                     "a valid circuit.")

            if basis == "X":
                density_matrix_measured = CT(ket_p) if outcome == 0 else CT(ket_m)
                self._correct_lookup_for_measurement_any(qubit, qubits, density_matrix_measured, new_density_matrix)
            else:
                density_matrix_measured = CT(ket_0) if outcome == 0 else CT(ket_1)
                self._correct_lookup_for_measurement_any(qubit, qubits, density_matrix_measured, new_density_matrix)

            measurement_outcomes.append(outcome_new)
            self._update_uninitialised_qubit_register([qubit], update_type="add")
            self._add_draw_operation("M_{}:{}".format(basis, outcome_new), qubit, noise)

            # Please not that the decoherence is implemented after the H gate. When the H gate should be taken into
            # account for decoherence small implementation alteration is necessary.
            self._increase_duration(self.measurement_duration, [qubit])

        measurement_outcomes = iter(measurement_outcomes)
        parity_outcome = [True if i == j else False for i, j in zip(measurement_outcomes, measurement_outcomes)]
        return all(parity_outcome)

    def _get_measurement_outcome_probability(self, qubit, density_matrix, outcome, keep_qubit=True):
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
        return self._operations.measurement_operations.get_measurement_outcome_probability(qubit, density_matrix,
                                                                                           outcome, keep_qubit)

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

    @staticmethod
    def _return_QC_object(num_qubits, init):
        return QuantumCircuit(num_qubits, init)

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
        return self._superoperator.superoperator_methods.get_noiseless_density_matrix(self,
                                                                                      stabilizer_protocol,
                                                                                      proj_type,
                                                                                      measure_error,
                                                                                      save,
                                                                                      file_name)

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
            file_path = os.path.join(os.path.dirname(__file__), "_superoperator", "saved_density_matrices", file_name)
        elif kind == "qasm":
            file_name = self._file_name_from_circuit(measure_error, extension=".qasm")
            file_path = os.path.join(os.path.dirname(__file__), "_draw", file_name)
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
        return self._superoperator.superoperator_methods.all_single_qubit_gate_possibilities(self, qubits, num_qubits)

    def _fuse_equal_config_up_to_permutation(self, superoperator, proj_type):
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
        return self._superoperator.superoperator_methods.fuse_equal_config_up_to_permutation(superoperator, proj_type)

    def _remove_not_likely_configurations(self, superoperator):
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
        return self._superoperator.superoperator_methods.remove_not_likely_configurations(superoperator)

    def _print_superoperator(self, superoperator, no_color):
        """ Prints the superoperator in a clear way to the console """
        self._superoperator.superoperator_methods.print_superoperator(self, superoperator, no_color)

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
        return self._superoperator.superoperator_methods.superoperator_to_csv(self, superoperator, proj_type, file_name)

    """
        ----------------------------------------------------------------------------------------------------------
                                            Circuit drawing Methods
        ----------------------------------------------------------------------------------------------------------     
    """

    def draw_circuit(self, no_color=False, color_nodes=False):
        """ Draws the circuit that corresponds to the operation that have been applied on the system,
        up until the moment of calling. """
        legenda = "\n--- Circuit ---\n\n @: noisy Bell-pair, #: perfect Bell-pair, o: control qubit " \
                  "(with target qubit at same level), [X,Y,Z,H]: gates, M: measurement,"\
                  " {}: noisy operation (gate/measurement)\n".format("~" if no_color else colored("~", 'red'))
        init = self._draw_init(no_color)
        self._draw_gates(init, no_color)
        init[-1] += "\n\n"
        if not no_color and color_nodes:
            self._color_qubit_lines(init)
        self._print_lines.append(legenda)
        self._print_lines.extend(init)
        if not self._thread_safe_printing:
            self.print()

    def draw_circuit_latex(self, meas_error=False):
        qasm_file_name = self._create_qasm_file(meas_error)
        create_pdf_from_qasm(qasm_file_name, qasm_file_name.replace(".qasm", ".tex"))

    def _draw_init(self, no_color):
        """ Returns an array containing the visual representation of the initial state of the qubits. """
        return self._draw.draw_circuit.draw_init(self, no_color)

    def _draw_gates(self, init, no_color):
        """ Adds the visual representation of the operations applied on the qubits """
        self._draw.draw_circuit.draw_gates(self, init, no_color)

    def _color_qubit_lines(self, init):
        self._draw.draw_circuit.color_qubit_lines(self, init)

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
        return self._draw.draw_circuit_latex.create_qasm_file(self, meas_error)

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
        self._draw.draw_circuit.add_draw_operation(self, operation, qubits, noise)

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
        self._draw.draw_circuit.correct_drawing_for_n_top_qubit_additions(self, n)

    def correct_drawing_for_circuit_fusion(self, other_draw_order, num_qubits_other):
        self._draw.draw_circuit.correct_drawing_for_circuit_fusion(self, other_draw_order, num_qubits_other)

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

