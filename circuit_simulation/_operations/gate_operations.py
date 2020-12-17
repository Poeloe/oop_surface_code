from circuit_simulation.basic_operations.basic_operations import *from circuit_simulation.gates.gates import *from circuit_simulation.gates.gate import SingleQubitGate, TwoQubitGatefrom circuit_simulation.states.states import *def create_1_qubit_gate(self, gate, tqubit, num_qubits=None, conj=False, lookup=True):    """        Private method that is used to create the single-qubit gate matrix used in for example the        apply_1_qubit_gate method.        Parameters        ----------        gate : ndarray            Array of dimension 2x2, examples are the well-known pauli matrices (X, Y, Z)        tqubit : int            Integer that indicates the target qubit. Note that the qubit counting starts at            0.        num_qubits : int, optional, default=None            Determines the size of the resulting one-qubit gate matrix. If not specified, the            num_qubits known for the entire QuantumCircuit object is used        Returns        -------        1_qubit_gate : sparse matrix with dimensions equal to the density_matirx attribute            Returns a matrix with dimensions equal to the dimensions of the density matrix of            the system.    """    gate_name = None    if type(gate) == SingleQubitGate:        if (gate.name, tqubit, num_qubits) in self._single_qubit_gate_lookup.keys() and lookup:            return self._single_qubit_gate_lookup[(gate.name, tqubit, num_qubits)]        gate_name = gate.name        gate = gate.sp_matrix        if conj:            gate = gate.conj().T    if num_qubits == 1:        if not sp.issparse(gate):            gate = sp.csr_matrix(gate)        return gate    if np.array_equal(gate, I_gate.matrix):        return sp.eye(2 ** num_qubits, 2 ** num_qubits)    first_id, second_id = _create_identity_operations(tqubit, num_qubits=num_qubits)    full_gate = KP(first_id, gate, second_id)    if num_qubits > 1 and gate_name is not None:        self._single_qubit_gate_lookup[(gate_name, tqubit, num_qubits)] = full_gate    return full_gatedef _create_identity_operations(tqubit, num_qubits=None):    """        Private method that is used to efficiently create identity matrices, based on the target        qubit specified. These matrices will work on the qubits other than the target qubit        Parameters        ----------        tqubit : int            Integer that indicates the target qubit. Note that the qubit counting starts at            0.        num_qubits : int, optional, default=None            Amount of qubits that is present in the specific density matrix that the identity operations            are requested for. If not specified, the amount of qubits of the QuantumCircuit object is used        Returns        -------        first_id : sparse identity matrix            Sparse identity matrix that will work on the qubits prior to the target qubit. If the target            qubit is the first qubit, the value will be 'None'        second_id : sparse identity matrix            Sparse identity matrix that will work on the qubits following after the target qubit. If the            target qubit is the last qubit, the value will be 'None'    """    first_id = None    second_id = None    if tqubit == 0:        second_id = sp.eye(2 ** (num_qubits - 1 - tqubit), 2 ** (num_qubits - 1 - tqubit))    elif tqubit == num_qubits - 1:        first_id = sp.eye(2 ** tqubit, 2 ** tqubit)    else:        first_id = sp.eye(2 ** tqubit, 2 ** tqubit)        second_id = sp.eye(2 ** (num_qubits - 1 - tqubit), 2 ** (num_qubits - 1 - tqubit))    return first_id, second_iddef create_2_qubit_gate(self, gate, cqubit, tqubit, num_qubits=None):    """    Create a controlled gate matrix for the density matrix according to the control and target qubits given.    This is done by            1.  first taking the Kronecker Product the identity matrix as many times as there are qubits                present in the system.            2.  Then for the two sub gates formed on the place of the control qubit the identity matrix                is replaced for a |0><0| and |1><1| matrix respectively.            3.  Then for the gate_2 the identity matrix on the target qubit index is replaced with the wanted gate.    So for creating a CNOT gate with the control on the 2nd qubit and target on the first qubit on a system with 3    qubits one will get:            1. I#I#I + I#I#I + I#I#I + I#I#I            2. I#|0><0|#I + I#|1><1|#I + 0#|0><1|#I + 0#|1><0|#I            3. I#|0><0|#I + X_t#|1><1|#I + 0#|0><1|#I + 0#|1><0|#I    (In which '#' is the Kronecker Product, and '0' is the zero matrix)    (https://quantumcomputing.stackexchange.com/questions/4252/    how-to-derive-the-cnot-matrix-for-a-3-qbit-system-where-the-control-target-qbi and    https://quantumcomputing.stackexchange.com/questions/9181/swap-gate-on-2-qubits-in-3-entangled-qubit-system)    The 'create_component_2_qubit_gate' method defined within creates one of the 4 components that is shown in    step 3 above. Thus 'first_part = create_component_2_qubit_gate(CT(ket_0), zero_state_matrix)' creates the first    component namely I#|0><0|#I in case of the CNOT mentioned.    Parameters    ----------    gate : TwoQubitGate object        TwoQubitGate object representing a 2-qubit gate    cqubit : int        Integer that indicates the control qubit. Note that the qubit counting starts at 0.    tqubit : int        Integer that indicates the target qubit. Note that the qubit counting starts at 0.    num_qubits : int, optional, default=None        Determines the size of the resulting two-qubit gate matrix. If not specified, the        num_qubits known for the entire QuantumCircuit object is used    """    if cqubit == tqubit:        raise ValueError("Control qubit cannot be the same as the target qubit!")    if type(gate) == TwoQubitGate:        if (gate.name, cqubit, tqubit, num_qubits) in self._two_qubit_gate_lookup.keys():            return self._two_qubit_gate_lookup[(gate.name, cqubit, tqubit, num_qubits)]    def create_component_2_qubit_gate(control_qubit_matrix, target_qubit_matrix):        # Initialise the only identity case with on the place of the control qubit the identity replaced        # with the specified control_qubit_matrix        control_gate = self._create_1_qubit_gate(control_qubit_matrix, cqubit, num_qubits=num_qubits)        # Initialise the only identity case with on the place of the target qubit the identity replaced        # with the specified target_qubit_matrix        if not np.array_equal(target_qubit_matrix, I_gate):            target_gate = self._create_1_qubit_gate(target_qubit_matrix, tqubit, num_qubits=num_qubits)            # Matrix multiply the two cases to obtain the total gate            return target_gate.dot(control_gate)        return control_gate    one_state_matrix = gate.sp_matrix if type(gate) == SingleQubitGate else gate.upper_left_matrix    zero_state_matrix = I_gate.sp_matrix if type(gate) == SingleQubitGate else gate.lower_right_matrix    first_part = create_component_2_qubit_gate(CT(ket_0), zero_state_matrix)    second_part = create_component_2_qubit_gate(CT(ket_1), one_state_matrix)    full_gate = first_part + second_part    if type(gate) == TwoQubitGate and not gate.is_cntrl_gate:        third_part = create_component_2_qubit_gate(CT(ket_0, ket_1), gate.upper_right_matrix)        fourth_part = create_component_2_qubit_gate(CT(ket_1, ket_0), gate.lower_left_matrix)        full_gate = first_part + second_part + third_part + fourth_part    if num_qubits > 3 and type(gate) == TwoQubitGate:        self._two_qubit_gate_lookup[(gate.name, cqubit, tqubit, num_qubits)] = full_gate    return full_gatedef efficient_SWAP(self, qubit_1, qubit_2, noise, pg, draw):    """        SWAP gate that can be used when performing a one-way swap i.e., when one qubit is uninitialised. It swaps the        qubit indices instead of applying the matrix operation.        Parameters        ----------        self : QuantumCircuit            The QuantumCircuit object to which in which the SWAP gate is applied        qubit_1 : int            qubit to swap        qubit_2 : int            qubit to swap        noise : bool            If swap is noisy        pg : float            Error probability of the swap gate        draw : bool            Whether the gate should be drawn in the circuit diagram    """    # If pulse sequence is taken into account, the SWAP gate must wait for the right point in the sequence    self._wait_for_refocus([qubit_1, qubit_2])    if noise and self.decoherence:        self._N_decoherence([qubit_1, qubit_2])    density_matrix_1, qubits_1, rel_qubit_1, rel_num_qubits_1 = self._get_qubit_relative_objects(qubit_1)    density_matrix_2, qubits_2, rel_qubit_2, rel_num_qubits_2 = self._get_qubit_relative_objects(qubit_2)    if noise:        if qubit_1 not in self._uninitialised_qubits:            density_matrix_1 = self._N_depolarising_channel(pg, rel_qubit_1, density_matrix_1, rel_num_qubits_1,                                                            times=3, SWAP=True)        if qubit_2 not in self._uninitialised_qubits:            density_matrix_2 = self._N_depolarising_channel(pg, rel_qubit_2, density_matrix_2, rel_num_qubits_2,                                                            times=3, SWAP=True)    qubits_1[rel_qubit_1] = qubit_2    qubits_2[rel_qubit_2] = qubit_1    for qubit in qubits_2:        self._qubit_density_matrix_lookup[qubit] = (density_matrix_2, qubits_2)    for qubit in qubits_1:        self._qubit_density_matrix_lookup[qubit] = (density_matrix_1, qubits_1)    self._update_uninitialised_qubit_register([qubit_1, qubit_2], update_type="swap")    # Efficient SWAP is performed with two CNOT gates, therefore the duration of twice a CNOT    self._increase_duration(CNOT_gate.duration * 3, [qubit_1, qubit_2])    if draw:        self._add_draw_operation(SWAP_gate, (qubit_1, qubit_2), noise)def handle_electron_is_target_qubit(self, tqubit, cqubit, *, noise, decoherence, draw, gate=None):    """    Handles the case that the electron qubit is the target qubit of an controlled gate. In this case in reality    Hadamard gates are applied both before and after the controlled gate and the control is then on the electron    qubit. The Hadamard on the nuclear qubit takes longer, such that the electron qubit should experience    decoherence. it thus for a CNOT looks like:    n ---O---       n ---H---X---H---                =    e ---X---       e ---H---O---H---    """    qubits = [tqubit, cqubit]    self._increase_duration(H_gate.duration, excluded_qubits=[cqubit])    self._N_decoherence(qubits) if noise and decoherence else None    if draw:        noise_h = not self.no_single_qubit_error and noise        self._add_draw_operation(H_gate, tqubit, noise_h), self._add_draw_operation(H_gate, cqubit, noise_h)        self._add_draw_operation(gate, [cqubit, tqubit], noise)        self._add_draw_operation(H_gate, tqubit, noise_h), self._add_draw_operation(H_gate, cqubit, noise_h)    return Falsedef determine_node_qubit_from_string(self, qubit_string: str):    """        Determines which qubit from the node is requested by the given string.        Parameters        ----------        self : QuantumCircuit            The QuantumCircuit object of which the qubit index should be returned        qubit_string : str            The string that represents a qubit in the node. String format is of type: "{node_name}-{qubit_name}"    """    if type(qubit_string) == int:        return qubit_string    # Raise error if no nodes are defined    if not self.nodes:        raise ValueError("There are no nodes defined. Qubits can therefore not be referred to with a string")    node_name, qubit_name = qubit_string.split('-')    # Raise error if node is not recognised    if node_name not in self.nodes:        raise ValueError("Node {} not recognised. Known nodes are {}".format(node_name, self.nodes.keys()))    node = self.nodes[node_name]    # Determine required qubit from name    if ('electron' in qubit_name or 'e' == qubit_name) and node.electron_qubits:        return node.electron_qubits[0]    elif 'data' in qubit_name and node.data_qubits:        return node.data_qubits    elif qubit_name.isdigit():        qubit_index = int(qubit_name)        try:            return node.qubits[int(qubit_name)]        except IndexError:            raise ValueError("Passed node index is out of range. You passed {}, while the node consist  of {} qubits"                             .format(qubit_index, len(node.qubits)))    else:        raise ValueError("Passed qubit node reference, '{}', not understood or known for the node.".format(qubit_name))