import os
import scipy.sparse as sp
from circuit_simulation.states.states import *
from circuit_simulation.gates.gates import *
from circuit_simulation.basic_operations.basic_operations import CT
from circuit_simulation._superoperator.superoperator import SuperoperatorElement
from itertools import combinations, product, combinations_with_replacement
from circuit_simulation.termcolor.termcolor import colored
from circuit_simulation.basic_operations.basic_operations import fidelity_elementwise
import pandas as pd
import math


def get_noiseless_density_matrix(self, stabilizer_protocol=False, proj_type=None, measure_error=False, save=False,
                                  file_name=None, qubits=None, idle_data_qubit=None):
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
    if self.cut_off_time_reached:
        qc = self._return_QC_object(len(qubits)*2, 0)
        qc._init_density_matrix_maximally_entangled_state(amount_qubits=len(qubits)*2)
        return qc.get_combined_density_matrix([0, 2, 4, 6])[0]
    if stabilizer_protocol and not idle_data_qubit:
        return _noiseless_stabilizer_protocol_density_matrix(self, proj_type, qubits, measure_error)
    if idle_data_qubit:
        init_type = 2 if idle_data_qubit == 4 else 5
        return self._return_QC_object(idle_data_qubit*2, init_type).total_density_matrix()[0]

    if file_name is None:
        file_name = self._absolute_file_path_from_circuit(measure_error)

    # Check if the noiseless system has been calculated before
    if os.path.exists(file_name):
        return sp.load_npz(file_name)

    # Get the initial parameters of the current QuantumCircuit object
    init_type = self._init_parameters['init_type']
    num_qubits = self._init_parameters['num_qubits']

    qc_noiseless = self._return_QC_object(num_qubits, init_type)

    for i, user_operation in enumerate(self._user_operation_order):
        operation = list(user_operation.keys())[0]
        parameters = list(user_operation.values())[0]

        if operation == "create_bell_pair":
            qc_noiseless.create_bell_pair(parameters[0], parameters[1], network_noise_type=parameters[4],
                                          bell_state_type=parameters[5])
        elif operation == "SWAP":
            qc_noiseless.SWAP(parameters[0], parameters[1])
        elif operation == "apply_gate":
            qc_noiseless.apply_gate(parameters[0], parameters[1], parameters[2])
        elif operation == "measure":
            uneven_parity = True if measure_error and i == (len(self._user_operation_order) - 1) else False
            qc_noiseless.measure(parameters[0], parameters[1], uneven_parity, probabilistic=True)
        else:
            method = getattr(qc_noiseless, operation)
            method(*parameters)

    # qc_noiseless.draw_circuit()

    if save:
        sp.save_npz(file_name, qc_noiseless.get_combined_density_matrix(qubits)[0])

    return qc_noiseless.get_combined_density_matrix(qubits)[0]


def _noiseless_stabilizer_protocol_density_matrix(self, proj_type, qubits, measure_error):
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
    num_qubits = len(qubits)*2 + 1
    qc = self._return_QC_object(num_qubits=num_qubits, init=0)
    qc._init_density_matrix_maximally_entangled_state(amount_qubits=num_qubits-1)
    qc.set_qubit_states({0: ket_p})
    gate = CZ_gate if proj_type == "Z" else CNOT_gate
    for i in range(1, qc.num_qubits, 2):
        qc.apply_gate(gate, cqubit=0, tqubit=i)

    qc.measure([0], outcome=0 if not measure_error else 1)

    return qc.get_combined_density_matrix([1])[0]


def get_state_fidelity(self, qubits=None, compare_matrix=None, set_ghz_fidelity=True):
    if qubits is None:
        qubits = list(set(self.qubits.keys()).difference(self._uninitialised_qubits).difference(self.data_qubits))

    if compare_matrix is None and set_ghz_fidelity:
        # Create ghz state with the weight equal to the amount of ghz qubits
        ghz_state = sp.lil_matrix((2**len(qubits), 2**len(qubits)))
        ghz_state[0, 0] = 1/2; ghz_state[0, 2**len(qubits)-1] = 1/2
        ghz_state[2**len(qubits)-1, 0] = 1/2; ghz_state[2**len(qubits)-1, 2**len(qubits)-1] = 1/2

        compare_matrix = ghz_state

    elif compare_matrix is None:
        compare_matrix = get_noiseless_density_matrix(self, qubits=qubits)

    noisy_matrix = self.get_combined_density_matrix(qubits)[0]
    fidelity = fidelity_elementwise(compare_matrix, noisy_matrix)

    if set_ghz_fidelity:
        self.ghz_fidelity = fidelity

    return fidelity


def create_superoperator_decomposition(self, qubits, qubits_matrix):
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
    # If Kraus decomposition is already created, return this
    if self._superoperator_decomposition is not None:
        return self._superoperator_decomposition

    num_qubits = len(qubits_matrix)
    # Create for each data qubit the full matrix size Pauli operator
    operations = [X_gate, Y_gate, Z_gate, I_gate]
    pauli_operators_per_qubit = []

    for qubit in qubits:
        rel_qubit = qubits_matrix.index(qubit)
        gates = []
        for operation in operations:
            gates.append({operation.representation: self._create_1_qubit_gate(operation, rel_qubit,
                                                                              num_qubits=num_qubits)})
        pauli_operators_per_qubit.append(gates)

    # Create all possible combinations of Pauli operators on the data qubits
    decomposition = list(product(*pauli_operators_per_qubit))

    # Post-process the decomposition data, such that the combination of gates is multiplied
    superoperator_decomposition = {}
    for element in decomposition:
        kraus_operator = "".join([list(el.keys())[0] for el in element])
        matrix = np.prod([list(el.values())[0] for el in element])
        superoperator_decomposition[kraus_operator] = matrix

    # Save the decomposition to the QC object, such that it only has to be constructed once for multiple iterations
    self._superoperator_decomposition = superoperator_decomposition
    return self._superoperator_decomposition


def get_error_density_matrices(self, kraus_operator, stabilizer_protocol, noiseless_density_matrix,
                               measerror_density_matrix, error_matrix):

    if kraus_operator in self._error_density_matrix_lookup and stabilizer_protocol:
        return self._error_density_matrix_lookup[kraus_operator]

    error_density_matrix = error_matrix * CT(noiseless_density_matrix, error_matrix)
    me_error_density_matrix = error_matrix * CT(measerror_density_matrix, error_matrix)

    self._error_density_matrix_lookup[kraus_operator] = (error_density_matrix, me_error_density_matrix)

    return error_density_matrix, me_error_density_matrix


def fuse_equal_config_up_to_permutation(superoperator):
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
    sorted_superoperator = []
    supop_el_dict = {}

    # Create dict with SuperoperatorElements equal in lie and error_array as items
    for supop_el in superoperator:
        key = str(supop_el.lie) + str(sorted(supop_el.error_array))
        if key not in supop_el_dict.keys():
            supop_el_dict[key] = [supop_el]
        else:
            supop_el_dict[key].append(supop_el)

    # For each grouped SuperoperatorElements in the created dict, sum the probability (take degenerate into account)
    for equal_supop_el in supop_el_dict.values():
        lie = equal_supop_el[0].lie
        error_array = sorted(equal_supop_el[0].error_array)
        p = sum([el.p for el in equal_supop_el])
        error_density_matrix = [el.error_density_matrix for el in equal_supop_el if el.error_array == error_array]
        error_density_matrix = error_density_matrix[0] if len(error_density_matrix) > 1 \
            else equal_supop_el[0].error_density_matrix
        degenerate_configurations = {"".join(el.error_array): el.error_density_matrix for el in equal_supop_el}

        # If permutations are degenerate, like IIZZ with ZZII and ZIIZ with IZZI this should be taken into account.
        # Sum is halved (considering degenerate pairs)
        amount_degenerate = [el1.error_density_matrix_equals(el2) for el1, el2 in combinations(equal_supop_el,
                                                                                               2)].count(True)
        if amount_degenerate > 0:
            p /= 2 if amount_degenerate == int(len(equal_supop_el)/2) else len(equal_supop_el)

        sorted_superoperator.append(SuperoperatorElement(p, lie, error_array, error_density_matrix,
                                                         degenerate_configurations))

    return sorted_superoperator


def remove_not_likely_configurations(superoperator):
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
        if supop_el_a.any_error_density_matrix_equals(supop_el_b):
            if supop_el_a.error_array.count("I") > supop_el_b.error_array.count("I") and supop_el_b in superoperator:
                supop_el_a.fused_configs.update(supop_el_b.fused_configs)
                superoperator.remove(supop_el_b)
            elif supop_el_a.error_array.count("I") < supop_el_b.error_array.count("I") and supop_el_a in superoperator:
                supop_el_b.fused_configs.update(supop_el_a.fused_configs)
                superoperator.remove(supop_el_a)
            elif supop_el_a in superoperator and supop_el_b in superoperator:
                supop_el_b.p /= 2
                supop_el_a.p /= 2

    return superoperator


def print_superoperator(self, superoperator, no_color):
    """ Prints the superoperator in a clear way to the console """
    self._print_lines.append("\n---- Superoperator ----\n")

    total = sum([supop_el.p for supop_el in superoperator])
    for supop_el in sorted(superoperator):
        probability = supop_el.p
        self.append_print_lines("\nProbability: {}".format(probability))
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
        self.append_print_lines("\n{} - {}".format(config, me))
    self.append_print_lines("\n\nSum of the probabilities is: {}\n".format(total))
    self.append_print_lines("\nTotal lde attempts: {}\n".format(self._total_lde_attempts))
    self.append_print_lines("\nAverage lde attempts per successful lde: {}\n".format(
        self._total_lde_attempts/self._total_succeeded_lde)) if self._total_succeeded_lde !=0 else None

    self.append_print_lines("\n---- End of Superoperator ----\n")

    if not self._thread_safe_printing:
        self.print()


def superoperator_to_dataframe(self, superoperator, proj_type, file_name=None, use_exact_path=False,
                               protocol_name=None, qubit_order=None):
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
    if file_name:
        if use_exact_path:
            file_name = file_name if not self.cut_off_time_reached else file_name + "_failed"
            path_to_file = file_name + ".csv"
        else:
            path_to_file = self._absolute_file_path_from_circuit(measure_error=False, kind="so")
            file_name = file_name if not self.cut_off_time_reached else file_name + "_failed"
            path_to_file = os.path.join(path_to_file.rpartition(os.sep)[0], file_name.replace(os.sep, "") + ".csv")

    if file_name and os.path.exists(path_to_file):
        data = pd.read_csv(path_to_file, sep=';', index_col=[0, 1])
    else:
        data = _create_new_superoperator_dataframe(self, protocol_name, qubit_order)

    stab_type = 'p' if proj_type == "Z" else 's'
    opp_stab = 's' if proj_type == "Z" else 'p'
    written_to = data.iloc[0, data.columns.get_loc("written_to")]

    for supop_el in superoperator:
        # When Z and X errors are equally likely, symmetry between proj_type and only H gate difference in error_array
        error_array_str = "".join((supop_el.error_array))
        opp_error_array_str = "".join((error_array_str.translate(str.maketrans({'X': 'Z', 'Z': 'X'}))))

        for error_array, current_stab_type in zip([error_array_str, opp_error_array_str], [stab_type, opp_stab]):
            current_index = (error_array, supop_el.lie)
            current_value_stab = data.loc[(error_array, supop_el.lie), current_stab_type]

            data.loc[current_index, current_stab_type] = (current_value_stab * written_to + supop_el.p)/(written_to + 1)

    _create_column_if_not_present(data, ['total_duration', 'total_lde_attempts', 'ghz_fidelity', 'avg_duration',
                                         'avg_lde_attempts'])

    # Increase the amount of writes to the file
    data.iloc[0, data.columns.get_loc("written_to")] += 1.0

    # Update values based on amounts of writes to the file
    _update_totals_and_averages(self, data, ['total_duration', '_total_lde_attempts'], ['ghz_fidelity'])

    # Remove rows that contain only zero probability
    data = data[(data.T.applymap(lambda x: x != 0 and x is not None and not pd.isna(x))).any()]

    if file_name:
        data.to_csv(path_to_file, sep=';')
        self._print_lines.append("\nCSV file has been saved at: {}\n".format(path_to_file))

    if not self._thread_safe_printing:
        self.print()

    return data


def _create_column_if_not_present(data, columns):
    for column_name in columns:
        if column_name not in data:
            data.iloc[0, data.columns.get_loc(column_name)] = 0
            data.iloc[1:, data.columns.get_loc(column_name)] = None


def _update_totals_and_averages(self, data, totals, averages):
    written_to = data.iloc[0, data.columns.get_loc("written_to")]
    for total in totals:
        value = getattr(self, total)
        avg = total.strip('_').replace("total", 'avg')
        data.iloc[0, data.columns.get_loc(total.strip('_'))] += value
        data.iloc[0, data.columns.get_loc(avg)] = data.iloc[0, data.columns.get_loc(total.strip('_'))] / written_to

    for value_name in averages:
        value = getattr(self, value_name) if getattr(self, value_name) is not None else 0
        average_value = (value + data.iloc[0, data.columns.get_loc(value_name)] * (written_to - 1)) / written_to
        data.iloc[0, data.columns.get_loc(value_name)] = average_value


def _create_new_superoperator_dataframe(self, protocol_name, qubit_order):
    num_qubits = len(qubit_order)
    error_index = ["".join(i) for i in product("IXYZ", repeat=num_qubits)]
    lie_index = [False, True]
    index = pd.MultiIndex.from_product([error_index, lie_index], names=['error_config', 'lie'])
    columns = ['p', 's']
    circuit_results = ['written_to', 'total_lde_attempts', 'avg_lde_attempts', 'total_duration', 'avg_duration',
                       'ghz_fidelity', 'protocol_name', 'qubit_order']
    circuit_properties = ['pg', 'pm', 'pm_1', 'pn', 'decoherence', 'p_bell_success', 'pulse_duration',
                          'network_noise_type', 'no_single_qubit_error', 'basis_transformation_noise',
                          'cut_off_time', 'probabilistic', 'fixed_lde_attempts']
    columns.extend(circuit_results)
    columns.extend(circuit_properties)
    dataframe = pd.DataFrame(0., index=index, columns=columns)
    for prop in circuit_properties:
        prop_value = getattr(self, prop) if getattr(self, prop) is not None else "None"
        dataframe.iloc[0, dataframe.columns.get_loc(prop)] = prop_value
        dataframe.iloc[1:, dataframe.columns.get_loc(prop)] = None
    for result in circuit_results:
        value = 0
        if result == 'protocol_name':
            value = protocol_name
        elif result == 'qubit_order':
            value = str(qubit_order)
        elif result == 'ghz_fidelity' and self.ghz_fidelity is not None:
            value = self.ghz_fidelity

        dataframe.iloc[0, dataframe.columns.get_loc(result)] = value
        dataframe.iloc[1:, dataframe.columns.get_loc(result)] = None

    return dataframe
