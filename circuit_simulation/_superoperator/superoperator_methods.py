import os
import scipy.sparse as sp
from circuit_simulation.states.states import *
from circuit_simulation.gates.gates import *
from oopsc.superoperator.superoperator import SuperoperatorElement
from itertools import combinations, product, combinations_with_replacement
from termcolor import colored
import pandas as pd


def get_noiseless_density_matrix(self, stabilizer_protocol, proj_type, measure_error=False, save=True,
                                  file_name=None, qubits=None):
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
        return _noiseless_stabilizer_protocol_density_matrix(self, proj_type, measure_error)
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
        if operation == "SWAP":
            qc_noiseless.SWAP(parameters[0], parameters[1])
        elif operation == "apply_1_qubit_gate":
            qc_noiseless.apply_1_qubit_gate(parameters[0], parameters[1])
        elif operation == "apply_2_qubit_gate":
            qc_noiseless.apply_2_qubit_gate(parameters[0], parameters[1], parameters[2])
        elif operation == "measure":
            uneven_parity = True if measure_error and i == (len(self._user_operation_order) - 1) else False
            qc_noiseless.measure(parameters[0], parameters[1], uneven_parity, probabilistic=False)

    qc_noiseless.draw_circuit()

    if save:
        sp.save_npz(file_name, qc_noiseless.get_combined_density_matrix(qubits)[0])

    return qc_noiseless.get_combined_density_matrix(qubits)[0]


def _noiseless_stabilizer_protocol_density_matrix(self, proj_type, measure_error):
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
    qc = self._return_QC_object(9, 2)
    qc.set_qubit_states({0: ket_p})
    gate = Z_gate if proj_type == "Z" else X_gate
    for i in range(1, qc.num_qubits, 2):
        qc.apply_2_qubit_gate(gate, 0, i)

    qc.measure([0], outcome=0 if not measure_error else 1)

    return qc.get_combined_density_matrix([1])[0]


def all_single_qubit_gate_possibilities(self, qubits, num_qubits):
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
            gates.append({operation.representation: self._create_1_qubit_gate(operation, rel_qubit,
                                                                              num_qubits=num_qubits)})
        gate_combinations.append(gates)

    return list(product(*gate_combinations))


def fuse_equal_config_up_to_permutation(superoperator, proj_type):
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
        # say 'Z' is the proj_type, then IIZZ with ZZII and ZIIZ with IZZI are degenerate. Sum is halved
        if error_array.count("I") == error_array.count(proj_type):
            p = sum([el.p for el in equal_supop_el]) / 2
        sorted_superoperator.append(SuperoperatorElement(p, lie, error_array))

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
        if supop_el_a.probability_lie_equals(supop_el_b):
            if supop_el_a.error_array.count("I") > supop_el_b.error_array.count("I") \
                    and supop_el_b in superoperator:
                superoperator.remove(supop_el_b)
            elif supop_el_a.error_array.count("I") < supop_el_b.error_array.count("I") \
                    and supop_el_a in superoperator:
                superoperator.remove(supop_el_a)

    return superoperator


def print_superoperator(self, superoperator, no_color):
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
    self._print_lines.append("\nTotal lde attempts: {}\n".format(self._total_lde_attempts))
    self._print_lines.append("\n---- End of Superoperator ----\n")

    if not self._thread_safe_printing:
        self.print()


def superoperator_to_csv(self, superoperator, proj_type, file_name=None, use_exact_path=False):
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
    path_to_file = self._absolute_file_path_from_circuit(measure_error=False, kind="so")
    if file_name is None:
        self._print_lines.append("\nFile name was created manually and is: {}\n".format(path_to_file))
    elif use_exact_path:
        path_to_file = file_name + ".csv"
    else:
        path_to_file = os.path.join(path_to_file.rpartition(os.sep)[0], file_name.replace(os.sep, "") + ".csv")
        self._print_lines.append("\nCSV file has been saved at: {}\n".format(path_to_file))

    error_index = ["".join(combi) for combi in (combinations_with_replacement('IXYZ', 4))]
    error_index.extend(error_index)
    lie_index = [False if i / (len(error_index) / 2) < 1 else True for i, _ in enumerate(error_index)]

    index = pd.MultiIndex.from_arrays([error_index, lie_index], names=['error_config', 'lie'])

    if os.path.exists(path_to_file):
        data = pd.read_csv(path_to_file, sep=';', index_col=[0, 1])
    else:
        columns = ['p', 's', 'pg', 'pm', 'pn', 'p_dec', 'ts', 'p_bell', 'bell_dur', 'meas_dur', 'written_to',
                   'lde_attempts', 'total_duration']
        data = pd.DataFrame(0., index=index, columns=columns)
        data.iloc[0, data.columns.get_loc('pg')] = self.pg
        data.iloc[0, data.columns.get_loc('pm')] = self.pm
        data.iloc[0, data.columns.get_loc('pn')] = self.pn
        data.iloc[0, data.columns.get_loc('p_dec')] = int(self.decoherence)
        data.iloc[0, data.columns.get_loc('p_bell')] = self.p_bell_success
        data.iloc[0, data.columns.get_loc('bell_dur')] = self.bell_creation_duration
        data.iloc[0, data.columns.get_loc('meas_dur')] = self.measurement_duration
        data.iloc[0, data.columns.get_loc('lde_attempts')] = self._total_lde_attempts
        data.iloc[0, data.columns.get_loc('total_duration')] = self.total_duration

    stab_type = 'p' if proj_type == "Z" else 's'
    opp_stab = 's' if proj_type == "Z" else 'p'

    for supop_el in superoperator:
        error_array = "".join(sorted(supop_el.error_array))
        current_index = (error_array, supop_el.lie)
        if current_index in data.index:
            current_value_stab = data.at[(error_array, supop_el.lie), stab_type]
            new_value_stab = (current_value_stab + supop_el.p) / 2 if current_value_stab != 0. else supop_el.p
            data.at[current_index, stab_type] = new_value_stab
        else:
            data.loc[current_index, stab_type] = supop_el.p

        # When Z and X errors are equally likely, symmetry between proj_type and only H gate difference in
        # error_array
        error_array = "".join(sorted(error_array.translate(str.maketrans({'X': 'Z', 'Z': 'X'}))))
        current_index_opp = (error_array, supop_el.lie)
        if current_index_opp in data.index:
            current_value_opp_stab = data.at[(error_array, supop_el.lie), opp_stab]
            new_value_opp_stab = (current_value_opp_stab + supop_el.p) / 2 if current_value_opp_stab != 0. \
                else supop_el.p
            data.at[current_index_opp, opp_stab] = new_value_opp_stab
        else:
            data.loc[current_index_opp, opp_stab] = supop_el.p

    if 'total_duration' in data:
        data.iloc[0, data.columns.get_loc("total_duration")] = (data.iloc[0, data.columns.get_loc("total_duration")] +
                                                                self.total_duration) / 2
    else:
        data['total_duration'] = 0
        data.iloc[0, data.columns.get_loc("total_duration")] = self.total_duration
    if 'lde_attempts' in data:
        data.iloc[0, data.columns.get_loc("lde_attempts")] = (data.iloc[0, data.columns.get_loc("lde_attempts")] +
                                                              self._total_lde_attempts) / 2
    else:
        data['lde_attempts'] = 0
        data.iloc[0, data.columns.get_loc("lde_attempts")] = self._total_lde_attempts
    if 'written_to' in data:
        data.iloc[0, data.columns.get_loc("written_to")] += 1.0
    else:
        data['written_to'] = 0
        data.iloc[0, data.columns.get_loc("written_to")] = 1

    # Remove rows that contain only zero probability
    data = data[(data.T != 0).any()]

    data.to_csv(path_to_file, sep=';')
    if not self._thread_safe_printing:
        self.print()
