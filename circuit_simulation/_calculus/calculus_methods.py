from scipy.linalg import eig, eigh
import scipy.sparse as sp
import copy
from circuit_simulation.states.states import *


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