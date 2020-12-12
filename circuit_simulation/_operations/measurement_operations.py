from circuit_simulation.basic_operations.basic_operations import *
import copy


def measurement_first_qubit(density_matrix, measure=0, noise=None, pm=0., no_normalisation=False):
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

    prob, temp_density_matrix = _get_prob_and_matrix_after_measurement(density_matrix_0, density_matrix_1,
                                                                       measure=measure, noise=noise, pm=pm,
                                                                       no_normalisation=no_normalisation)

    return prob, temp_density_matrix


def measure_arbitrary_qubit(self, density_matrix, num_qubits, qubit, measure=0, noise=None, pm=None,
                            no_normalisation=False):
    measurement_operator_0 = sp.csr_matrix([[1, 0], [0, 0]])
    measurement_operator_1 = sp.csr_matrix([[0, 0], [0, 1]])

    full_operator_0 = self.create_1_qubit_gate(measurement_operator_0, qubit, num_qubits=num_qubits)
    full_operator_1 = self.create_1_qubit_gate(measurement_operator_1, qubit, num_qubits=num_qubits)

    density_matrix_0 = full_operator_0 * CT(density_matrix, full_operator_0)
    density_matrix_1 = full_operator_1 * CT(density_matrix, full_operator_1)

    prob, temp_density_matrix = _get_prob_and_matrix_after_measurement(density_matrix_0, density_matrix_1,
                                                                       measure=measure, noise=noise, pm=pm,
                                                                       no_normalisation=no_normalisation)

    return prob, temp_density_matrix


def _get_prob_and_matrix_after_measurement(density_matrix_0, density_matrix_1, measure, no_normalisation, noise, pm):
    if measure == 0 and noise:
        temp_density_matrix = (1 - pm) * density_matrix_0 + pm * density_matrix_1
    elif noise:
        temp_density_matrix = (1 - pm) * density_matrix_1 + pm * density_matrix_0
    elif measure == 0:
        temp_density_matrix = density_matrix_0
    else:
        temp_density_matrix = density_matrix_1
    prob = trace(temp_density_matrix)
    if prob != 0 and not no_normalisation:
        temp_density_matrix = temp_density_matrix / prob
    return prob, temp_density_matrix


def _measure_arbitrary_qubit(qubit, density_matrix, outcome, keep_qubit=False):
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
    # Dimension of the chess pattern tile in the density matrix that contain values of the to be measure qubit
    dim_tile = int(d / (2 ** (qubit + 1)))

    # Get the non zero rows and columns of the sparse matrix.
    non_zero_rows = density_matrix.nonzero()[0]
    non_zero_columns = density_matrix.nonzero()[1]

    # Keeps the qubit in the density matrix. It is probably faster to use the measurement operators method for this
    if keep_qubit:
        new_density_matrix = sp.lil_matrix(copy.copy(density_matrix))
        start = 0 if outcome == 1 else dim_tile
        rows_columns_to_zero = [i+j for i in range(start, d, dim_tile * 2)
                                for j in range(dim_tile)]
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
        # Create the new matrix that will be the resulting density matrix after the measurement (without the
        # measured qubit) which will be half the size of the original density matrix
        new_density_matrix = sp.lil_matrix((int(d/2), int(d/2)), dtype=density_matrix.dtype)

        # Start of the chess pattern. If 0 outcome is measured, one needs to keep the 'white' tiles of the chess
        # board. When 1 is measured one needs the 'black' tiles.
        start = 0 if outcome == 0 else dim_tile

        # Find the rows in the density matrix that are kept, keeping the chess pattern and dimension of the tiles
        # into account
        surviving_columns_rows = [i+j for i in range(start, d, dim_tile * 2) for j in range(dim_tile)]

        # Get the rows and columns, that contain non-zero elements, that intersect with the rows and columns of the
        # tiles that should be kept
        non_zero_rows_unique = np.array(list(set(surviving_columns_rows).intersection(non_zero_rows)))
        non_zero_columns_unique = np.array(list(set(surviving_columns_rows).intersection(non_zero_columns)))

        if non_zero_columns_unique.size != 0:
            for row in non_zero_rows_unique:
                # Each row and column should shift, since the density matrix is half the size. This collapse is equal
                # for outcome=0 to removing all the black tiles of the chess pattern and connecting the white tiles
                # after. For outcome=1, this is the same but then removing the white tiles.
                multiplier = divmod(row, dim_tile*2)[0] if outcome == 0 else divmod(row, dim_tile*2)[0] + 1
                new_row = row - (multiplier * dim_tile) if multiplier > 0 else row

                column_indices = [i for i, e in enumerate(non_zero_rows) if e == row]
                valid_columns = [c for c in non_zero_columns[column_indices] if c in surviving_columns_rows]
                new_columns = []
                for column in valid_columns:
                    multiplier = (divmod(column, dim_tile*2)[0] if outcome == 0 else divmod(column, dim_tile*2)[0] + 1)
                    new_columns.append(column - (multiplier * dim_tile) if multiplier > 0 else column)

                new_density_matrix[new_row, new_columns] = density_matrix[row, valid_columns]

    prob = trace(new_density_matrix)
    new_density_matrix = new_density_matrix / trace(new_density_matrix)

    return prob, new_density_matrix


def measurement_by_diagonalising(self, qubit, density_matrix, measure=0, eigenval=None, eigenvec=None):
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